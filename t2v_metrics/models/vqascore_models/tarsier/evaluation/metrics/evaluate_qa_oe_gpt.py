# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import numpy as np
import ast
import time
from typing import List, Dict
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
import func_timeout
from func_timeout import func_set_timeout
import os

from tools.gpt_api import azure_gpt4_client
from tools.color import Color

@func_set_timeout(5)
def call_azure_gpt_api(question, answer, prediction, model):

    completion = azure_gpt4_client.chat.completions.create(
        extra_headers={"X-TT-LOGID": "yuanliping.0o0"},  # 请务必带上此header，方便定位问题
        model=model,
        messages=[
            {
                "role": "system",
                "content":
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {prediction}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}"
            }
        ]
    )
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

def try_call_api(question, answer, prediction, model, verbose=False):
    retry_exceptions = [
        "qpm limit, you can apply for expansion on the platform",
        "reach token limit, you can apply for expansion on the platform",
        "Request timed out",
        "The service is temporarily unable to process your request.",
        "upstream failed to respond",
        "502 Bad Gateway",
        "429 Too Many Requests",
        "Retrying request to"
    ]
    while True:
        try:
            gpt_q = call_azure_gpt_api(question, answer, prediction, model)
            return gpt_q, True
        except func_timeout.exceptions.FunctionTimedOut as e:
            if verbose:
                print(e)
        except Exception as e:
            e = str(e)
            if verbose:
                print(e)
            hit = any([x in e for x in retry_exceptions])
            if hit:
                time.sleep(5)
            else:
                return e, False

def process_one_sample(inputs):
    data, model, verbose = inputs
    prompt, response, prediction = data['question'], data['response'].lower(), data['prediction'].lower()
    result = None
    try:
        result, success = try_call_api(prompt, response, prediction, model, verbose)
        if not success:
            raise ValueError(result)
        result = ast.literal_eval(result)
        pred, score = result['pred'], result['score']
        # check pred
        if pred not in ['yes', 'no']:
            raise ValueError()
        # check score
        result['score'] = float(result['score'])
        if score < 0 or score > 5:
            raise ValueError()
    except Exception as e:
        if verbose:
            print(e)
            print(f'invalid GPT response: {result}')
        return {'success': False, 'result': result, 'data': data}
    return {'success': True, 'result': result, 'data': data}

class GPTMetric:
    def __init__(self, dataset_name, verbose=False) -> None:
        self.dataset_name = dataset_name
        self.num_worker = 64
        self.model = 'gpt-35-turbo-0125'
        self.results = []
        self.invalid_results = []
        self.dataset = []
        self.verbose = verbose
    
    def add(self, data):
        self.dataset.append(data)
    
    def process(self, dataset: List[Dict]):
        pool = Pool(processes = self.num_worker, )
        inputs = [(d, self.model, self.verbose) for d in dataset]
        results = pool.uimap(process_one_sample, inputs, chunksize = 1)

        for result in tqdm(results, total = len(dataset)):
            self.update_metric(result)
        pool.close()
        pool.join()
        pool.clear() # MUST
    
    def update_metric(self, result):
        if result['success']:
            self.results.append(result)
        else:
            self.invalid_results.append(result)
  
    def summarize_metric(self):
        if self.verbose:
            for result in self.results + self.invalid_results:
                print(f"Success: {Color.red(str(result['success']))}")
                print(Color.blue(json.dumps(result['data'], ensure_ascii=False)))
                print(Color.green(json.dumps(result['result'], ensure_ascii=False)))
        preds, scores = [], []
        for result in self.results:
            pred, score = result['result']['pred'], result['result']['score']
            preds.append(pred)
            scores.append(score)
        avg_score = np.average(scores)
        acc = np.average([p == 'yes' for p in preds])
        print(f'=====Evaluation Summary=====')
        self.eval_records = [
            f'Dataset: {self.dataset_name}\tMetric: GPT Accuracy',
            f'#Successful Results: {len(self.results)}\n#Failed Results: {len(self.invalid_results)}',
            f'Accuracy: {round(acc*100, 1)}',
            f'Average Score: {round(avg_score, 3)}',
        ]
        for info in self.eval_records:
            print(info)
    
    def save_results(self, pred_path):
        if os.path.isdir(pred_path):
            output_dir = os.path.join(pred_path, 'eval_records')
        else:
            output_dir = os.path.join(os.path.dirname(pred_path), 'eval_records')
        os.makedirs(output_dir, exist_ok=True)
        fout = open(os.path.join(output_dir, f'{self.dataset_name}_eval_result.txt'), 'w')
        for info in self.eval_records:
            fout.write(info+'\n')
        fout.close()
