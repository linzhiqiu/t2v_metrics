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

from tools.gpt_api import azure_gpt4_client
import re
import os
from copy import deepcopy
from traceback import format_exc

def count_f1(r, p):
    return 2*r*p/(r+p)

@func_set_timeout(60)
def call_azure_gpt_api(events, reference, prediction, model):
    if len(events) == 0:
        events = [reference.replace('\n', ' ')]  
    completion = azure_gpt4_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content":
                        "Given a video description and a list of events. For each event, classify the relationship between the video description and the event into three classes: entailment, neutral, contradiction.\n"
                        "- \"entailment\" means that the video description entails the event.\n"
                        "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
                        "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
                        f"Video Description:\n{prediction}\n\n"
                        f"Events: {events}\n"

                        "Output a JSON formed as:\n"
                        "{\n"
                        "  \"events\": [\n"
                        "    {\"event\": \"copy an event here\", \"relationship\": \"put class name here\",  \"reason\": \"give your reason here\"},\n"
                        "    ...\n"
                        "  ]\n"
                        "}\n\n"
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
            }
        ]
    )
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

@func_set_timeout(60)
def call_azure_gpt_api_for_events(caption, model):
    completion = azure_gpt4_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content":
                        "Bellow is a description of a video clip:\n"
                        f"Video Description: {caption}\n\n"

                        "Extract at most 10 key events from the above video description paragraph. Requirements\n:"
                        "- An event must include an action, motion or movement (NOT STATIC INFORMATION). DON'T repeat same events.\n"
                        "- Every event is represented by a brief sentence within 10 words, with a subject, a predicate and optionally an object, avoid unnecessary appearance descriptions.\n"
                        "- Every event must be atomic, meaning that it cannot be further split into multiple events.\n"
                        "- Scene cuts and camera motions are NOT events.\n"
                        "- Substitute pronouns by the nouns they refer to.\n\n"
                        "Please generate the response in the form of a Python dictionary string with keys \"events\". The value of \"events\" is a List(str), of which each item is an event. "
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {\"events\": [event1, event2, ...]}"
            }
        ]
    )
    return json.loads(completion.model_dump_json())['choices'][0]['message']['content']

def try_call_api_for_eval(events, answer, prediction, model, verbose=False, max_retry=-1):
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
    retry = 0
    while True and (retry<max_retry or max_retry<0):
        retry += 1
        try:
            gpt_q = call_azure_gpt_api(events, answer, prediction, model)
            gpt_q = gpt_q.strip()
            gpt_q = re.sub(r'\n+', '\n', gpt_q)
            gpt_q = re.sub(r'\s+', ' ', gpt_q)
            
            if gpt_q.startswith("```json"):
                gpt_q = gpt_q.replace("```json", "").replace("```", "").strip()
            elif gpt_q.startswith("```python"):
                gpt_q = gpt_q.replace("```python", "").replace("```", "").strip()
            if not gpt_q.startswith('{'):
                gpt_q = '{' + gpt_q
            if not gpt_q.endswith('}'):
                gpt_q = gpt_q + '}'
            gpt_q = gpt_q.replace("True", "true").replace("False", "false")
            gpt_q = gpt_q.replace("} {", "}, {").replace("}{", "}, {")
            gpt_q = gpt_q.replace(",\n}", "\n}").replace(", \n}", "\n}").replace(", }", "}").replace(",}", "}")
            gpt_q = gpt_q.replace(",\n]", "\n]").replace(", \n]", "\n]").replace(", ]", "]").replace(",]", "]")
            gpt_q = gpt_q.replace("[Placeholder]", "null")
            gpt_q = gpt_q.replace("{Events:", "").strip()
            
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
    return f"Exceed max try: {max_retry}", False

def try_call_api_for_events(caption, model, verbose=False):
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
            gpt_q = call_azure_gpt_api_for_events(caption, model)
            if gpt_q.startswith("```json"):
                gpt_q = gpt_q.replace("```json", "").replace("```", "").strip()
            elif gpt_q.startswith("```python"):
                gpt_q = gpt_q.replace("```python", "").replace("```", "").strip()
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

def extract_events(inputs, is_pred=False, max_retry=10):
    data, model, verbose = inputs
    if is_pred:
        caption = data['prediction'].lower()
    else:
        caption = data['response'].lower()
    caption = caption.replace("\"", "\'")
    retry = 0
    while True and (retry<max_retry or max_retry<0):
        retry += 1
        result, success = try_call_api_for_events(caption, model, verbose)
        if not success:
            print(f"[error]: try_call_api_for_events failed!", flush=True)
            continue
        try:
            result = ast.literal_eval(result)
            events = result['events']
            if verbose:
                print("pred_events=" if is_pred else "gt events=", events, ":", caption)
            assert isinstance(events, list) and (len(events)==0 or isinstance(events[0], str))
            return events
        except Exception as e:
            print(format_exc(), flush=True)
            continue
    print("[error]: Exceed max_retry!", flush=True)
    raise ValueError("[error]: Exceed max_retry!")
        

def evaluate_one_sample(events, response, prediction, model, verbose, return_hit_num=False, is_recall=False, max_retry=10):
    retry = 0
    while True and (retry<max_retry or max_retry<0):
        retry += 1
        try:
            assert isinstance(events, list)
            result = None
            result, success = try_call_api_for_eval(events, response, prediction, model, verbose)
            if not success:
                print("[error]: try_call_api_for_eval failed!", flush=True)
                continue
            try:
                events_filled = json.loads(result)
                events_filled = events_filled['events']
            except Exception as e:
                print("load json failed:", result)
                continue
            assert len(events) == len(events_filled) or (len(events) == 0 and len(events_filled) == 1)
            num_matched_events = 0
            try:
                for event in events_filled:
                    pred = event['relationship'].strip().lower()
                    assert pred in ['entailment', 'neutral', 'contradiction']
                    pos_classes = ['entailment'] if is_recall else ['entailment', 'neutral']
                    if pred in pos_classes:
                        num_matched_events += 1
            except Exception as e:
                print(f"Invalid response: {events_filled}")
                continue
            if len(events) == 0:
                motion_score = 1.0
            else:
                motion_score = num_matched_events / len(events)
            if return_hit_num:
                return motion_score, events_filled, f"hit: {num_matched_events} / {len(events)}"
            return motion_score
        except Exception as e:
            print(format_exc(), flush=True)
            continue
    print("[error]: Exceed max_retry!", flush=True)
    raise ValueError(f"[error]: Exceed max_retry!")


def process_one_sample(inputs):
    data, model, verbose = inputs
    response, prediction = data['response'].lower(), data['prediction'].lower()
    result = None
    try:
        if isinstance(data.get('events', None), list):
            gt_events = data['events']
        else:
            gt_events = extract_events(inputs, is_pred=False)
        pred_events = extract_events(inputs, is_pred=True)
        assert isinstance(gt_events, list) and isinstance(pred_events, list)
        result = {}
        motion_score_r, events_filled_r, hit_num_r = evaluate_one_sample(gt_events, response, prediction, model, verbose, return_hit_num=True, is_recall=True)
        motion_score_p, events_filled_p, hit_num_p = evaluate_one_sample(pred_events, prediction, response, model, verbose, return_hit_num=True, is_recall=True)
        result['score_r'] = motion_score_r
        result['score_p'] = motion_score_p
        result['eval_infos'] = {
            'idx': data['idx'],
            'gt': response,
            'pred': prediction,
            'events_gt': events_filled_r,
            'hit_num_recall': hit_num_r,
            'events_pred': events_filled_p,
            "hit_num_precision": hit_num_p,
        }
        if 'extra_info' in data:
            result['extra_info'] = data['extra_info']
    except Exception as e:
        if verbose:
            print(e)
            print(f'invalid GPT response: {result}')
        result = None
        return {'success': False, 'result': result, 'data': data}
    return {'success': True, 'result': result, 'data': data}

class DREAMGPTMetric:
    def __init__(self, dataset_name, verbose=False) -> None:
        self.dataset_name = dataset_name
        self.num_worker = 64
        # self.model = 'gpt-35-turbo'
        self.model = 'gpt-35-turbo-0125'
        # self.model='gpt-4-1106-preview'
        self.results = []
        self.invalid_results = []
        self.dataset = []
        self.verbose = verbose
        self.eval_infos = []
        self.buckets = {
            "subjects": {
                '<=1': [], '==2': [], '==3': [], '>=4': []
            },
            "shots": {'<=1': [], '==2': [], '==3': [], '>=4': []
            },
            "events": {'<=3': [], 'in [4, 5]': [], 'in [6, 7]': [], '>=8': []
            }
        }
    
    def add(self, data):
        self.dataset.append(data)
    
    def select_bucket(self, bucket_name, num):
        for key in self.buckets[bucket_name]:
            if eval(f"{num}{key}"):
                return key
        return ''
    
    def add_to_bucket(self, bucket_name, data):
        sub_bucket = self.select_bucket(bucket_name, data['result']['extra_info'][f'n_{bucket_name}'])
        if sub_bucket:
            self.buckets[bucket_name][sub_bucket].append(data)
    
    def process(self, dataset: List[Dict]):
        self._process_group_by_subtask(dataset)
    
    def _process(self, dataset: List[Dict], subtask=None):
        pool = Pool(processes = self.num_worker, )
        inputs = [(d, self.model, self.verbose) for d in dataset]
        results = pool.uimap(process_one_sample, inputs, chunksize = 1)

        for result in tqdm(results, total = len(dataset), desc=f'eval {subtask}'):
            if subtask:
                result['subtask'] = subtask
            self.update_metric(result)
        pool.close()
        pool.join()
        pool.clear() # MUST

    def _process_group_by_subtask(self, dataset: List[Dict]):
        def _group_by_subtask(dataset):
            subtasks = {}
            for data in dataset:
                if data['dataset'] not in subtasks:
                    subtasks[data['dataset']] = []
                subtasks[data['dataset']].append(data)
            return subtasks
        subtasks = _group_by_subtask(dataset)
        for subtask, subdata in subtasks.items():
            self._process(subdata, subtask)

    def update_metric(self, result):
        if result['success']:
            self.results.append(result)
        else:
            self.invalid_results.append(result)
    
    def summarize_metric(self):
        self._summarize_metric_by_subtask()
        self._summarize_metric_by_bucket()

    def _summarize_metric_by_subtask(self):
        from prettytable import PrettyTable
        self.table = PrettyTable(['Task', 'F1 Score', 'Action Recall', 'Action Precision', 'Success', 'Failed'])
        def _group_by_subtask():
            sub_results = {}
            sub_invalid_results = {}
            for data in self.results:
                if data['subtask'] not in sub_results:
                    sub_results[data['subtask']] = []
                sub_results[data['subtask']].append(data)
            for data in self.invalid_results:
                if data['subtask'] not in sub_invalid_results:
                    sub_invalid_results[data['subtask']] = []
                sub_invalid_results[data['subtask']].append(data)
            return sub_results, sub_invalid_results
        sub_results, sub_invalid_results = _group_by_subtask()
        overall_avg_recall = []
        overall_avg_precision = []
        subtasks = list(sub_results.keys())
        subtasks.sort()
        for subtask in subtasks:
            sub_rsts = sub_results[subtask]
            sub_in_rsts = sub_invalid_results.get(subtask, [])
            recalls = []
            precisions = []
            for result in sub_rsts:
                r, p, infos = result['result']['score_r'], result['result']['score_p'], result['result']['eval_infos']
                recalls.append(r)
                precisions.append(p)
                self.eval_infos.append(infos)
            avg_recall = np.average(recalls)
            avg_precision = np.average(precisions)
            f1 = count_f1(avg_recall, avg_precision)
            overall_avg_recall.append(avg_recall)
            overall_avg_precision.append(avg_precision)
            task_name = subtask
            self.table.add_row([task_name, round(f1, 3), round(avg_recall, 3), round(avg_precision, 3), len(sub_rsts), len(sub_in_rsts)])
        overall_recall = np.average(overall_avg_recall)
        overall_precision = np.average(overall_avg_precision)
        overall_f1 = count_f1(overall_recall, overall_precision)
        self.table.add_row(['OVERALL', round(overall_f1, 3), round(overall_recall, 3), round(overall_precision, 3), len(self.results), len(self.invalid_results)])
        print(f'=====DREAM Evaluation Summary=====')
        print(self.table)
        

    def _summarize_metric_by_bucket(self):
        from prettytable import PrettyTable
        self.bucket_tables = []
        for bucket in self.buckets:
            table = PrettyTable(['Score'] + list(self.buckets[bucket].keys()))
            for data in self.results:
                self.add_to_bucket(bucket_name=bucket, data=data)
            bucket_result = {}
            for sub_bucket in self.buckets[bucket]:
                recalls = []
                precisions = []
                for result in self.buckets[bucket][sub_bucket]:
                    r, p = result['result']['score_r'], result['result']['score_p']
                    recalls.append(r)
                    precisions.append(p)
                avg_recall = np.average(recalls)
                avg_precision = np.average(precisions)
                f1 = count_f1(avg_recall, avg_precision)
                bucket_result[sub_bucket] = (avg_recall, avg_precision, f1)
            
            raw = []
            scores = ['Recall', 'Precision', 'F1']
            for i in range(len(scores)):
                raw = [scores[i]]
                for sub_bucket in bucket_result:
                    raw.append(round(bucket_result[sub_bucket][i], 3))
                table.add_row(raw)
            sample_num = ['Count']
            for k in self.buckets[bucket]:
                sample_num.append(len(self.buckets[bucket][k]))
            table.add_row(sample_num)
            bucket_info = f'\n=====DREAM Evaluation Split by Bucket #{bucket}====='
            print(bucket_info)
            print(table)
            self.bucket_tables.append(bucket_info)
            self.bucket_tables.append(deepcopy(table))
    
    def save_results(self, pred_path):
        if os.path.isdir(pred_path):
            output_dir = os.path.join(pred_path, 'eval_records')
        else:
            output_dir = os.path.join(os.path.dirname(pred_path), 'eval_records')
        os.makedirs(output_dir, exist_ok=True)
        fout = open(os.path.join(output_dir, f'{self.dataset_name}_eval_result.txt'), 'w')
        print(self.table, file=fout)
        for bucket_info in self.bucket_tables:
            print(bucket_info)
        fout.close()
    
    def save_eval_infos(self, pred_path):
        if os.path.isdir(pred_path):
            output_dir = os.path.join(pred_path, 'eval_records')
        else:
            output_dir = os.path.join(os.path.dirname(pred_path), 'eval_records')
        os.makedirs(output_dir, exist_ok=True)
        fout = open(os.path.join(output_dir, 'DREAM_eval_infos.jsonl'), 'w')
        for info in self.eval_infos:
            fout.write(json.dumps(info) +'\n')
        fout.close()
        print(f"DREAM evaluation information saved in: {os.path.join(output_dir, 'DREAM_eval_infos.jsonl')}", flush=True)

