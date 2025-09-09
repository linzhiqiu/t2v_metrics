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
from typing import List, Dict
import os
from pycocoevalcap.cider.cider import Cider
from tools.ptbtokenizer import PTBTokenizer

from tools.color import Color

class CIDErMetric:
    def __init__(self, dataset_name, verbose=False) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = PTBTokenizer()
        self.scorer = Cider()
        self.score = None
        self.results = []
        self.dataset = []
        self.verbose = verbose
    
    def add(self, data):
        self.dataset.append(data)

    def process(self, dataset: List[Dict]):
        references, predictions = {}, {}
        for i, data in enumerate(dataset):
            ref = data['response']
            pred = data['prediction']

            if isinstance(ref, str):
                ref = [ref]

            references[i] = [{'caption': r.lower()} for r in ref]
            predictions[i] = [{'caption': pred.lower()}]

        references = self.tokenizer.tokenize(references)
        predictions = self.tokenizer.tokenize(predictions)
        score, scores = self.scorer.compute_score(references, predictions)
        self.score = score
        for data, s in zip(dataset, scores):
            self.results.append({
                'score': s,
                'data': data,
            })

    def summarize_metric(self):
        if self.verbose:
            for result in self.results:
                print(Color.blue(json.dumps(result['data'])))
                print(Color.red(f"CIDEr score: {result['score']}"))
        print(f'=====Evaluation Summary=====')
        self.eval_records = [
            f'Dataset: {self.dataset_name}\tMetric: CIDEr',
            f'#Successful Results: {len(self.results)}',
            f'CIDEr score: {round(self.score*100, 1)}'
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
