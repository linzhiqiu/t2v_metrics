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
import os
from typing import List, Dict
from tools.color import Color


class AccuracyMetric:
    def __init__(self, dataset_name, verbose=False) -> None:
        self.dataset_name = dataset_name
        self.results = []
        self.invalid_results = []
        self.dataset = []
        self.verbose = verbose
    
    def add(self, data):
        self.dataset.append(data)
    
    def process(self, dataset: List[Dict]):
        if self.dataset_name in ['MVBench', 'TVBench',]:
            return self._process_group_by_subtask(dataset)
        else:
            return self._process(dataset)

    def _process(self, dataset: List[Dict], subtask=None):
        for data in dataset:
            prompt, response, prediction = data['prompt'], data['response'], data['prediction']
            prediction = prediction.replace('(', '').replace(')', '').strip()
            if len(prediction) <= 0:
                success = False
            else:
                prediction = prediction[0]
                if '0'<=prediction<='5':
                    prediction = chr(int(prediction) + ord('A'))
                success = prediction.isupper() and prediction.isalpha() and len(prediction) == 1
            if success:
                rst = {
                    'success': success,
                    'data': data,
                    'result': {'acc': response == prediction}
                }
                if subtask:
                    rst['subtask'] = subtask
                self.results.append(rst)
            else:
                rst = {
                    'success': success,
                    'data': data,
                    'result': {'acc': response == prediction}
                }
                if subtask:
                    rst['subtask'] = subtask
                self.invalid_results.append(rst)
    
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

    def summarize_metric(self):
        if self.dataset_name in ['MVBench', 'TVBench',]:
            return self._summarize_metric_by_subtask()
        else:
            return self._summarize_metric()

    def _summarize_metric(self):
        if self.verbose:
            for result in self.results + self.invalid_results:
                print(f"{Color.red('Success: ' + str(result['success']))}")
                print(Color.blue(json.dumps(result['data'], ensure_ascii=False)))
                print(f"{Color.green('Accuracy: ' + str(result['result']['acc']))}")

        accs = []
        for result in self.results:
            acc = result['result']['acc']
            accs.append(acc)
        avg_acc = np.average(accs)
    
        self.eval_records = [
            f'=====Evaluation Summary=====',
            f'Dataset: {self.dataset_name}\tMetric: Accuracy',
            f'#Successful Results: {len(self.results)}\n#Failed Results: {len(self.invalid_results)}',
            f'Accuracy: {round(avg_acc*100, 1)}',
        ]   
        for info in self.eval_records:
            print(info)
    
    def _summarize_metric_by_subtask(self):
        from prettytable import PrettyTable
        self.table = PrettyTable(['Task','Accuracy','Success','Failed'])
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
        oa_accs = []
        subtasks = list(sub_results.keys())
        subtasks.sort(key=lambda x:f"{x.split('/')[-1].split(' ')[0][0]}{x.split('/')[-1].split(' ')[1][0]}")
        for subtask in subtasks:
            sub_rsts = sub_results[subtask]
            sub_in_rsts = sub_invalid_results.get(subtask, [])
            accs = []
            for result in sub_rsts:
                acc = result['result']['acc']
                accs.append(acc)
                oa_accs.append(acc)
            avg_acc = np.average(accs)
            task_name = f"{subtask.split('/')[-1].split(' ')[0][0]}{subtask.split('/')[-1].split(' ')[1][0]}"
            self.table.add_row([task_name, round(avg_acc*100, 1), len(sub_rsts), len(sub_in_rsts)])
        self.table.add_row(['OVERALL', round(np.average(oa_accs)*100, 1), len(self.results), len(self.invalid_results)])
        print(f'=====Evaluation Summary=====')
        print(self.table)
    
    def save_results(self, pred_path):
        if os.path.isdir(pred_path):
            output_dir = os.path.join(pred_path, 'eval_records')
        else:
            output_dir = os.path.join(os.path.dirname(pred_path), 'eval_records')
        os.makedirs(output_dir, exist_ok=True)
        fout = open(os.path.join(output_dir, f'{self.dataset_name}_eval_result.txt'), 'w')
        if self.dataset_name in ['MVBench', 'TVBench',]:
            print(self.table, file=fout)
        else:
            for info in self.eval_records:
                fout.write(info+'\n')
        fout.close()
