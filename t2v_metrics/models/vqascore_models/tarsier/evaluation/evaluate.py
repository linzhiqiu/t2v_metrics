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
import os
import random

from .metrics import CIDErMetric, GPTMetric, DREAMGPTMetric, AccuracyMetric, VideoMMEAccuracyMetric
from tools.rw_utils import read_jsonlines
from tools.color import Color
from dataset.utils import get_benchmarks

def extract_item_for_eval(data_dict):
    item = {}
    prompt, prediction, reference = [], [], []
    for msg in data_dict['messages']:
        for content in msg['content']:
            if content['type'] == 'text':
                if msg['role'] == 'user':
                    prompt.append(content['text'])
                elif msg['role'] == 'assistant':
                    if content.get('reference'):
                        reference.append(content['reference'])
                        prediction.append(content['text'])
                        # prediction.append(content['reference']) # debug                   
    
    item['prompt'] = ''.join(prompt)
    item['prediction'] = ''.join(prediction)
    item['response'] = ''.join(reference)
        
    item['dataset'] = data_dict['dataset']
    item['idx'] = f"{data_dict['dataset']}_{data_dict['idx']}"
    extra_info = data_dict.get('extra_info', None)
    vid = data_dict.get('vid', None)
    if vid is not None:
        item['vid'] = vid
    if extra_info:
        item['events'] = extra_info.get('events', None)
        item['extra_info'] = extra_info
    if 'is_hard' in data_dict:
        item['is_hard'] = data_dict['is_hard']
    
    return item

def read_dataset(path, dataset_name):
    if os.path.isdir(path):
        lines = []
        for f in os.listdir(path):
            if f.endswith('.jsonl'):
                lines.extend(read_jsonlines(os.path.join(path, f)))
    else:
        lines = read_jsonlines(path)
    dataset = []
    idxs = set()
    for l in lines:
        if l['dataset'].split('/')[0] != dataset_name:
            continue
        idx = f"{l['dataset']}_{l['idx']}"
        if idx in idxs:
            continue

        idxs.add(idx)
        item = extract_item_for_eval(l)
        dataset.append(item)
    return dataset

METRIC_MAPPING = {
    'CIDErMetric': CIDErMetric,
    'GPTMetric': GPTMetric,
    'AccuracyMetric': AccuracyMetric,
    'DREAMGPTMetric': DREAMGPTMetric,
    'VideoMMEAccuracyMetric': VideoMMEAccuracyMetric
}

def evaluate(pred_file, METRIC, dataset_name, sample_num=-1, verbose = False):
    dataset = read_dataset(pred_file, dataset_name)
    if len(dataset) == 0:
        return
    if sample_num > 0:
        dataset = random.sample(dataset, sample_num)
    metric = METRIC(dataset_name=dataset_name, verbose=verbose)
    metric.process(dataset)
    metric.summarize_metric()
    metric.save_results(pred_file)
    if isinstance(metric, DREAMGPTMetric):
        metric.save_eval_infos(pred_file)

def evaluate_all(pred_file, METRIC2DATASET, sample_num=-1, verbose = False):
    for METRIC, dataset_name in METRIC2DATASET:
        if isinstance(METRIC, str):
            METRIC = METRIC_MAPPING[METRIC]
        print(f"### Start Evaluating on {dataset_name}")
        evaluate(pred_file, METRIC, dataset_name, sample_num, verbose)
        print(f"### Finish Evaluating on {dataset_name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--benchmarks', nargs='+', default=["all"], help="Default as 'all' to evaluate on all benchmarks; Also could be task types: ('dream', 'caption', 'mc_qa', 'oe_qa'); And specific benchmark names: ('dream', 'msvd-caption', 'msr-vtt-caption', 'vatex-caption', 'next-qa', 'egoschema', 'mvbench', 'tvbench', 'video-mme', 'msvd-qa', 'msr-vtt-qa', 'tgif-qa', 'anet-qa')")
    parser.add_argument('--sample_num', type=int, default=-1)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    args.benchmarks = get_benchmarks(args.benchmarks)
    print("### Selected Benchmarks:", args.benchmarks)

    Benchmark2Metric = {
        # Multi-chocie QA
        'next-qa': 'AccuracyMetric',
        'egoschema': 'AccuracyMetric',
        'mvbench': 'AccuracyMetric',
        'tvbench': 'AccuracyMetric',
        'video-mme': 'VideoMMEAccuracyMetric',

        # Open-ended QA
        'msvd-qa': 'GPTMetric',
        'msr-vtt-qa': 'GPTMetric',
        'tgif-qa': 'GPTMetric',
        'anet-qa': 'GPTMetric',

        # Caption DREAM
        'dream': 'DREAMGPTMetric',

        # Caption CIDEr
        'msvd-caption': 'CIDErMetric',
        'msr-vtt-caption': 'CIDErMetric',
        'vatex-caption': 'CIDErMetric',
    }

    Benchmark2Dataset = {
        'dream': 'DREAM',

        'next-qa': 'Next-QA-val-multi_choice',
        'egoschema': 'EgoSchema',
        'mvbench': 'MVBench',
        'tvbench': 'TVBench',
        'video-mme': 'Video-MME',

        'msvd-qa': 'MSVD-QA-val',
        'msr-vtt-qa': 'MSR-VTT-QA-val',
        'tgif-qa': 'TGIF-QA-test',
        'anet-qa': 'ActivityNet-QA-test',

        'msvd-caption': 'MSVD-Caption-test',
        'msr-vtt-caption': 'MSR-VTT-Caption-test',
        'vatex-caption': 'VATEX-test',
    }

    METRIC2DATASET = []
    
    for bm in args.benchmarks:
        if bm not in Benchmark2Metric:
            print(Color.red(f"Unknown benchmark: {bm}"))
            continue

        METRIC2DATASET.append([Benchmark2Metric[bm], Benchmark2Dataset[bm]])   

    evaluate_all(args.pred_file, METRIC2DATASET, args.sample_num, args.verbose)

    # python3 -m evaluation.evaluate --pred_file $pred_file --sample_num=100
