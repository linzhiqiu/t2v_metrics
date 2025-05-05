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
from typing import Optional, List, Union
import os
from tools.color import Color

CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


class VideoMMEAccuracyMetric:
    def __init__(self, dataset_name, verbose=False) -> None:
        self.dataset_name = dataset_name
        self.results = []
        self.invalid_results = []
        self.dataset = []
        self.verbose = verbose
    
    def add(self, data):
        self.dataset.append(data)
    
    def process(self, dataset: List[Dict]):
        return self._process(dataset)

    def _process(self, dataset: List[Dict]):
        for data in dataset:
            prompt, response, prediction = data['prompt'], data['response'], data['prediction']
            extra_info = data['extra_info']
            prediction = prediction.replace('(', '').replace(')', '').strip()
            if len(prediction) <= 0:
                success = False
            else:
                prediction = prediction[0]
                if '1'<=prediction<='5':
                    prediction = chr(int(prediction) + ord('A'))
                success = prediction.isupper() and prediction.isalpha() and len(prediction) == 1
            if success:
                rst = {
                    'success': success,
                    'data': data,
                    'result': {'acc': response == prediction},
                    'extra_info': extra_info,
                    'missing': False
                }
                self.results.append(rst)
            else:
                rst = {
                    'success': success,
                    'data': data,
                    'result': {'acc': False},
                    'extra_info': extra_info,
                    'missing': True
                }
                self.results.append(rst)
                self.invalid_results.append(rst)

    def summarize_metric(self):
        if self.verbose:
            for result in self.results + self.invalid_results:
                print(f"{Color.red('Success: ' + str(result['success']))}")
                print(Color.blue(json.dumps(result['data'], ensure_ascii=False)))
                print(f"{Color.green('Accuracy: ' + str(result['result']['acc']))}")
        print(f'=====Evaluation Summary=====')
        print(f'Dataset: {self.dataset_name}\tMetric: Accuracy')
        print(f'#Successful Results: {len(self.results) - len(self.invalid_results)}\n#Failed Results: {len(self.invalid_results)}')
        self.eval_your_results(
            video_types = ["short","medium","long"],
            skip_missing = True,
            return_categories_accuracy = True,
            return_sub_categories_accuracy = False,
            return_task_types_accuracy = False,
        )

    def merge_results(self):
        results_merged_by_vid = {}
        for result in self.results:
            vid = result['extra_info']['vid']
            if vid not in results_merged_by_vid:
                results_merged_by_vid[vid] = {
                    'video_id': vid,
                    "duration": result['extra_info']['duration'],
                    "domain": result['extra_info']['domain'],
                    "sub_category": result['extra_info']['sub_category'],
                    'questions': [],
                    'missing': False
                }
            if result['missing']:
                results_merged_by_vid[vid]['missing'] = True
            results_merged_by_vid[vid]['questions'].append({
                'qid': result['extra_info']['idx'],
                'task_type': result['extra_info']['task_type'],
                'acc': result['result']['acc']
            }
            )
        return results_merged_by_vid            
    
    def eval_your_results(
        self,
        video_types: Optional[Union[List[str], str]] = None,
        skip_missing: Optional[bool] = False,
        return_categories_accuracy: Optional[bool] = True,
        return_sub_categories_accuracy: Optional[bool] = False,
        return_task_types_accuracy: Optional[bool] = False,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "response"

    ):
        """
        This copy from https://github.com/thanku-all/parse_answer/blob/main/eval_your_results.py
        Evaluate your results against the ground truth

        Args:
        - your_results_path (str): Path to your results file
        - video_types (Optional[List[str], str]): List of video types to evaluate. 
        - skip_missing (Optional[bool]): If True, missing files will be skipped. If False, an error will be raised if there are missing files.
        - return_categories_accuracy (Optional[bool]): If True, the accuracy for each video category will be returned.
        - return_sub_categories_accuracy (Optional[bool]): If True, the accuracy for each video sub category will be returned.
        - return_task_types_accuracy (Optional[bool]): If True, the accuracy for each task category will be returned.
        - gt_answer_key (Optional[str]): Key to access the ground truth answer in the results file.
        - your_answer_key (Optional[str]): Key to access your answer in the results file.
        """

        # Load your results
        # with open(your_results_path, 'r') as f:
        #     your_results = json.load(f)
        your_results = list(self.merge_results().values())
        self.eval_records = []
        if isinstance(video_types, str):
            video_types = video_types.split(",")

        q_type_dict = {}
        v_type_dict = {}
        v_sub_type_dict = {}


        for video_type in video_types:

            # Filter your results based on video types
            your_results_video_type = [item for item in your_results if item['duration'] == video_type]

            # Task Categories
            q_type_dict[video_type] = {}
            for q_type in TASK_CATEGORIES:
                q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

            # Video categories
            v_type_dict[video_type] = {}
            for v_type in CATEGORIES:
                v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}
            
            v_sub_type_dict[video_type] = {}
            for v_sub_type in SUB_CATEGORIES:
                v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

            if not skip_missing:
                # Check if the number of files in your results and ground truth are the same
                print(len(your_results_video_type))
                assert len(your_results_video_type) == 300, f"Number of files in {video_type} is not 300. Check if there are missing files."

            for item in your_results_video_type:

                if skip_missing and item["missing"]:
                    continue

                # Get the video category, sub category and question category
                video_category = item["domain"]
                video_sub_category = item["sub_category"]
                
                questions = item["questions"]

                for question in questions:
                    q_type = question["task_type"]

                    # Get the ground truth and your response
                    # gt_answer = question[gt_answer_key]
                    # response = question[your_answer_key]
                    acc = question['acc']

                    # Extract the answer from the response
                    # extration = extract_characters_regex(response)
        
                    if acc is not None:
                        q_type_dict[video_type][q_type]["answered"] += 1
                        q_type_dict[video_type][q_type]["correct"] += acc

                        v_type_dict[video_type][video_category]["answered"] += 1
                        v_type_dict[video_type][video_category]["correct"] += acc

                        v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                        v_sub_type_dict[video_type][video_sub_category]["correct"] += acc


        # Print the results for each video type
        for video_type in video_types:
            info = f"=====================================\nEvaluation on video Type: {video_type}\n====================================="
            self.eval_records.append(info)
            print(info)
            if return_categories_accuracy:
                info = f"-------------------------------------\nVideo Categories\n-------------------------------------"
                self.eval_records.append(info)
                print(info)
                for v_type in v_type_dict[video_type]:
                    info = f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%"
                    self.eval_records.append(info)
                    print(info)
            if return_sub_categories_accuracy:
                info = f"-------------------------------------\nVideo Sub Categories\n-------------------------------------"
                self.eval_records.append(info)
                for v_sub_type in v_sub_type_dict[video_type]:
                    info = f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%"
                    self.eval_records.append(info)
                    print(info)
            if return_task_types_accuracy:
                info = f"-------------------------------------\nTask Categories\n-------------------------------------"
                self.eval_records.append(info)
                print(info)
                for q_type in q_type_dict[video_type]:
                    info = f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%"
                    self.eval_records.append(info)
                    print(info)
            info = f"-------------------------------------\nOverall Performance\n-------------------------------------"
            
            print(info)
            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
            info = f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
            self.eval_records.append(info)
            print(info+'\n')

        # Print the results for the entire dataset
        info = f"=====================================\nEvaluation on the entire dataset\n====================================="
        self.eval_records.append(info)
        print(info)

        if return_categories_accuracy:
            info = f"-------------------------------------\nVideo Categories\n-------------------------------------"
            self.eval_records.append(info)
            print(info)
            for v_type in CATEGORIES:
                total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
                total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
                info = f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
                self.eval_records.append(info)
                print(info)
        

        if return_sub_categories_accuracy:
            info = f"-------------------------------------\nVideo Sub Categories\n-------------------------------------"
            self.eval_records.append(info)
            print(info)

            for v_sub_type in SUB_CATEGORIES:
                total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
                total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
                info = f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
                self.eval_records.append(info)
                print(info)


        if return_task_types_accuracy:
            info = f"-------------------------------------\nTask Categories\n-------------------------------------"
            self.eval_records.append(info)
            print(info)
            for q_type in TASK_CATEGORIES:

                total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
                total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
                info = f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
                self.eval_records.append(info)
                print(info)

        info = f"*************************************\nOverall Performance\n*************************************"
        self.eval_records.append(info)
        print(info)
        total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
        total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
        info = f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%"
        self.eval_records.append(info)
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
