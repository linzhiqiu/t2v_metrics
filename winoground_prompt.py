# Evaluate on all dataset using a specific score
# CUDA_VISIBLE_DEVICES=1 python winoground_prompt.py --model clip-flant5-xxl
# CUDA_VISIBLE_DEVICES=0 python winoground_prompt.py --model llava-v1.5-13b
# CUDA_VISIBLE_DEVICES=3 python winoground_prompt.py --model instructblip-flant5-xxl
import argparse
import os
import t2i_metrics
from dataset import Winoground, EqBen_Mini


qa_dict = [
    # {'question': 'Is this figure showing "{}"? Please answer yes or no.', 'answer': 'Yes'},
    # {'question': 'Is this figure showing "{}"? Please answer yes or no.', 'answer': 'Yes.'},
    # {'question': 'Is this figure showing "{}"? Please answer yes or no.', 'answer': 'yes'},
    # {'question': 'Is this figure showing "{}"? Please answer yes or no.', 'answer': 'Yes, it is.'},
    # {'question': 'Is this figure showing "{}"?', 'answer': 'Yes'},
    # {'question': 'Is this figure showing "{}"?', 'answer': 'Yes, it is.'},
    # {'question': 'Is this figure showing "{}"? Please answer true or false.', 'answer': 'True'},
    
    
    # {'question': "Is this figure showing '{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "Is this figure showing '{}'? Please answer yes or no.", 'answer': 'Yes.'},
    # {'question': "Is this figure showing '{}'? Please answer yes or no.", 'answer': 'yes'},
    # {'question': "Is this figure showing '{}'?", 'answer': 'Yes'},
    # {'question': "Is this figure showing '{}'?", 'answer': 'Yes, it is.'},
    # {'question': "Is this figure showing '{}'? Please answer yes or no.", 'answer': 'Yes, it is.'},
    # {'question': "Is this figure showing '{}'? Please answer true or false.", 'answer': 'True'},


    # {'question': 'Is "{}" an accurate description of this figure? Please answer yes or no.', 'answer': 'Yes'},
    # {'question': 'Can "{}" be seen in this figure? Please answer yes or no.', 'answer': 'Yes'},
    # {'question': 'Is this figure showing "{}"? Please answer correct or wrong.', 'answer': 'Correct'},
    # {'question': 'Question: Is this figure showing "{}"? Please answer yes or no.', 'answer': 'Yes'},
    # {'question': 'Question: Is this figure showing "{}"? Short answer:', 'answer': 'Yes'},
    # {'question': 'Question: Is this figure showing "{}"? Short answer:', 'answer': 'Yes'},
    # {'question': 'Is this figure showing "{}"?', 'answer': 'Yes'},
    # {'question': 'Q: Is this figure showing "{}"? A:', 'answer': 'Yes'},


    # {'question': "Are '{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "Do '{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "Is '{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "Does '{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "'{}'? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': "{}? Please answer yes or no.", 'answer': 'Yes'},
    # {'question': 'Is it "{}"? Please answer yes or no.', 'answer': 'Yes'},
    # {'question': 'Is this "{}"? Please answer yes or no.', 'answer': 'Yes'},




    {'question': 'Does this figure show "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does this figure show {}? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does this figure show "{}"?', 'answer': 'Yes'},
    {'question': 'Does it show "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': '"{}"?', 'answer': 'Yes'},
    {'question': '{}?', 'answer': 'Yes'},
    {'question': '"{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': '{}? Please answer yes or no.', 'answer': 'Yes'},
    
    {'question': 'Is "{}" an accurate description of this figure? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Can "{}" be seen in this figure? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does this image show "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does this picture show "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Does this photo show "{}"? Please answer yes or no.', 'answer': 'Yes'},
    {'question': 'Is this figure showing "{}"? Please answer yes or no.', 'answer': 'Yes'},
    
    {'question': 'Does this figure show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Does this figure show "{}"? Please answer correct or wrong.', 'answer': 'Correct'},
    {'question': 'Does this figure show "{}"?', 'answer': 'Yes, it is.'},
    {'question': 'Does this figure show "{}"? Please answer true or false.', 'answer': 'True'},
    
    {'question': 'Does this figure show "{}"? Answer the question using a single word or phrase.', 'answer': 'Yes'},
    {'question': 'Use the provided image to answer the question: Does this figure show "{}"? Provide your answer as short as possible.', 'answer': 'Yes'},
    {'question': 'The question "Does this figure show "{}"?" can be answered using the image. A short answer is ', 'answer': 'Yes'},
    {'question': 'What is the answer to the following question? "Does this figure show "{}"?"', 'answer': 'Yes'},
    {'question': 'Based on the image, respond to this question with a short answer: "Does this figure show "{}"?"', 'answer': 'Yes'},
    
    
    {'question': 'Question: Does this figure show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this figure show {}? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this figure show "{}"?', 'answer': 'yes'},
    {'question': 'Question: Does it show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: "{}"?', 'answer': 'yes'},
    {'question': 'Question: {}?', 'answer': 'yes'},
    {'question': 'Question: "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: {}? Please answer yes or no.', 'answer': 'yes'},
    
    {'question': 'Question: Is "{}" an accurate description of this figure? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Can "{}" be seen in this figure? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this image show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this picture show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this photo show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Is this figure showing "{}"? Please answer yes or no.', 'answer': 'yes'},
    
    {'question': 'Question: Does this figure show "{}"? Please answer yes or no.', 'answer': 'yes'},
    {'question': 'Question: Does this figure show "{}"? Please answer correct or wrong.', 'answer': 'correct'},
    {'question': 'Question: Does this figure show "{}"?', 'answer': 'yes, it is.'},
    {'question': 'Question: Does this figure show "{}"? Please answer true or false.', 'answer': 'true'},
    
    {'question': 'Question: Does this figure show "{}"? Answer the question using a single word or phrase.', 'answer': 'yes'},
    {'question': 'Question: Use the provided image to answer the question: Does this figure show "{}"? Provide your answer as short as possible.', 'answer': 'yes'},
    {'question': 'Question: The question "Does this figure show "{}"?" can be answered using the image. A short answer is ', 'answer': 'yes'},
    {'question': 'Question: What is the answer to the following question? "Does this figure show "{}"?"', 'answer': 'yes'},
    {'question': 'Question: Based on the image, respond to this question with a short answer: "Does this figure show "{}"?"', 'answer': 'yes'},
    
    
]

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2i_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model", default="clip-flant5-xxl", type=str)
    return parser.parse_args()


def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    

    result_path =f"winoground_results/{args.model}_prompts_new.json"
    if os.path.exists(result_path):
        print(f"Results for {args.model} already exists at {result_path}.")
    else:
        score_func = t2i_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)
        qa_results = {}
        for qa in qa_dict:
            kwargs = {}
            kwargs['question_template'] = qa['question']
            kwargs['answer_template'] = qa['answer']
            print(f"Performance of {kwargs}.")
            qa_results[f"Q_{qa['question']}_A_{qa['answer']}"] = {}
            for dataset_cls in [
                Winoground,
                EqBen_Mini,
            ]:
                dataset = dataset_cls(root_dir=args.root_dir)
                scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
                results = dataset.evaluate_scores(scores)
                qa_results[f"Q_{qa['question']}_A_{qa['answer']}"][dataset_cls.__name__] = results['all']
        import json
        with open(result_path, "w") as f:
            json.dump(qa_results, f, indent=4)
    
    
# def gather_all_results(models=['clip-flant5-xxl', 'llava-v1.5-13b', 'instructblip-flant5-xxl']):
#     import json
#     qa_results = {}
#     for model in models:
#         result_path =f"winoground_results/{model}_prompts_new.json"
#         if os.path.exists(result_path):
#             with open(result_path, "r") as f:
#                 qa_results[model] = json.load(f)
    
#     lines = []
#     winoground_group = []
#     for item in qa_dict:
#         question = item['question']
#         answer = item['answer']
#         question_formatted = question.format('\\{\\}')
#         line = f"{question_formatted} & {answer} "
#         for model in models:
#             for dataset in ['Winoground', 'EqBen_Mini']:
#                 line += f" & {qa_results[model][f'Q_{question}_A_{answer}'][dataset]['group']*100.:.1f} "
#         line += "\\\\"
#         lines.append(line)
#         winoground_group.append(float(qa_results['clip-flant5-xxl'][f'Q_{question}_A_{answer}']['Winoground']['group']*100.))
    
#     # sort the lines by winoground group
#     lines = [line for _, line in sorted(zip(winoground_group, lines))]
#     with open(f"winoground_results/all_prompts_new.txt", "w") as f:
#         f.write("\n".join(lines))


# def gather_all_results_question_only(models=['clip-flant5-xxl', 'llava-v1.5-13b', 'instructblip-flant5-xxl']):
#     import json
#     qa_results = {}
#     for model in models:
#         result_path =f"winoground_results/{model}_prompts_new.json"
#         if os.path.exists(result_path):
#             with open(result_path, "r") as f:
#                 qa_results[model] = json.load(f)
    
#     lines = []
#     winoground_group = []
#     for item in qa_dict:
#         question = item['question']
#         answer = item['answer']
#         if answer != 'Yes':
#             continue
#         question_formatted = question.format('\\{\\}')
#         line = f"{question_formatted} "
#         for model in models:
#             for dataset in ['Winoground', 'EqBen_Mini']:
#                 group_score = qa_results[model][f'Q_{question}_A_{answer}'][dataset]['group']
#                 if model == 'instructblip-flant5-xxl':
#                     try:
#                         group_score = qa_results[model][f'Q_Question: {question}_A_yes'][dataset]['group']
#                         # print(f"Key exists for {question}")
#                     except:
#                         import pdb; pdb.set_trace()
#                 line += f" & {group_score*100.:.1f} "
#         line += "\\\\"
#         lines.append(line)
#         winoground_group.append(float(qa_results['clip-flant5-xxl'][f'Q_{question}_A_{answer}']['Winoground']['group']*100.))
    
#     # sort the lines by winoground group
#     lines = [line for _, line in sorted(zip(winoground_group, lines))]
#     with open(f"winoground_results/all_prompts_new.txt", "w") as f:
#         f.write("\n".join(lines))

def gather_all_results_answer_only(models=['clip-flant5-xxl', 'llava-v1.5-13b', 'instructblip-flant5-xxl']):
    import json
    qa_results = {}
    for model in models:
        result_path =f"winoground_results/{model}_prompts_new.json"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                qa_results[model] = json.load(f)
    
    lines = []
    winoground_group = []
    for item in qa_dict:
        question = item['question']
        answer = item['answer']
        if answer == 'Yes':
            continue
        if question.startswith("Question: "):
            continue
        question_formatted = question.format('\\{\\}')
        line = f"{question_formatted} & P({answer}) "
        for model in models:
            for dataset in ['Winoground', 'EqBen_Mini']:
                group_score = qa_results[model][f'Q_{question}_A_{answer}'][dataset]['group']
                if model == 'instructblip-flant5-xxl':
                    try:
                        group_score = qa_results[model][f'Q_Question: {question}_A_{answer.lower()}'][dataset]['group']
                        # print(f"Key exists for {question}")
                    except:
                        import pdb; pdb.set_trace()
                line += f" & {group_score*100.:.1f} "
        line += "\\\\"
        lines.append(line)
        winoground_group.append(float(qa_results['clip-flant5-xxl'][f'Q_{question}_A_{answer}']['Winoground']['group']*100.))
    
    # sort the lines by winoground group
    lines = [line for _, line in sorted(zip(winoground_group, lines))]
    with open(f"winoground_results/all_prompts_negate.txt", "w") as f:
        f.write("\n".join(lines))
    
if __name__ == "__main__":
    main()
    # gather_all_results_question_only()
    gather_all_results_answer_only()

