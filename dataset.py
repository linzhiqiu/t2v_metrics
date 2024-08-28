import os
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
import scipy
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
import cv2

def calc_pearson(metric1_scores, metric2_scores):
    pearson = 100*np.corrcoef(metric1_scores, metric2_scores)[0, 1]
    return pearson

# The original Kendall-Tau is not robust against ties. 
# We adopt the pairwise accuracy with tau optimization as proposed in EMNLP'23 Best paper
# Code borrowed from: 
# https://github.com/google-research/mt-metrics-eval/blob/main/mt_metrics_eval/ties_matter.ipynb

def _MatrixSufficientStatistics(
    x,
    y,
    epsilon: float,
    ) -> Tuple[int, int, int, int, int]:
    """Calculates tau sufficient statistics using matrices in NumPy.

    An absolute difference less than `epsilon` in x pairs is considered to be
    a tie.

    Args:
    x: Vector of numeric values.
    y: Vector of numeric values.
    epsilon: The threshold for which an absolute difference in x scores should
        be considered a tie.

    Returns:
    The number of concordant pairs, discordant pairs, pairs tied only in x,
    paired tied only in y, and pairs tied in both x and y.
    """
    x = np.asarray(x)
    x1, x2 = np.meshgrid(x, x.T)
    x_diffs = x1 - x2
    # Introduce ties into x by setting the diffs to 0 if they are <= epsilon
    x_is_tie = np.abs(x_diffs) <= epsilon
    x_diffs[x_is_tie] = 0.0
    
    y1, y2 = np.meshgrid(y, y.T)
    y_diffs = y1 - y2
    y_is_tie = y_diffs == 0.0

    n = len(y)
    num_pairs = int(scipy.special.comb(n, 2))
    # All of the counts are divided by 2 because each pair is double counted. The
    # double counted data will always be an even number, so dividing by 2 will
    # be an integer.
    con = int(
        ((x_diffs > 0) & (y_diffs > 0) | (x_diffs < 0) & (y_diffs < 0)).sum() / 2
    )
    t_x = int((x_is_tie & ~y_is_tie).sum() / 2)
    t_y = int((~x_is_tie & y_is_tie).sum() / 2)
    t_xy = int(((x_is_tie & y_is_tie).sum() - n) / 2)  # -n removes diagonal
    dis = num_pairs - (con + t_x + t_y + t_xy)
    return con, dis, t_x, t_y, t_xy


def KendallVariants(
    gold_scores,
    metric_scores,
    variant: str = 'acc23',
    epsilon: float = 0.0,
) -> Tuple[float, float]:
    """Lightweight, optionally factored versions of variants on Kendall's Tau.

    This function calculates the sufficient statistics for tau in two different
    ways, either using a Fenwick Tree (`_FenwickTreeSufficientStatistics`) when
    `epsilon` is 0 or NumPy matrices (`_MatrixSufficientStatistics`) otherwise.
    Note that the latter implementation has an O(n^2) space requirement, which
    can be significant for long vectors.

    This implementation makes several changes to the SciPy implementation of
    Kendall's tau:
    1) For the Fenwick tree version, the cython function for computing discordant
        pairs is replaced by inline python. This works up to 2x faster for small
        vectors (< 50 elements), which can be advantageous when processing many
        such vectors.
    2) The p-value calculation and associated arguments are omitted.
    3) The input vectors are assumed not to contain NaNs.

    Args:
    gold_scores: Vector of numeric values.
    metric_scores: Vector of numeric values.
    variant: Either 'b', 'c', '23', or 'acc23' to compute the respective tau
        variant. See https://arxiv.org/abs/2305.14324 for details about the
        '23' and 'acc23' variants.
    epsilon: The threshold for which an absolute difference in metric scores
        should be considered a tie.

    Returns:
    A tuple (k, 0) where the first element is the Kendall statistic and the
    second is a dummy value for compatibility with `scipy.stats.kendalltau`.
    """
    if epsilon < 0:
        raise ValueError('Epsilon must be non-negative.')
    if epsilon > 0 and variant == 'c':
        # It's not clear how to define minclasses with a non-zero epsilon.
        raise ValueError('Non-zero epsilon with tau-c not supported.')

    # The helper functions and tau_optimization expect metric_scores first, the
    # reverse of the convention used for public methods in this module.
    x, y = metric_scores, gold_scores

    assert x is not None and y is not None
    x = np.asarray(x)
    y = np.asarray(y)
    # assert no NaNs
    assert not np.any(np.isnan(x)), f"NaN found in metric_scores: {x}"
    assert not np.any(np.isnan(y)), f"NaN found in gold_scores: {y}"

    # if epsilon > 0:
    con, dis, xtie_only, ytie_only, tie_both = _MatrixSufficientStatistics(
        x, y, epsilon
    )

    size = y.size
    xtie = xtie_only + tie_both
    ytie = ytie_only + tie_both
    tot = con + dis + xtie_only + ytie_only + tie_both

    if variant in ['b', 'c'] and (xtie == tot or ytie == tot):
        return np.nan, 0

    if variant == 'b':
        tau = (con - dis) / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == 'c':
        minclasses = min(len(set(x)), len(set(y)))
        tau = 2 * (con - dis) / (size**2 * (minclasses - 1) / minclasses)
    elif variant == '23':
        tau = (con + tie_both - dis - xtie_only - ytie_only) / tot
    elif variant == 'acc23':
        tau = (con + tie_both) / tot
    else:
        raise ValueError(
            f'Unknown variant of the method chosen: {variant}. '
            "variant must be 'b', 'c', '23', or 'acc23'.")

    return tau, 0

def calc_metric(gold_scores, metric_scores, variant: str="pairwise_acc_with_tie_optimization", sample_rate=1.0):
    gold_scores = np.array(gold_scores)
    metric_scores = np.array(metric_scores)
    assert gold_scores.shape == metric_scores.shape
    if gold_scores.ndim == 1:
        # No grouping
        gold_scores = gold_scores.reshape(1, -1)
        metric_scores = metric_scores.reshape(1, -1)
    else:
        # Group by item (last dim is number of system)
        pass
    
    # Calculate metric using KendallTau (including Pairwise Accuracy)
    if variant == "pairwise_acc_with_tie_optimization":
        import tau_optimization
        result = tau_optimization.tau_optimization(metric_scores, gold_scores, tau_optimization.TauSufficientStats.acc_23, sample_rate=sample_rate)
        return result.best_tau, result.best_threshold
    elif variant == "pairwise_acc_ignore_tie":
        import tau_optimization
        result = tau_optimization.tau_optimization(metric_scores, gold_scores, tau_optimization.TauSufficientStats.acc_ignore_tie, sample_rate=sample_rate)
        return result.taus[0], result.thresholds[0]
    elif variant == 'tau_with_tie_optimization':
        import tau_optimization
        result = tau_optimization.tau_optimization(metric_scores, gold_scores, tau_optimization.TauSufficientStats.tau_23, sample_rate=sample_rate)
        return result.best_tau, result.best_threshold
    elif variant == "tau_b":
        taus = []
        for gold_score, metric_score in zip(gold_scores, metric_scores):
            tau, _ = KendallVariants(gold_score, metric_score, variant="b")
            taus.append(tau)
    elif variant == "tau_c":
        taus = []
        for gold_score, metric_score in zip(gold_scores, metric_scores):
            tau, _ = KendallVariants(gold_score, metric_score, variant="c")
            taus.append(tau)
    # average all non-Nan taus
    taus = np.array(taus)
    return np.nanmean(taus)    

def get_winoground_scores(scores_i2t):
    ids = list(range(scores_i2t.shape[0]))
    winoground_scores = []
    for id, score_i2t in zip(ids, scores_i2t):
        winoground_scores.append({
            "id" : id,
            "c0_i0": score_i2t[0][0],
            "c0_i1": score_i2t[1][0],
            "c1_i0": score_i2t[0][1],
            "c1_i1": score_i2t[1][1]}
        )
    return winoground_scores

def get_winoground_acc(scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)
    
    for result in scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    result = {
        'text': text_correct_count/denominator,
        'image': image_correct_count/denominator,
        'group': group_correct_count/denominator,
    }
    return result


class Winoground(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir='./',
                 return_image_paths=True):
        self.root_dir = os.path.join(root_dir, "winoground")
        if not os.path.exists(self.root_dir):
            subprocess.call(
                ["gdown", "--no-cookies", "1Lril_90vjsbL_2qOaxMu3I-aPpckCDiF", "--output",
                 os.path.join(root_dir, "winoground.zip")]
            )
            subprocess.call(
                ["unzip", "-q", "winoground.zip"],
                cwd=root_dir
            )
        csv_file = os.path.join(self.root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(csv_file).to_dict(orient='records')
        json_file = os.path.join(self.root_dir, 'examples.jsonl')
        self.winoground = [json.loads(line) for line in open(json_file, 'r')]
        self.return_image_paths = return_image_paths
        if return_image_paths:
            assert image_preprocess is None
            self.preprocess = None
            
        self.preprocess = image_preprocess
        self.original_tags = self.get_original_tags()
        self.new_tags = self.get_new_tags(path=os.path.join(self.root_dir, "why_winoground_hard.json"))
    
    def __len__(self):
        return len(self.winoground)

    def __getitem__(self, idx):
        assert self.metadata[idx]['id'] == idx
        image_0_path = os.path.join(self.root_dir, self.metadata[idx]['image_0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[idx]['image_1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.preprocess(self.image_loader(image_0_path))
            image_1 = self.preprocess(self.image_loader(image_1_path))
        
        caption_0 = self.metadata[idx]['caption_0']
        caption_1 = self.metadata[idx]['caption_1']
        item = {
            "images": [image_0, image_1],
            "texts": [caption_0, caption_1]
        }
        return item
    
    def get_original_tags(self):
        tags = {}
        for example in self.winoground:
            if example['num_main_preds'] == 1:
                if '1 Main Pred' not in tags:
                    tags["1 Main Pred"] = []
                tags['1 Main Pred'].append(example["id"])
            elif example['num_main_preds'] == 2:
                if '2 Main Pred' not in tags:
                    tags["2 Main Pred"] = []
                tags['2 Main Pred'].append(example["id"])
            else:
                # This won't happen
                raise ValueError(f"num_main_preds: {example['num_main_preds']}")
            if example["collapsed_tag"] not in tags:
                tags[example["collapsed_tag"]] = []
            tags[example["collapsed_tag"]].append(example["id"])
        return tags

    def get_new_tags(self, path="./why_winoground_hard.json"):
        new_tag_dict = json.load(open(path))
        tags = {}
        for idx in new_tag_dict:
            curr_tags = new_tag_dict[idx]
            if len(curr_tags) == 0:
                if "No Tag" not in tags:
                    tags["No Tag"] = []
                tags["No Tag"].append(int(idx))
            for tag in curr_tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(int(idx))
        return tags
    
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("Winoground performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'Winoground': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        results = {}
        results['all'] = acc
        for tag in self.original_tags:
            results[tag] = get_winoground_acc([winoground_scores[i] for i in self.original_tags[tag]])
        for tag in self.new_tags:
            results[tag] = get_winoground_acc([winoground_scores[i] for i in self.new_tags[tag]])
            # print(f"Winoground {tag} text score: {results[tag]['text']}")
            # print(f"Winoground {tag} image score: {results[tag]['image']}")
            # print(f"Winoground {tag} group score: {results[tag]['group']}")
        return results


class SeeTrue(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True):
        import pandas as pd
        self.root_dir = os.path.join(root_dir, 'seetrue')
        if not os.path.exists(self.root_dir):
            if download:
                os.makedirs(self.root_dir, exist_ok=True)
                import subprocess
                image_zip_file = os.path.join(root_dir, "images.zip")
                subprocess.call(
                    ["wget", "https://huggingface.co/datasets/yonatanbitton/SeeTRUE/resolve/main/images.zip",
                     image_zip_file], cwd=self.root_dir
                )
                env = os.environ.copy()
                env["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
                
                subprocess.call(["unzip", "images.zip"], cwd=self.root_dir, env=env)
        
        csv_path = os.path.join("datasets", "SeeTRUE.csv")
        if not os.path.exists(csv_path):
            subprocess.call(
                ["wget", "https://huggingface.co/datasets/yonatanbitton/SeeTRUE/resolve/main/SeeTRUE.csv"],
                cwd="datasets"
            )
        self.dataset = pd.read_csv(os.path.join("datasets", "SeeTRUE.csv"))
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        
        
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset.image[idx]
        
        image_path = os.path.join(self.root_dir, 'images', image_path)
        if self.return_image_paths:
            image = image_path
        else:
            image = Image.open(image_path).convert('RGB')
            image = self.image_preprocess(image)
        
        texts = [str(self.dataset.text[idx])]
        item = {"images": [image], "texts": texts}
        return item
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        human_avg_scores = self.dataset.label.to_list()
        our_scores = [float(scores_i2t[idx][0][0]) for idx in range(len(self.dataset))]
        import math
        human_avg_scores_without_nan = []
        our_scores_without_nan = []
        for idx in range(len(self.dataset)):
            if math.isnan(our_scores[idx]):
                print(f"Warning: {idx} has nan score! Skipping this for evaluation")
                import pdb; pdb.set_trace()
                continue
            human_avg_scores_without_nan.append(human_avg_scores[idx])
            our_scores_without_nan.append(our_scores[idx])
        
        ''' Calc ROC_AUC score per dataset_source group '''
        stats = []
        for dataset_source, df_dataset in self.dataset.groupby('dataset_source'):
            num_samples = len(df_dataset)
            subset_indices = df_dataset.index
            num_pos = df_dataset['label'][subset_indices].sum()
            num_neg = num_samples - num_pos
            roc_auc = roc_auc_score(df_dataset['label'][subset_indices], np.array(our_scores_without_nan)[subset_indices])
            stats.append([dataset_source, num_samples, num_pos, num_neg, roc_auc])
        df_stats = pd.DataFrame(stats, columns=['dataset_source', 'num_samples', 'num_pos', 'num_neg', 'roc_auc'])
        print(df_stats)
        results = {
            'per_dataset_source': df_stats,
        }
        return results


class TIFA160_DSG(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True):
        self.root_dir = os.path.join(root_dir, 'tifa160')
        if not os.path.exists(self.root_dir):
            if download:
                os.makedirs(root_dir, exist_ok=True)
                import subprocess
                image_zip_file = os.path.join(root_dir, "tifa160.zip")
                subprocess.call(
                    ["gdown", "--no-cookies", "1hHVMeVDZlnJz1FFhy_BxiZGIz1tEMm0s", "--output",
                     image_zip_file]
                )
                subprocess.call(["unzip", "-q", "tifa160.zip"], cwd=root_dir)
        
        self.dataset = json.load(open(os.path.join("datasets", "tifa160.json"), 'r'))
        self.dsg_human_likert_scores = pd.read_csv(os.path.join("datasets", "dsg_tifa160_anns.csv"))
        self.model_type_to_names = { # map from dsg format to tifa format
            'mini-dalle': 'mini_dalle',
            'vq-diffusion': 'vq_diffusion',
            'sd1dot5': 'stable_diffusion_v1_5',
            'sd2dot1': 'stable_diffusion_v2_1',
            'sd1dot1': 'stable_diffusion_v1_1',
        }
        self.model_types = [self.model_type_to_names[m_name] for m_name in self.dsg_human_likert_scores['model_type'].to_list()]
        self.source_ids = self.dsg_human_likert_scores['source_id'].to_list()
        self.keys = [f"{self.source_ids[idx]}_{self.model_types[idx]}" for idx in range(len(self.model_types))]
        self.path_names = [f"{k}.jpg" for k in self.keys]
        # self.item_ids = self.dsg_human_likert_scores['item_id'].to_list()
        self.answers = self.dsg_human_likert_scores['answer'].to_list()
        self.dsg_items = {}
        for key_idx, k in enumerate(self.keys):
            if k in self.dsg_items:
                self.dsg_items[k]['human_scores'].append(self.answers[key_idx])
            else:
                self.dsg_items[k] = {
                    'human_scores': [self.answers[key_idx]],
                    'text': self.dataset[self.keys[key_idx]]['text'],
                    'image_path': self.path_names[key_idx],
                    'text_id': self.source_ids[key_idx],
                }
        
        self.image_preprocess = image_preprocess
        self.items = list(self.dataset.keys())
        self.return_image_paths = return_image_paths
        self.all_samples = {} # key is text_id, value is all indices (for each diffusion model) and human scores
        # compute 'human_avg'
        for _, k in enumerate(self.dsg_items):
            self.dsg_items[k]['human_avg'] = float(np.mean(self.dsg_items[k]['human_scores']))
            assert self.dsg_items[k]['text_id'] == self.dataset[k]['text_id']
            assert self.dsg_items[k]['text'] == self.dataset[k]['text']
            assert self.dsg_items[k]['image_path'] == self.dataset[k]['image_path']
            text_id = self.dsg_items[k]['text_id']
            k_idx = self.items.index(k)
            if text_id not in self.all_samples:
                self.all_samples[text_id] = {
                    'text_id': text_id,
                    'text': self.dsg_items[k]['text'],
                    'indices': [k_idx],
                }
            else:
                self.all_samples[text_id]['indices'].append(k_idx)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        k = self.items[idx]
        item = self.dataset[k]
        
        image_path = os.path.join(self.root_dir, item['image_path'])
        if self.return_image_paths:
            image = image_path
        else:
            image = Image.open(image_path).convert('RGB')
            image = self.image_preprocess(image)
        
        texts = [str(item['text'])]
        item = {"images": [image], "texts": texts}
        return item
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        human_avg_scores = self.get_metric_scores('human_avg')
        our_scores = [float(scores_i2t[idx][0][0]) for idx in range(len(self.items))]
        import math
        human_avg_scores_without_nan = []
        our_scores_without_nan = []
        for idx in range(len(self.items)):
            if math.isnan(our_scores[idx]):
                print(f"Warning: {self.items[idx]} has nan score! Skipping this for evaluation")
                continue
            human_avg_scores_without_nan.append(human_avg_scores[idx])
            our_scores_without_nan.append(our_scores[idx])
        pearson_no_grouping = calc_pearson(human_avg_scores_without_nan, our_scores_without_nan)
        print(f"Pearson's Correlation (no grouping): ", pearson_no_grouping)
        
        kendall_b_no_grouping = calc_metric(human_avg_scores_without_nan, our_scores_without_nan, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b_no_grouping)
        
        pairwise_acc_no_grouping = calc_metric(human_avg_scores_without_nan, our_scores_without_nan, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc_no_grouping)
        
        # check accuracy of the score picking the highest human score
        # human_scores_group_by_item = None
        # our_scores_group_by_item = None
        # for idx, text_id in enumerate(self.all_samples):
        #     indices = self.all_samples[text_id]['indices']
        #     if human_scores_group_by_item is None:
        #         human_scores_group_by_item = np.zeros((len(self.all_samples), len(indices)))
        #         our_scores_group_by_item = np.zeros((len(self.all_samples), len(indices)))
            
        #     human_scores_group_by_item[idx] = [human_avg_scores[idx] for idx in indices]
        #     our_scores_group_by_item[idx] = [our_scores[idx] for idx in indices]

            
        # kendall_b_group_by_item = calc_metric(human_scores_group_by_item, our_scores_group_by_item, variant="tau_b")
        # print(f'Kendall Tau-B Score (group by item): ', kendall_b_group_by_item)
        
        # pairwise_acc_group_by_item = calc_metric(human_scores_group_by_item, our_scores_group_by_item, variant="pairwise_acc_with_tie_optimization")
        # print(f'Pairwise Accuracy Score (group by item): ', pairwise_acc_group_by_item)
        
        # return pearson_no_grouping, kendall_b_no_grouping, kendall_c_no_grouping, kendall_no_grouping, pairwise_acc_no_grouping, kendall_b_group_by_item, kendall_c_group_by_item, kendall_group_by_item, pairwise_acc_group_by_item
        # return a dictionary
        results = {
            'pearson_no_grouping': pearson_no_grouping,
            'kendall_b_no_grouping': kendall_b_no_grouping,
            'pairwise_acc_no_grouping': pairwise_acc_no_grouping,
            # 'kendall_b_group_by_item': kendall_b_group_by_item,
            # 'pairwise_acc_group_by_item': pairwise_acc_group_by_item,
        }
        return results

    
    def get_metric_scores(self, metric):
        if metric == 'human_avg':
            return [self.dsg_items[k][metric] for k in self.items]
        return [self.dataset[k][metric] for k in self.items]

    
class Flickr8K_CF(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True,
                 json_path="crowdflower_flickr8k.json"):
        self.root_dir = root_dir
        if not os.path.exists(os.path.join(root_dir, 'flickr8k')):
            if download:
                os.makedirs(root_dir, exist_ok=True)
                import subprocess
                image_zip_file = os.path.join(root_dir, "flickr8k.zip")
                subprocess.call(
                    ["gdown", "--no-cookies", "1WEg-xbUZ971P3Q0RDA8nVfKJrtpjTqCM", "--output",
                     image_zip_file]
                )
                subprocess.call(["unzip", "-q", "flickr8k.zip"], cwd=root_dir)
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        self.dataset = json.load(open(os.path.join(self.root_dir, "flickr8k", json_path), 'r'))
        print('Loaded {} images'.format(len(self.dataset)))

        self.images = []
        self.refs = []
        self.candidates = []
        self.human_scores = []
        self.all_samples = {} # key is image_id, value is all indices (for each generated caption) and human scores
        for k, v in list(self.dataset.items()):
            for human_judgement in v['human_judgement']:
                if np.isnan(human_judgement['rating']):
                    raise ValueError(f"Human judgement score is nan for {k}")
                self.images.append(os.path.join(self.root_dir, "flickr8k", v['image_path']))
                self.refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
                self.candidates.append(' '.join(human_judgement['caption'].split()))
                self.human_scores.append(human_judgement['rating'])
                if k not in self.all_samples:
                    self.all_samples[k] = {
                        'image_id': k,
                        'indices': [len(self.images) - 1],
                    }
                else:
                    self.all_samples[k]['indices'].append(len(self.images) - 1)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.return_image_paths:
            image = image_path
        else:
            image = Image.open(image_path).convert('RGB')
            image = self.image_preprocess(image)
        texts = [self.candidates[idx]]
        for i in range(len(texts)):
            texts[i] = texts[i].strip(".").strip(" ")
        item = {"images": [image], "texts": texts}
        return item
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        human_avg_scores = self.human_scores
        our_scores = [float(scores_i2t[idx][0][0]) for idx in range(len(self.images))]
        import math
        human_avg_scores_without_nan = []
        our_scores_without_nan = []
        for idx in range(len(self.images)):
            if math.isnan(our_scores[idx]):
                print(f"Warning: {self.images[idx]} has nan score! Skipping this for evaluation")
                continue
            human_avg_scores_without_nan.append(human_avg_scores[idx])
            our_scores_without_nan.append(our_scores[idx])
        pearson_no_grouping = calc_pearson(human_avg_scores_without_nan, our_scores_without_nan)
        print(f"Pearson's Correlation (no grouping): ", pearson_no_grouping)
        
        kendall_b_no_grouping = calc_metric(human_avg_scores_without_nan, our_scores_without_nan, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b_no_grouping)
        
        pairwise_acc_no_grouping = calc_metric(human_avg_scores_without_nan, our_scores_without_nan, variant="pairwise_acc_with_tie_optimization", sample_rate=0.1)
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc_no_grouping)
        
        results = {
            'pearson_no_grouping': pearson_no_grouping,
            'kendall_b_no_grouping': kendall_b_no_grouping,
            'pairwise_acc_no_grouping': pairwise_acc_no_grouping,
        }
        return results


class EqBen_Mini(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir='./',
                 return_image_paths=True):
        self.preprocess = image_preprocess
        
        self.root_dir = os.path.join(root_dir, "eqben_vllm")
        if not os.path.exists(self.root_dir):
            # https://drive.google.com/file/d/11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM/view?usp=sharing
            os.makedirs(self.root_dir, exist_ok=True)
            subprocess.call(
                ["gdown", "--no-cookies", "11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM", "--output", 
                 os.path.join(self.root_dir, "eqben_vllm.zip")]
            )
            subprocess.call(["unzip", "-q", "eqben_vllm.zip"], cwd=self.root_dir)
            
        self.root_dir = os.path.join(root_dir, "eqben_vllm", "images")
        self.subset_types = {
            'eqbensd': ['eqbensd'],
            'eqbenk': ['eqbenkubric_cnt', 'eqbenkubric_loc', 'eqbenkubric_attr'],
            'eqbeng': ['eqbengebc'],
            'eqbenag': ['eqbenag'],
            'eqbeny': ['eqbenyoucook2'],
        }
        json_file = os.path.join(root_dir, "eqben_vllm", "all_select.json")
        self.metadata = json.load(open(json_file, 'r'))
        self.subset_indices = {subset_type: [] for subset_type in self.subset_types}
        for item_idx, item in enumerate(self.metadata):
            image_path = item['image0']
            for subset_type in self.subset_types:
                if image_path.split('/')[0] in self.subset_types[subset_type]:
                    self.subset_indices[subset_type].append(item_idx)
                    break
        
        self.return_image_paths = return_image_paths
        self.transform = image_preprocess
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
     
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        image_0_path = os.path.join(self.root_dir, self.metadata[index]['image0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[index]['image1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.transform(self.image_loader(image_0_path))
            image_1 = self.transform(self.image_loader(image_1_path))
        
        caption_0 = self.metadata[index]['caption0']
        caption_1 = self.metadata[index]['caption1']
        item = {"images": [image_0, image_1], "texts": [caption_0, caption_1]}
        return item
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("EQBen_Mini performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'EQBen_Mini': <70} {acc['text']: <10.2%} {acc['image']: <10.2%} {acc['group']: <10.2%}")
        results = {}
        results['all'] = acc
        for subset_type in self.subset_types:
            subset_indices = self.subset_indices[subset_type]
            subset_scores = [winoground_scores[idx] for idx in subset_indices]
            subset_acc = get_winoground_acc(subset_scores)
            print(f"{'EQBen_Mini ' + subset_type: <70} {subset_acc['text']: <10.2%} {subset_acc['image']: <10.2%} {subset_acc['group']: <10.2%}")
            results[subset_type] = subset_acc
        return results


class T2VScore(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True,
                 image_save_dir="t2vscore_images",
                 num_frames=36,
                 eval_mode='avg_frames',
                 extract_videos=False):
        self.root_dir = os.path.join(root_dir, 't2vscore')
        self.models = ['floor33', 'gen2', 'pika', 'modelscope', 'zeroscope']
        self.eval_mode = eval_mode
        self.download_links = {
            'floor33': 'https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset/resolve/main/floor33.tar.gz',
            'gen2': 'https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset/resolve/main/gen2_december.tar.gz',
            'pika': 'https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset/resolve/main/pika_v1_december.tar.gz',
            'modelscope': 'https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset/resolve/main/modelscope.tar.gz',
            'zeroscope': 'https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset/resolve/main/zeroscope.tar.gz',
        }
        if not os.path.exists(self.root_dir):
            if download:
                os.makedirs(self.root_dir, exist_ok=True)
                import subprocess
                for model in self.models:
                    model_file_name = self.download_links[model].split('/')[-1]
                    model_name = model_file_name.split('.tar.gz')[0]
                    image_zip_file = os.path.join(self.root_dir, model_file_name)
                    if not os.path.exists(image_zip_file):
                        subprocess.call(
                            ["wget", self.download_links[model], "-O", model_file_name], cwd=self.root_dir
                        )
                    model_dir = os.path.join(self.root_dir, model)
                    if not os.path.exists(model_dir):
                        subprocess.call(["tar", "-xvf", model_file_name], cwd=self.root_dir)
                        if model_name != model:
                            if model_name == 'pika_v1_december':
                                model_name = 'pika_v1_december_1' # Because the naming is off
                            subprocess.call(["mv", model_name, model], cwd=self.root_dir)
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
        self.image_save_dir = os.path.join(root_dir, image_save_dir)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir, exist_ok=True)
                        
        self.dataset = json.load(open(os.path.join("datasets", "t2vscore_alignment_score.json"), 'r'))
        self.dataset_quality = json.load(open(os.path.join("datasets", "t2vscore_quality_score.json"), 'r'))
        
        t2v_videos_file = os.path.join(self.root_dir, "t2v_videos.json")
        t2v_prompt_to_videos_file = os.path.join(self.root_dir, "t2v_prompt_to_videos.json")
        if os.path.exists(t2v_videos_file) and os.path.exists(t2v_prompt_to_videos_file) and not extract_videos:
            self.videos = json.load(open(t2v_videos_file, 'r'))
            self.prompt_to_videos = json.load(open(t2v_prompt_to_videos_file, 'r'))
            print(f"Load from pre-extracted folder (which converted videos into sequence of images)")
            print(f"If you modify the dataset class, please re-extract the videos by setting extract_videos=True")
            return

        self.videos = [] # list of videos
        self.prompt_to_videos = {}
        for model in self.models:
            model_dir = os.path.join(self.image_save_dir, model)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            for prompt_idx in self.dataset:
                if not model in self.dataset[prompt_idx]['models']:
                    continue
                # ensure at least one human rating for all models, otherwise skip
                if len(self.dataset[prompt_idx]['models'][model]) == 0:
                    continue
                else:
                    for m in self.models:
                        if m in self.dataset[prompt_idx]['models']:
                            assert len(self.dataset[prompt_idx]['models'][m]) > 0
                
                video_path = os.path.join(self.root_dir, model, f"{int(prompt_idx):04d}.mp4")
                cap = cv2.VideoCapture(video_path)
                current_frames = []
                while True:
                    # Read a frame from the video
                    ret, frame = cap.read()
                    
                    # If the frame was not retrieved, we've reached the end of the video
                    if not ret:
                        break
                    
                    # Convert the color space from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert the frame to a PIL Image
                    img_pil = Image.fromarray(frame_rgb)
                    
                    output_path = os.path.join(model_dir, f"video_{int(prompt_idx):04d}_frame_{len(current_frames):04d}.jpg")
                    if not os.path.exists(output_path):
                        img_pil.save(output_path)
                    
                    # Append the PIL Image path to the list
                    current_frames.append(output_path)
                
                if len(current_frames) < num_frames:
                    current_frames = current_frames + [current_frames[-1]] * (num_frames - len(current_frames))
                elif len(current_frames) > num_frames:
                    current_frames = current_frames[:num_frames]
                
                # Sample 4 frames (including the first) uniformly from the video
                sample_4_frames = [current_frames[0], current_frames[num_frames // 3], current_frames[num_frames // 3 * 2], current_frames[-1]]
                if len(current_frames) < 4:
                    raise ValueError(f"Video {video_path} has less than 4 frames")
                
                # Release the video capture object
                cap.release()
                self.videos.append({
                    'prompt_idx': prompt_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': model,
                    'video_path': video_path,
                    'num_frames': len(current_frames),
                    'frames': current_frames,
                    'sample_4_frames': sample_4_frames,
                    'human_alignment': self.dataset[prompt_idx]['models'][model],
                    'human_quality': self.dataset_quality[prompt_idx]['models'][model],
                })
                if prompt_idx not in self.prompt_to_videos:
                    self.prompt_to_videos[prompt_idx] = []
                self.prompt_to_videos[prompt_idx].append(len(self.videos) - 1)
        print(f"Number of frames: {num_frames}")
        json.dump(self.videos, open(t2v_videos_file, 'w'))
        json.dump(self.prompt_to_videos, open(t2v_prompt_to_videos_file, 'w'))
                
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        item = self.videos[idx]
        
        image_paths = item['frames']
        if self.eval_mode == 'avg_frames':
            pass
        elif self.eval_mode == 'first_frame':
            image_paths = [image_paths[0]]
        elif self.eval_mode == 'last_frame':
            image_paths = [image_paths[-1]]
        elif self.eval_mode == 'sample_4_frame':
            image_paths = item['sample_4_frames']
        else:
            raise ValueError(f"Invalid eval_mode: {self.eval_mode}")
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    
    def get_scores_from_author(self, model='CLIP Score'):
        score_file = "datasets/t2vscore_results.csv"
        scores_file = pd.read_csv(score_file).to_dict(orient='records')
        scores_dict = {}
        for _, item in enumerate(scores_file):
            video_id = str(item['video_id'])
            model_name = item['model_name']
            prompt = item['prompt']
            if not video_id in scores_dict:
                scores_dict[video_id] = {'prompt': prompt, 'models': {}}
            scores_dict[video_id]['models'][model_name] = item[model]
        scores = []
        for item in self.videos:
            scores.append(scores_dict[item['prompt_idx']]['models'][item['model']])
        return np.array(scores).reshape(-1, 1, 1)
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'pairwise_acc': pairwise_acc,
        }
        return results
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        # take average of all frames
        human_avg_scores_alignment = [np.array(self.videos[idx]['human_alignment']).mean() for idx in range(len(self.videos))]
        # human_avg_scores_quality = [np.array(self.videos[idx]['human_quality']).mean() for idx in range(len(self.videos))]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in range(len(self.videos))]
        alignment_correlation = self.correlation(our_scores, human_avg_scores_alignment)
        # quality_correlation = self.correlation(our_scores, human_avg_scores_quality)
        results = {
            'alignment': alignment_correlation,
            # 'quality': quality_correlation,
        }
        return results


class StanfordT23D(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True,
                 image_save_dir="stanfordt23d_images",
                 num_views=120,
                 eval_mode='rgb_grid_3_x_3',
                 extract_images=False):
        self.root_dir = os.path.join(root_dir, 'stanfordt23d')
        self.models = ['dreamfusion', 'instant3d', 'latent-nerf', 'magic3d', 'mvdream',' shap-e']
        self.views_four = [5, 35, 65, 95]
        self.views_nine = [2, 15, 28, 41, 54, 67, 80, 93, 106]
        self.eval_mode = eval_mode
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
        self.image_save_dir = os.path.join(root_dir, image_save_dir)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir, exist_ok=True)
        
        if not os.path.exists(self.root_dir):
            if download:
                import subprocess
                download_link = "https://huggingface.co/datasets/zhiqiulin/vqascore_ablation/resolve/main/stanfordt23d.zip"
                model_file_name = download_link.split('/')[-1]
                image_zip_file = os.path.join(self.root_dir, model_file_name)
                if not os.path.exists(image_zip_file):
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=root_dir
                    )
                subprocess.call(["unzip", "-q", model_file_name], cwd=root_dir)
                        
        self.dataset = json.load(open(os.path.join("datasets", "stanfordt23d.json"), 'r'))
        self.num_views = num_views
        stanford_t23d_file = os.path.join(self.root_dir, "stanfordt23d_images.json")
        stanford_t23d_prompt_to_images_file = os.path.join(self.root_dir, "stanfordt23d_prompt_to_images.json")
        if os.path.exists(stanford_t23d_file) and os.path.exists(stanford_t23d_prompt_to_images_file) and not extract_images:
            self.images = json.load(open(stanford_t23d_file, 'r'))
            self.prompt_to_images = json.load(open(stanford_t23d_prompt_to_images_file, 'r'))
            print(f"Load from pre-extracted folder (which converted grid images into individual 2d rgb/normal images)")
            print(f"If you modify the dataset class, please re-extract the grid images by setting extract_images=True")
            return

        self.images = [] # list of videos
        self.prompt_to_images = {}
        for model in self.models:
            model_dir = os.path.join(self.image_save_dir, model)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            for prompt_idx in self.dataset:
                if not model in self.dataset[prompt_idx]['models']:
                    continue
                # ensure at least one human rating for all models, otherwise skip
                if len(self.dataset[prompt_idx]['models'][model]) == 0:
                    continue
                else:
                    for m in self.models:
                        if m in self.dataset[prompt_idx]['models']:
                            assert len(self.dataset[prompt_idx]['models'][m]) > 0
                
                folder_path = os.path.join(self.root_dir, model, str(prompt_idx), "0")
                rgb_views = []
                normal_views = []
                for view in range(num_views):
                    rgb_view_path = os.path.join(folder_path, f"rgb_{view}.jpg")
                    rgb_views.append(rgb_view_path)
                    normal_view_path = os.path.join(folder_path, f"normal_{view}.jpg")
                    normal_views.append(normal_view_path)
                    assert os.path.exists(rgb_view_path)
                    assert os.path.exists(normal_view_path)
                    
                sample_4_rgb_views = [rgb_views[view] for view in self.views_four]
                sample_9_rgb_views = [rgb_views[view] for view in self.views_nine]
                sample_4_normal_views = [normal_views[view] for view in self.views_four]
                sample_9_normal_views = [normal_views[view] for view in self.views_nine]
                
                img_pil = Image.open(sample_4_rgb_views[0]).convert('RGB')
                image_width, image_height = img_pil.size
                
                for grid_size, sample_rgb_views, sample_normal_views in [
                    (2, sample_4_rgb_views, sample_4_normal_views),
                    (3, sample_9_rgb_views, sample_9_normal_views)
                    ]:
                    new_image_rgb = Image.new('RGB', (image_width * grid_size, image_height * grid_size))
                    new_image_normal = Image.new('RGB', (image_width * grid_size, image_height * grid_size))
                    for grid_frame_i in range(grid_size * grid_size):
                        frame_grid_rgb = Image.open(sample_rgb_views[grid_frame_i]).convert('RGB')
                        new_image_rgb.paste(frame_grid_rgb, (image_width * (grid_frame_i % grid_size), image_height * (grid_frame_i // grid_size)))
                        frame_grid_normal = Image.open(sample_normal_views[grid_frame_i]).convert('RGB')
                        new_image_normal.paste(frame_grid_normal, (image_width * (grid_frame_i % grid_size), image_height * (grid_frame_i // grid_size)))
                    grid_image_path_rgb = os.path.join(model_dir, f"rgb_{int(prompt_idx)}_grid_{grid_size}x{grid_size}.jpg")
                    grid_image_path_normal = os.path.join(model_dir, f"normal_{int(prompt_idx)}_grid_{grid_size}x{grid_size}.jpg")
                    if not os.path.exists(grid_image_path_rgb):
                        new_image_rgb.save(grid_image_path_rgb)
                    if not os.path.exists(grid_image_path_normal):
                        new_image_normal.save(grid_image_path_normal)
                        
                self.images.append({
                    'prompt_idx': prompt_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': model,
                    'folder_path': folder_path,
                    'num_views': len(rgb_views),
                    'rgb_views': rgb_views,
                    'normal_views': normal_views,
                    'sample_4_rgb_views': sample_4_rgb_views,
                    'sample_9_rgb_views': sample_9_rgb_views,
                    'sample_4_normal_views': sample_4_normal_views,
                    'sample_9_normal_views': sample_9_normal_views,
                    'rgb_grid_2_x_2': [os.path.join(model_dir, f"rgb_{int(prompt_idx)}_grid_2x2.jpg")],
                    'normal_grid_2_x_2': [os.path.join(model_dir, f"normal_{int(prompt_idx)}_grid_2x2.jpg")],
                    'rgb_grid_3_x_3': [os.path.join(model_dir, f"rgb_{int(prompt_idx)}_grid_3x3.jpg")],
                    'normal_grid_3_x_3': [os.path.join(model_dir, f"normal_{int(prompt_idx)}_grid_3x3.jpg")],
                    'human_alignment': self.dataset[prompt_idx]['models'][model],
                })
                if prompt_idx not in self.prompt_to_images:
                    self.prompt_to_images[prompt_idx] = []
                self.prompt_to_images[prompt_idx].append(len(self.images) - 1)
        print(f"Number of views: {num_views}")
        json.dump(self.images, open(stanford_t23d_file, 'w'))
        json.dump(self.prompt_to_images, open(stanford_t23d_prompt_to_images_file, 'w'))
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        
        assert self.eval_mode in item, f"Invalid eval_mode: {self.eval_mode}"
        image_paths = item[self.eval_mode]
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'pairwise_acc': pairwise_acc,
        }
        return results
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        # take average of all frames
        human_avg_scores_alignment = [np.array(self.images[idx]['human_alignment']).mean() for idx in range(len(self.images))]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in range(len(self.images))]
        alignment_correlation = self.correlation(our_scores, human_avg_scores_alignment)
        results = {
            'alignment': alignment_correlation,
        }
        return results


class Pickapic_v1(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir='./',
                 return_image_paths=True,
                 download=True):
        self.root_dir = os.path.join(root_dir, "pickapic_v1")
        
        if not os.path.exists(self.root_dir):
            if download:
                import subprocess
                download_link = "https://huggingface.co/datasets/zhiqiulin/vqascore_ablation/resolve/main/pickapic_v1.zip"
                model_file_name = download_link.split('/')[-1]
                image_zip_file = os.path.join(self.root_dir, model_file_name)
                if not os.path.exists(image_zip_file):
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=root_dir
                    )
                subprocess.call(["unzip", "-q", model_file_name], cwd=root_dir)
                        
        self.data_path = os.path.join(self.root_dir, "test_captions.json")

        with open(self.data_path, 'r') as f:
            self.all_data = json.load(f)

        self.selected_idxs = [1, 9, 385, 14, 138, 5, 31, 33, 39, 352, 21, 417, 399, 17, 82, 412, 78, 
            53, 54, 59, 60, 308, 76, 142, 98, 259, 317, 110, 113, 118, 112, 119, 144, 148, 149, 153, 
            159, 162, 172, 111, 124, 196, 197, 220, 35, 141, 252, 475, 368,
            214, 150, 43, 221, 163, 228, 236, 57, 326, 257, 266, 268, 62, 274, 277, 278, 281, 105, 285,
            286, 301, 419, 91, 312, 316, 318, 319, 334, 335, 339, 340, 347, 350, 367,
            374, 375, 382, 376, 387, 345, 405, 411, 478, 441, 444, 99, 384, 472, 479, 490, 493]
        self.dataset = []
        for new_id, seleted_id in enumerate(self.selected_idxs):
            assert seleted_id == self.all_data[seleted_id]['id']
            item = {
                'id': new_id,
                "caption": self.all_data[seleted_id]["caption"],
                "label_0": self.all_data[seleted_id]["label_0"],
                "label_1": self.all_data[seleted_id]["label_1"],
                "image_0": self.all_data[seleted_id]["image_0"],
                "image_1": self.all_data[seleted_id]["image_1"]
            }
            self.dataset.append(item)

        self.return_image_paths = return_image_paths
        self.preprocess = image_preprocess
        

    def open_image(self, image):
        image = Image.open(image)
        image = image.convert("RGB")
        return image                 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image_0_path = os.path.join(self.root_dir, self.dataset[idx]['image_0'])
        image_1_path = os.path.join(self.root_dir, self.dataset[idx]['image_1'])

        caption = self.dataset[idx]['caption']

        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = self.open_image(image_0_path)
            image_1 = self.open_image(image_1_path)
            if self.preprocess:
                image_0 = self.preprocess(image_0)
                image_1 = self.preprocess(image_1)
        item ={
            "images":[image_0, image_1],
            "texts":[caption]
        }
        return item
    
    def calc_acc(self, probs, ds):
 
        def get_label(example):
            if example["label_0"] == 1:
                label = "0"
            else:
                label = "1"
            return label
        
        def get_pred(prob_0, prob_1):
            if prob_0 >= prob_1:
                pred = "0"
            else:
                pred = "1"
            return pred        
        
        res = []
        for example, (prob_0, prob_1) in zip(ds, probs):
            label = get_label(example)
            pred = get_pred(prob_0, prob_1)
            if pred == label:
                res.append(1)
            else:
                res.append(0)
        return sum(res) / len(res)
    
    def evaluate_scores(self, scores):
        scores = scores.transpose(1, 2).cpu().tolist()
        probs = []
        for idx in range(len(scores)):
            probs.append((scores[idx][0][0],scores[idx][0][1]))
        acc = self.calc_acc(probs, self.dataset)
        print("ACC:", acc)
        return acc, probs


class GenAI_Bench_Image_800(Dataset):
    # GenAIBench with 800 prompts x 6 images (from 6 models)
    def __init__(self,
                 image_preprocess=None,
                 root_dir="/data3/baiqil/GenAI-1600/",
                 download=True,
                 return_image_paths=True):
        self.root_dir = os.path.join(root_dir, 'GenAI_corePrompts')
        self.models = ['DALLE_3', 'SDXL_Turbo', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base']
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
        
        # self.download_links = {
        #     model_name: f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/{model_name}.zip" for model_name in self.models
        # }
        if not os.path.exists(self.root_dir):
            import pdb; pdb.set_trace()
            if download:
                import subprocess
                os.makedirs(self.root_dir, exist_ok=True)
                for model in self.models:
                    model_file_name = self.download_links[model].split('/')[-1]
                    image_zip_file = os.path.join(self.root_dir, model_file_name)
                    if not os.path.exists(image_zip_file):
                        subprocess.call(
                            ["wget", self.download_links[model], "-O", model_file_name], cwd=self.root_dir
                        )
                    model_dir = os.path.join(self.root_dir, model)
                    if not os.path.exists(model_dir):
                        subprocess.call(["unzip", "-q", model_file_name], cwd=self.root_dir)
        
        for filename in ['genai_image', 
                        #  'genai_skills'
                         ]:
            path = os.path.join(self.root_dir, f"{filename}.json")
            if not os.path.exists(path):
                import pdb; pdb.set_trace()
                if download:
                    import subprocess
                    download_link = f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/{filename}.json"
                    model_file_name = download_link.split('/')[-1]
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=self.root_dir
                    )
        
        self.dataset = json.load(open(os.path.join(self.root_dir, f"genai_image.json"), 'r'))
        print(f"Loaded dataset: genai_image.json with {len(self.dataset)} prompts")
        
        self.images = [] # list of images
        self.prompt_to_images = {}
        for model in self.models:
            for prompt_idx in self.dataset:
                if not model in self.dataset[prompt_idx]['models']:
                    print(f"Prompt {prompt_idx} does not have model {model}")
                    continue
                
                self.images.append({
                    'prompt_idx': prompt_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': model,
                    'image': os.path.join(self.root_dir, model, f"{prompt_idx}.jpeg"),
                    'human_alignment': self.dataset[prompt_idx]['models'][model],
                })
                if prompt_idx not in self.prompt_to_images:
                    self.prompt_to_images[prompt_idx] = []
                self.prompt_to_images[prompt_idx].append(len(self.images) - 1)
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        image_paths = [item['image']]
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        
        return item
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'pairwise_acc': pairwise_acc,
        }
        return results
    
    def evaluate_scores(self, scores, indices=list(range(527))):
        scores_i2t = scores
        human_avg_scores_alignment = [np.array(self.images[idx]['human_alignment']).mean() for idx in range(len(self.images))]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in range(len(self.images))]
        alignment_correlation = self.correlation(our_scores, human_avg_scores_alignment)
        results = {
            'alignment': alignment_correlation,
        }
        return results


class GenAIBench_Image(Dataset):
    # GenAIBench with 527 prompts x 6 images (from 6 models)
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True):
        self.root_dir = os.path.join(root_dir, 'GenAI-Image-527')
        self.models = ['DALLE_3', 'SDXL_Turbo', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base']
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
        
        self.download_links = {
            model_name: f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/{model_name}.zip" for model_name in self.models
        }
        if not os.path.exists(self.root_dir):
            if download:
                import subprocess
                os.makedirs(self.root_dir, exist_ok=True)
                for model in self.models:
                    model_file_name = self.download_links[model].split('/')[-1]
                    image_zip_file = os.path.join(self.root_dir, model_file_name)
                    if not os.path.exists(image_zip_file):
                        subprocess.call(
                            ["wget", self.download_links[model], "-O", model_file_name], cwd=self.root_dir
                        )
                    model_dir = os.path.join(self.root_dir, model)
                    if not os.path.exists(model_dir):
                        subprocess.call(["unzip", "-q", model_file_name], cwd=self.root_dir)
        
        for filename in ['genai_image', 'genai_skills']:
            path = os.path.join(self.root_dir, f"{filename}.json")
            if not os.path.exists(path):
                if download:
                    import subprocess
                    download_link = f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/{filename}.json"
                    model_file_name = download_link.split('/')[-1]
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=self.root_dir
                    )
        
        self.dataset = json.load(open(os.path.join(self.root_dir, f"genai_image.json"), 'r'))
        print(f"Loaded dataset: genai_image.json")
        
        self.images = [] # list of images
        self.prompt_to_images = {}
        for model in self.models:
            for prompt_idx in self.dataset:
                if not model in self.dataset[prompt_idx]['models']:
                    continue
                
                self.images.append({
                    'prompt_idx': prompt_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': model,
                    'image': os.path.join(self.root_dir, model, f"{prompt_idx}.jpeg"),
                    'human_alignment': self.dataset[prompt_idx]['models'][model],
                })
                if prompt_idx not in self.prompt_to_images:
                    self.prompt_to_images[prompt_idx] = []
                self.prompt_to_images[prompt_idx].append(len(self.images) - 1)
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        
        image_paths = [item['image']]
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'pairwise_acc': pairwise_acc,
        }
        return results
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        human_avg_scores_alignment = [np.array(self.images[idx]['human_alignment']).mean() for idx in range(len(self.images))]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in range(len(self.images))]
        alignment_correlation = self.correlation(our_scores, human_avg_scores_alignment)
        results = {
            'alignment': alignment_correlation,
        }
        return results


class GenAIBench_Video(Dataset):
    def __init__(self,
                 image_preprocess=None,
                 root_dir="./",
                 download=True,
                 return_image_paths=True,
                 image_save_dir="genai_video_{}_extracted_images",
                 num_prompts=527, # Must be 527 (VQAScore paper) or 800 (GenAI-Bench paper)
                 num_frames=36,
                 eval_mode='avg_frames',
                 extract_videos=False):
        self.root_dir = os.path.join(root_dir, f'GenAI-Video-{num_prompts}')
        print(f"Root dir: {self.root_dir}")
        self.models = [
                        'Floor33', 'Gen2', 'Pika_v1', 'Modelscope'
                       ]
        self.eval_mode = eval_mode
        
        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
        image_save_dir = image_save_dir.format(num_prompts)
        self.image_save_dir = os.path.join(root_dir, image_save_dir)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir, exist_ok=True)
            
        self.download_links = {
            model_name: f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-{num_prompts}/resolve/main/{model_name}.zip" for model_name in self.models
        }
        if not os.path.exists(self.root_dir):
            if download:
                import subprocess
                os.makedirs(self.root_dir, exist_ok=True)
                for model in self.models:
                    model_file_name = self.download_links[model].split('/')[-1]
                    image_zip_file = os.path.join(self.root_dir, model_file_name)
                    if not os.path.exists(image_zip_file):
                        subprocess.call(
                            ["wget", self.download_links[model], "-O", model_file_name], cwd=self.root_dir
                        )
                    model_dir = os.path.join(self.root_dir, model)
                    if not os.path.exists(model_dir):
                        subprocess.call(["unzip", "-q", model_file_name], cwd=self.root_dir)
        
        for filename in ['genai_video', 'genai_skills']:
            path = os.path.join(self.root_dir, f"{filename}.json")
            if not os.path.exists(path):
                if download:
                    import subprocess
                    download_link = f"https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-{num_prompts}/resolve/main/{filename}.json"
                    model_file_name = download_link.split('/')[-1]
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=self.root_dir
                    )
                        
        self.dataset = json.load(open(os.path.join(self.root_dir, f"genai_video.json"), 'r'))
        genai_videos_file = os.path.join(self.root_dir, f"genai_videos_extracted_images.json")
        genai_prompt_to_videos_file = os.path.join(self.root_dir, f"genai_prompt_to_videos.json")
        if os.path.exists(genai_videos_file) and os.path.exists(genai_prompt_to_videos_file) and not extract_videos:
            self.videos = json.load(open(genai_videos_file, 'r'))
            self.prompt_to_videos = json.load(open(genai_prompt_to_videos_file, 'r'))
            print(f"Load from pre-extracted folder (which converted videos into sequence of images)")
            print(f"If you modify the dataset class, please re-extract the videos by setting extract_videos=True")
            return

        self.videos = [] # list of videos
        self.prompt_to_videos = {}
        for model in self.models:
            model_dir = os.path.join(self.image_save_dir, model)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            for prompt_idx in self.dataset:
                if not model in self.dataset[prompt_idx]['models']:
                    continue
                
                video_path = os.path.join(self.root_dir, model, f"{prompt_idx}.mp4")
                cap = cv2.VideoCapture(video_path)
                current_frames = []
                while True:
                    # Read a frame from the video
                    ret, frame = cap.read()
                    
                    # If the frame was not retrieved, we've reached the end of the video
                    if not ret:
                        break
                    
                    # Convert the color space from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert the frame to a PIL Image
                    img_pil = Image.fromarray(frame_rgb)
                    
                    output_path = os.path.join(model_dir, f"video_{prompt_idx}_frame_{len(current_frames):04d}.jpg")
                    img_pil.save(output_path)
                    
                    # Append the PIL Image path to the list
                    current_frames.append(output_path)
                
                if len(current_frames) == 0:
                    print(f"Skipping video: {video_path}")
                    import pdb; pdb.set_trace()
                if len(current_frames) < num_frames:
                    current_frames = current_frames + [current_frames[-1]] * (num_frames - len(current_frames))
                elif len(current_frames) > num_frames:
                    current_frames = current_frames[:num_frames]
                
                # Release the video capture object
                cap.release()
                self.videos.append({
                    'prompt_idx': prompt_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': model,
                    'video_path': video_path,
                    'num_frames': len(current_frames),
                    'frames': current_frames,
                    'human_alignment': self.dataset[prompt_idx]['models'][model],
                })
                if prompt_idx not in self.prompt_to_videos:
                    self.prompt_to_videos[prompt_idx] = []
                self.prompt_to_videos[prompt_idx].append(len(self.videos) - 1)
        print(f"Number of frames: {num_frames}")
        json.dump(self.videos, open(genai_videos_file, 'w'))
        json.dump(self.prompt_to_videos, open(genai_prompt_to_videos_file, 'w'))
        
                
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        item = self.videos[idx]
        
        image_paths = item['frames']
        if self.eval_mode == 'avg_frames':
            pass
        elif self.eval_mode == 'sample_4_frame':
            image_paths = [image_paths[0], image_paths[8], image_paths[16], image_paths[24]]
        elif self.eval_mode == 'sample_9_frame':
            image_paths = [image_paths[0], image_paths[4], image_paths[8], image_paths[12], image_paths[16], image_paths[20], image_paths[24], image_paths[28], image_paths[32]]
        else:
            raise ValueError(f"Invalid eval_mode: {self.eval_mode}")
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'pairwise_acc': pairwise_acc,
        }
        return results
    
    def evaluate_scores(self, scores):
        scores_i2t = scores
        video_indices = list(range(len(self.videos)))
        # take average of all frames
        human_alignment_scores = []
        for model in self.models:
            for prompt_idx in self.dataset:
                human_alignment_scores.append(self.dataset[prompt_idx]['models'][model])
        human_avg_scores_alignment = [np.array(human_alignment_scores[idx]).mean() for idx in video_indices]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in video_indices]
        alignment_correlation = self.correlation(our_scores, human_avg_scores_alignment)
        results = {
            'alignment': alignment_correlation,
        }
        return results
    
    
    

class GenAIBench_Ranking(Dataset):
    # GenAIBench with 800 prompts x 9 images by one generative modele
    def __init__(self,
                 gen_model='DALLE_3',
                 image_preprocess=None,
                 root_dir="./datasets", 
                 download=True,
                 return_image_paths=True):
        assert gen_model in ['DALLE_3', 'SDXL_Base'], "Invalid gen_model"
        self.gen_model = gen_model
        self.root_dir = os.path.join(root_dir, 'GenAI-Image-Ranking-800')

        self.image_preprocess = image_preprocess
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.image_preprocess is None, "Cannot return image paths and apply transforms"
            
        self.download_links = {
            gen_model: f"https://huggingface.co/datasets/zhiqiulin/GenAI-Image-Ranking-800/resolve/main/{gen_model}.zip"
        }

        model_dir = os.path.join(self.root_dir, self.gen_model)
        if not os.path.exists(model_dir):
            if download:
                import subprocess
                os.makedirs(model_dir, exist_ok=True)
                model_file_name = self.download_links[self.gen_model].split('/')[-1]
                image_zip_file = os.path.join(self.root_dir, model_file_name)
                if not os.path.exists(image_zip_file):
                    subprocess.call(
                        ["wget", self.download_links[self.gen_model], "-O", model_file_name], cwd=self.root_dir
                    )
                subprocess.call(["unzip", "-q", model_file_name], cwd=self.root_dir)
        
        filenames = ['human_rating', 'genai_skills']
        for filename in filenames:
            path = os.path.join(self.root_dir, f"{filename}.json")
            if not os.path.exists(path):
                if download:
                    import subprocess
                    download_link = f"https://huggingface.co/datasets/zhiqiulin/GenAI-Image-Ranking-800/resolve/main/{filename}.json"
                    model_file_name = download_link.split('/')[-1]
                    subprocess.call(
                        ["wget", download_link, "-O", model_file_name], cwd=self.root_dir
                    )
        
        self.dataset = json.load(open(os.path.join(self.root_dir, f"human_rating.json"), 'r'))
        print(f"Loaded dataset from: human_rating.json")
        
        self.images = [] # list of images
        self.images_to_prompt_idx = []
        
        for prompt_idx in self.dataset:
            assert prompt_idx == self.dataset[prompt_idx]["id"]
            assert self.gen_model in self.dataset[prompt_idx]['models'], f"Prompt {prompt_idx} does not have model {self.gen_model}"
            for img_idx in range(1, 10):    
                self.images.append({
                    'prompt_idx': prompt_idx,
                    'img_idx': img_idx,
                    'prompt': self.dataset[prompt_idx]['prompt'],
                    'model': self.gen_model,
                    'image': os.path.join(self.root_dir, self.gen_model, f"{'%05d'%int(prompt_idx)}_{'%02d'%img_idx}.jpeg"),
                    'human_score': np.mean(self.dataset[prompt_idx]['models'][self.gen_model][str(img_idx)]),
                })
            self.images_to_prompt_idx.append(int(prompt_idx))

                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        
        image_paths = [item['image']]
            
        if self.return_image_paths:
            image = image_paths
        else:
            image = [Image.open(image_path).convert('RGB') for image_path in image_paths]
            image = [self.image_preprocess(img) for img in image]
        
        texts = [str(item['prompt'])]
        item = {"images": image, "texts": texts}
        return item
    
    def correlation(self, our_scores, human_scores):
        pearson = calc_pearson(human_scores, our_scores)
        print(f"Pearson's Correlation (no grouping): ", pearson)
        
        kendall_b = calc_metric(human_scores, our_scores, variant="tau_b")
        print(f'Kendall Tau-B Score (no grouping): ', kendall_b)
        
        our_scores_per_prompt = np.array(our_scores).reshape(-1, 9)
        human_scores_per_prompt = np.array(human_scores).reshape(-1, 9)
        # Take the argmax and argmin of the human scores per prompt
        argmax_human_idx = np.argmax(human_scores_per_prompt, axis=1)
        argmin_human_idx = np.argmin(human_scores_per_prompt, axis=1)
        # Check if our ranking is correct
        ranking_accuracy_for_arg_max_and_min = our_scores_per_prompt[np.arange(len(our_scores_per_prompt)), argmax_human_idx] > our_scores_per_prompt[np.arange(len(our_scores_per_prompt)), argmin_human_idx]
        print(f"Ranking accuracy for human argmax and argmin: {ranking_accuracy_for_arg_max_and_min.mean()}")
        # Check all argmax human with 5.0 score
        perfect_images_indices = np.where(human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx] == 5.0)[0]
        # print the accuracy for perfect and non-perfect images
        print(f"Ranking accuracy for {len(perfect_images_indices)} pairs with 5.0 human score: {ranking_accuracy_for_arg_max_and_min[perfect_images_indices].mean()}")
        print(f"Ranking accuracy for {len(human_scores_per_prompt)-len(perfect_images_indices)} pairs without 5.0 human score: {ranking_accuracy_for_arg_max_and_min[~perfect_images_indices].mean()}")
        # pairwise_acc = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
        # print(f'Pairwise Accuracy Score (no grouping): ', pairwise_acc)
        # Check score difference between max and min
        score_diff = human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx] - human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmin_human_idx]
        # Show various statistics
        print(f"Argmax scores: mean={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx].mean():.2f}, std={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx].std():.2f}, min={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx].min():.2f}, max={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmax_human_idx].max():.2f}")
        print(f"Argmin scores: mean={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmin_human_idx].mean():.2f}, std={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmin_human_idx].std():.2f}, min={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmin_human_idx].min():.2f}, max={human_scores_per_prompt[np.arange(len(human_scores_per_prompt)), argmin_human_idx].max():.2f}")
        print(f"Score difference between max and min: mean={score_diff.mean():.2f}, std={score_diff.std():.2f}, min={score_diff.min():.2f}, max={score_diff.max():.2f}")
        print(f"Score difference between max and min: 25th percentile={np.percentile(score_diff, 25):.2f}, 50th percentile={np.percentile(score_diff, 50):.2f}, 75th percentile={np.percentile(score_diff, 75):.2f}")
        # Show for perfect images
        score_diff_perfect = score_diff[perfect_images_indices]
        print(f"Score difference between max and min for perfect images: mean={score_diff_perfect.mean():.2f}, std={score_diff_perfect.std():.2f}, min={score_diff_perfect.min():.2f}, max={score_diff_perfect.max():.2f}")
        print(f"Score difference between max and min for perfect images: 25th percentile={np.percentile(score_diff_perfect, 25):.2f}, 50th percentile={np.percentile(score_diff_perfect, 50):.2f}, 75th percentile={np.percentile(score_diff_perfect, 75):.2f}")
        # Show for non-perfect images
        score_diff_non_perfect = score_diff[~perfect_images_indices]
        print(f"Score difference between max and min for non-perfect images: mean={score_diff_non_perfect.mean():.2f}, std={score_diff_non_perfect.std():.2f}, min={score_diff_non_perfect.min():.2f}, max={score_diff_non_perfect.max():.2f}")
        print(f"Score difference between max and min for non-perfect images: 25th percentile={np.percentile(score_diff_non_perfect, 25):.2f}, 50th percentile={np.percentile(score_diff_non_perfect, 50):.2f}, 75th percentile={np.percentile(score_diff_non_perfect, 75):.2f}")
        
        for low, high in [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]:
            indices = np.where((score_diff >= low) & (score_diff < high))[0]
            print(f"Ranking accuracy for score_diff in ({low}, {high}) with {len(indices)} samples: {ranking_accuracy_for_arg_max_and_min[indices].mean():.2f}")
            print(f"Average our score difference for score_diff in ({low}, {high}): {np.mean(our_scores_per_prompt[np.arange(len(our_scores_per_prompt)), argmax_human_idx][indices] - our_scores_per_prompt[np.arange(len(our_scores_per_prompt)), argmin_human_idx][indices]):.2f}")
        # pairwise_acc_per_prompt = calc_metric(our_scores_per_prompt, human_scores_per_prompt, variant="pairwise_acc_with_tie_optimization")
        # print(f'Pairwise Accuracy Score (grouping by prompt): ', pairwise_acc_per_prompt)
        
        # Show the average our scores for images with (1.0, 3.0), (3.0, 4.0), (4.0, 5.0) human score
        for low, high in [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]:
            indices = np.where((human_scores_per_prompt >= low) & (human_scores_per_prompt < high))[0]
            print(f"Average our scores for {len(indices)} of samples where human score in ({low}, {high}): {our_scores_per_prompt.mean(axis=1)[indices].mean():.2f}")
        
        results = {
            'pearson': pearson,
            'kendall_b': kendall_b,
            'ranking_accuracy': ranking_accuracy_for_arg_max_and_min,
            # 'pairwise_acc': pairwise_acc,
            # 'pairwise_acc_per_prompt': pairwise_acc_per_prompt,
        }
        return results

    def evaluate_scores(self, scores):
        scores_i2t = scores
        human_avg_scores_alignment = [np.array(self.images[idx]['human_score']).mean() for idx in range(len(self.images))]
        our_scores = scores_i2t.mean(axis=1)
        our_scores = [float(our_scores[idx][0]) for idx in range(len(self.images))]
        results = self.correlation(our_scores, human_avg_scores_alignment)
        return results