from .utils import pre_text
from os.path import basename
import numpy as np

from .base_dataset import ImageVideoBaseDataset
from .utils import load_anno
from .video_utils import VIDEO_READER_FUNCS
import logging

logger = logging.getLogger(__name__)


class ImgTxtRetTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False):
        super(ImgTxtRetTrainDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # each caption has multiple image as ground_truth, e.g., ssv2
        self.has_multi_vision_gt = has_multi_vision_gt
        self.match_ids = {}

        n = 0
        for ann in self.anno_list:
            key = ann["caption"] if has_multi_vision_gt else basename(ann["image"])
            if key not in self.match_ids:
                self.match_ids[key] = n
                n += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        try:
            ann = self.anno_list[index]
            image, index = self.load_and_transform_media_data(index, ann["image"])
            caption = pre_text(ann["caption"])
            key = ann["caption"] if self.has_multi_vision_gt else basename(ann["image"])
            if 'neg_caption' in ann:
                neg_caption = ann['neg_caption']
                return image, (caption, neg_caption), self.match_ids[key]
            else:
                return image, caption, self.match_ids[key]
        except Exception as e:
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class VidTxtRetTrainDataset(ImgTxtRetTrainDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=3,
            is_paragraph_retrieval=False, has_multi_vision_gt=False,
            trimmed30=False
    ):
        super(VidTxtRetTrainDataset, self).__init__(ann_file, transform, has_multi_vision_gt)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval
        self.trimmed30 = trimmed30
        if trimmed30:
            logger.info("Trimming the video, only use the first 30s")

        if is_paragraph_retrieval:
            self.anno_list = preprocess_para_retrieval_data(self.anno_list)


class ImgTxtRetEvalDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, has_multi_vision_gt=False):
        super(ImgTxtRetEvalDataset, self).__init__()
        self.raw_anno_list = load_anno(ann_file)
        self.transform = transform
        self.has_multi_vision_gt = has_multi_vision_gt  # each caption has multiple image as ground_truth

        self.text = None
        self.image = None
        self.txt2img = None
        self.img2txt = None
        self.build_data()

    def build_data(self):
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        if self.has_multi_vision_gt:
            self.build_data_multi_img_gt()
        else:
            self.build_data_multi_txt_gt()
        self.anno_list = [dict(image=e) for e in self.image]

    def build_data_multi_img_gt(self):
        """each text may have multiple ground_truth image, e.g., ssv2"""
        img_id = 0
        for txt_id, ann in enumerate(self.raw_anno_list):
            self.text.append(pre_text(ann["caption"]))
            self.txt2img[txt_id] = []
            _images = ann["image"] \
                if isinstance(ann["image"], list) else [ann["image"], ]
            for i, image in enumerate(_images):
                self.image.append(image)
                self.txt2img[txt_id].append(img_id)
                self.img2txt[img_id] = txt_id
                img_id += 1

    def build_data_multi_txt_gt(self):
        """each image may have multiple ground_truth text， e.g., COCO and Flickr30K"""
        txt_id = 0
        for img_id, ann in enumerate(self.raw_anno_list):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            _captions = ann["caption"] \
                if isinstance(ann["caption"], list) else [ann["caption"], ]
            for i, caption in enumerate(_captions):
                self.text.append(pre_text(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])
        return image, index


class VidTxtRetEvalDataset(ImgTxtRetEvalDataset):
    media_type = "video"

    def __init__(
            self, ann_file, transform, num_frames=4,
            video_reader_type="decord", sample_type="rand", num_tries=1,
            is_paragraph_retrieval=False, has_multi_vision_gt=False,
            trimmed30=False
    ):
        super(VidTxtRetEvalDataset, self).__init__(ann_file, transform, has_multi_vision_gt)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.is_paragraph_retrieval = is_paragraph_retrieval
        self.trimmed30 = trimmed30
        if trimmed30:
            logger.info("Trimming the video, only use the first 30s")

        if is_paragraph_retrieval:
            self.anno_list = preprocess_para_retrieval_data(self.raw_anno_list)
        self.build_data()


def preprocess_para_retrieval_data(anno_list):
    processed_anno_list = []
    for d in anno_list:
        d["caption"] = " ".join(d.pop("caption"))
        processed_anno_list.append(d)
    return processed_anno_list


class VidTxtRetMCEvalDataset(ImageVideoBaseDataset):
    """For MSRVTT-MC test task"""
    media_type = "video"

    def __init__(self, ann_file, transform, num_frames=4,
                 video_reader_type="decord", sample_type="rand", num_tries=1):
        super(VidTxtRetMCEvalDataset, self).__init__()
        self.anno_list = load_anno(ann_file)
        self.transform = transform
        # video args
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        ann = self.anno_list[index]
        image, index = self.load_and_transform_media_data(index, ann["image"])
        caption = [pre_text(e) for e in ann["caption"]]  # len=5
        answer = ann["answer"]
        return image, caption, answer, ann
