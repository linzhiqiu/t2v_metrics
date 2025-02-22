import torch
import os

import os.path as osp
from copy import deepcopy
from importlib import import_module

from ...video_utils import get_video_details, load_frames_from_video
import numpy as np

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .multi_modality.tasks.shared_utils import setup_model

from .multi_modality.utils.config_utils import setup_main

def download_internvideo2(model_name, cache_dir):
    repo_id = f"zhiqiulin/{model_name}"
    filename = f"{model_name}.pth"
    model_path = os.path.join(cache_dir, model_name) + ".pth"
    if not os.path.exists(model_path):
        hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        os.system(f"wget -O {model_path} {hf_url}")
        # if download fails raise an error
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {hf_url} because the download failed")

    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
    config_file = os.path.join(current_dir, "configs.py")  # Construct the new file path
    config = setup_main(config_file=config_file, pretrained_path=model_path)
    
    
    is_pretrain = config.mode == "pt"
    device = torch.device(config.device)

    # train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
    #     config, mode=config.mode
    # )
    # num_steps_per_epoch = sum(len(d) for d in train_loaders)
    # config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    # config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    # cudnn.benchmark = len(train_media_types) == 1

    find_unused_parameters = config.model.get('find_unused_parameters', False)

    model_cls = eval(config.model.get('model_cls'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        add_decoder=False,
        pretrain=is_pretrain,
        find_unused_parameters=find_unused_parameters,
    )

    for epoch in range(start_epoch, config.scheduler.epochs):
        eval_res = {}
        for test_name, test_loader in test_name2loaders.items():
            if test_name not in config.test_types:
                logger.info(
                    f"Skip eval {test_name} split. All test_types {config.test_types}"
                )
                continue
            res = evaluation_wrapper(
                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
            )
            eval_res.update(res)

        if is_main_process():
            cur_recall = eval_res[best_key[0]][best_key[1]]

            eval_res = pd.DataFrame(eval_res)
            # logger.info(f"Epoch {epoch}")
            # logger.info(f"\n{eval_res.transpose().to_string(max_cols=30)}")

            eval_res.to_json(join(config.output_dir, "eval_res_latest.json"))

            state_dict = model_without_ddp.state_dict()

            for k in config.get("no_save_params_prefix", []):
                kk = [x for x in state_dict.keys() if x.startswith(k)]
                # logger.info(f"Not saving {len(kk)} params with prefix {k}")
                for kkk in kk:
                    state_dict.pop(kkk)

            if not config.evaluate and cur_recall > best:
                if not (hasattr(config, "deepspeed") and config.deepspeed.enable):
                    try:
                        with io.BytesIO() as buffer:
                            torch.save(save_obj, buffer)
                            client_ckpt.put(f"{ceph_ckpt_path}/ckpt_best_{best_ckpt_id}.pth", buffer.getvalue())
                            logger.info(f"Save to ceph ({f'{ceph_ckpt_path}/ckpt_best_{best_ckpt_id}.pth'})!!!")
                    except Exception as e:
                        print(e)
                        torch.save(save_obj, join(config.output_dir, f"ckpt_best_{best_ckpt_id}.pth"))
                        logger.warn(f"Ceph is not working, save to local ({join(config.output_dir, f'ckpt_best_{best_ckpt_id}.pth')})!!!")
                else:
                    
                    now_ckpt_path = f"{config.output_dir}/{tag}/mp_rank_00_model_states.pt"
                    best_ckpt_path = f"{config.output_dir}/best_mp_rank_00_model_states.pt"

                    if os.path.exists(now_ckpt_path):
                        shutil.copy(now_ckpt_path, best_ckpt_path)
                        logger.info(f"Copy {now_ckpt_path} to {best_ckpt_path}!!!")
                    else:
                        logger.warn(f"Can't find {now_ckpt_path}, there's some wrong!!!")

                eval_file = "eval_res_best.json"
                eval_res.to_json(join(config.output_dir, eval_file))
                best = cur_recall
                best_epoch = epoch


        if config.evaluate:
            break
        
        start_step = global_step

    return model_path, config
