# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class DatasetConf:
    name: str = ""
    annotation: str = ""
    root_dir: Optional[str] = None


def read_yaml_to_configs(yaml_file_path: str) -> dict:
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

    dataset_config = {}
    for dataset_name, dataset_info in yaml_data.items():
        dataset_config[dataset_name] = DatasetConf(
            name=dataset_name,
            annotation=dataset_info["annotation"],
            root_dir=dataset_info.get("root_dir"),
        )

    return dataset_config


# Determine the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the datasets.yaml file
yaml_file_path = os.path.join(current_directory, "configs", "datasets.yaml")
# Read the YAML file
dataset_config = read_yaml_to_configs(yaml_file_path)
