import io
import pickle
import random
import logging
from typing import List, Dict

import mlflow
import numpy as np
import torch
from hydra import initialize, compose
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def instantiate_config(config_path: str):
    config_dir = config_path.rsplit("/", 1)[0]
    config_file = config_path.rsplit("/", 1)[-1]
    with initialize(config_path=config_dir, job_name=config_path):
        cfg = compose(config_name=config_file)
    return cfg


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
    elif isinstance(element, int) or isinstance(element, str):
        mlflow.log_param(parent_name, element)
    else:
        logger.warning(f"Configuration field {parent_name} with value {element} not logged in mlflow.")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def get_hyperparameters(nested_dict) -> Dict:
    hyperparameters = {}
    contrastive_loss_ratio = nested_dict.hyperparameters.model.contrastive.contrastive_loss_ratio
    temperature = nested_dict.hyperparameters.model.contrastive.temperature
    contrastive_loss_ratio_list = list(np.arange(contrastive_loss_ratio[0], contrastive_loss_ratio[1], contrastive_loss_ratio[2]))
    contrastive_loss_ratio_list.append(contrastive_loss_ratio[1])
    temperature_list = list(np.arange(temperature[0], temperature[1], temperature[2]))
    temperature_list.append(temperature[1])
    hyperparameters["contrastive_loss_ratio"] = [round(float(c), 3) for c in contrastive_loss_ratio_list]
    hyperparameters["temperature"] = [round(float(c), 3) for c in temperature_list]
    return hyperparameters


def config_generator(config, hyperparameters: Dict):
    for ratio in hyperparameters["contrastive_loss_ratio"]:
        for temp in hyperparameters["temperature"]:
            config.model.contrastive.contrastive_loss_ratio = ratio
            config.model.contrastive.temperature = temp
            config.model.output_path = f"./outputs/tweeteval/experiments/{str(round(ratio, 2))}/{str(round(temp, 2))}/"
            yield config
