import copy
from datetime import datetime

import mlflow
from omegaconf import DictConfig
from utils import log_params_from_omegaconf_dict


def fill_config_with_num_classes(cfg_layer: DictConfig, num_classes: int) -> DictConfig:
    updated_config = copy.deepcopy(cfg_layer)
    for n, (key, value) in enumerate(list(cfg_layer.items())):
        if n == len(list(cfg_layer.values())) - 1:
            updated_config[key]["n_out"] = num_classes
    return updated_config


def set_run_training(func):
    def run(*args):
        a = args[0]
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment(a.config.name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(a.config)
            func(*args)
    return run


def set_run_testing(func):
    def run(*args):
        with mlflow.start_run():
            func(*args)
    return run


def log_metrics(func):
    def run(*args, **kwargs):
        result = func(*args, **kwargs)
        # log metric
        mlflow.log_metric("loss", result.loss, step=1)
        if "train" in result.path_to_plot.split("/")[-1]:
            mode = "train"
        elif "validation" in result.path_to_plot.split("/")[-1]:
            mode = "validation"
        elif "test" in result.path_to_plot.split("/")[-1]:
            mode = "test"
        else:
            raise ValueError
        mlflow.log_metric(f"{mode}_acc", result.acc, step=1)
        mlflow.log_metric(f"{mode}_f1_micro", result.f1_micro, step=1)
        mlflow.log_metric(f"{mode}_f1_macro", result.f1_macro, step=1)
        mlflow.log_metric(f"{mode}_precision_macro", result.precision_macro, step=1)
        mlflow.log_metric(f"{mode}_recall_macro", result.recall_macro, step=1)
        for key, value in result.f1_per_class.items():
            mlflow.log_metric(f"{mode}_f1_for_class_{key}", value, step=1)
        try:
            mlflow.log_artifact(result.path_to_plot)
        except FileNotFoundError:
            pass
        return result
    return run


def get_data_time() -> str:
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string
