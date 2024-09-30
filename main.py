import glob
import logging
import os
import warnings
from pathlib import Path
from time import sleep

import torch
from omegaconf import DictConfig, ListConfig

from sequence_classifier.engines.trainer import BatchLearningTrainer
from sequence_classifier.parsers.parser import parse
from utils import instantiate_config, get_hyperparameters, config_generator


def get_trainer(config_name: str):
    return BatchLearningTrainer


def fetch_files_from_dir(root: str, file_extension: str = ".yaml"):
    return [fname for fname in walk_through_files(root, file_extension)]


def walk_through_files(path, file_extension='.csv'):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dirpath, filename)


def run(cfg: DictConfig):
    if "hyperparameters" in cfg:
        hyperparameters = get_hyperparameters(cfg)
        configs = config_generator(cfg, hyperparameters)
        for config in configs:
            trainer = trainer_class(config)
            trainer.run()
            torch.cuda.empty_cache()
            sleep(10)

    else:
        if isinstance(cfg.seed, int):
            trainer = trainer_class(cfg)
            trainer.run()
            torch.cuda.empty_cache()
            sleep(10)
        elif isinstance(cfg.seed, ListConfig):
            for seed in cfg.seed:
                cfg.seed = seed
                trainer = trainer_class(cfg)
                trainer.run()
                torch.cuda.empty_cache()
                sleep(10)
        else:
            raise ValueError(f"Seed must be of type int or list of int.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    args = parse()
    if Path(args.config).is_file():
        cfg = instantiate_config(args.config)
        trainer_class = get_trainer(args.config)
        run(cfg)
    elif Path(args.config).is_dir():
        configs = fetch_files_from_dir(args.config, ".yaml")
        logger.warning(f"List of experiments to be run with the following configs: \n"
                       f"{configs}")
        for config in configs:
            cfg = instantiate_config(config)
            trainer_class = get_trainer(config)
            run(cfg)
    else:
        raise FileNotFoundError(f"{args.config} does not exist.")

