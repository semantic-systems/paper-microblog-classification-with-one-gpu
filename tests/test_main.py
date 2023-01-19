import glob
from pathlib import Path
from omegaconf import DictConfig
from utils import instantiate_config


def test_load_and_iterate_configs_in_a_folder(config_dir_path):
    config_names = glob.glob(str(Path(config_dir_path).absolute()) + "*.yaml")
    for config_name in config_names:
        cfg = instantiate_config(config_name, job_name="not important")
        assert isinstance(cfg, DictConfig)

