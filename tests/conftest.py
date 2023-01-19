import pytest
from hydra import initialize, compose
from sequence_classifier.models.SingleLabelSequenceClassification import SingleLabelSequenceClassification


@pytest.fixture()
def hydra_config():
    with initialize(config_path="./configs", job_name="test"):
        # config is relative to a module
        cfg = compose(config_name="test_config_1.yaml")
    return cfg


@pytest.fixture()
def model_instance(hydra_config):
    model = SingleLabelSequenceClassification(hydra_config)
    return model


@pytest.fixture()
def config_dir_path():
    return "./configs/test_cohort_1/"
