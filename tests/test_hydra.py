from omegaconf import DictConfig


def test_nested_dict_in_hydra_is_iterable(hydra_config):
    cfg = hydra_config
    assert isinstance(cfg, DictConfig)
    layers = [(layer.n_in, layer.n_out) for layer in cfg.model.layers.values()]
    assert layers == [(768, 128), (128, 20)]
