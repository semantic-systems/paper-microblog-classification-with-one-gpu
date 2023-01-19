import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from sequence_classifier.data_generators.samplers import FixedSizeCategoricalSampler


def test_episodic_batch_sampler():
    n_way = 5
    k_shot = 3
    iterations = 20
    n_query = 5
    dataset = load_dataset('banking77', split="train").train_test_split(test_size=0.4)["test"]
    sampler = FixedSizeCategoricalSampler(data_source=dataset,
                                          n_way=n_way,
                                          k_shot=k_shot,
                                          iterations=iterations,
                                          n_query=n_query)
    data_loader = DataLoader(dataset, sampler=sampler)
    # remove header
    sample = next(iter(data_loader))
    assert len(sample["text"]) == len(sample["label"])
    assert (len(data_loader) == iterations)
    for batch in data_loader:
        assert (len(batch["text"]) == len(batch["label"]) == n_way * k_shot + n_way * n_query)
        classes, counts = np.unique(batch["label"], return_counts=True)
        assert len(classes) == n_way
        assert all([count == k_shot + n_way for count in counts])
