from datasets import load_dataset


def test_trec_is_dataset():
    assert load_dataset('./sequence_classifier/custom_datasets/TRECIS.py', split="train")
