from typing import Union, Tuple, List, Optional
from tqdm import tqdm
from transformers import FSMTModel, FSMTTokenizer, FSMTForConditionalGeneration, PreTrainedModel
from data_augmenters.tweet_normalizer import normalizeTweet, clean_up_tokenization
from utils import instantiate_config
import nlpaug.augmenter.word as naw


class DataAugmenter(object):

    def augment(self, **kwargs) -> Union[str, List]:
        raise NotImplementedError


class TweetsAugmenter(object):

    def augment(self, **kwargs) -> List:
        raise NotImplementedError

    @staticmethod
    def normalize_tweets(tweets: Union[str, List]) -> List:
         return list(map(normalizeTweet, tweets))

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        # rules for post-processing the decoded sequence
        return clean_up_tokenization(out_string)


class FSMTBackTranslationAugmenter(TweetsAugmenter):
    def __init__(self,
                 device: str = "cpu",
                 from_model: str = "facebook/wmt19-en-de",
                 to_model: str = "facebook/wmt19-de-en",
                 ):
        super(FSMTBackTranslationAugmenter, self).__init__()
        self.from_model, self.from_tokenizer = self.instantiate_machine_translation_model(from_model, device=device)
        self.to_model, self.to_tokenizer = self.instantiate_machine_translation_model(to_model, device=device)

    def augment(self, data: Union[str, list], num_return_sequences: int = 1, temperature: float = 1.0) -> List:
        normalized_data = self.normalize_tweets(data)
        src_input_ids = self.from_tokenizer(normalized_data, padding=True, truncation=True, return_tensors="pt").input_ids
        from_model_outputs = self.from_model.generate(src_input_ids, num_beams=5, num_return_sequences=num_return_sequences, temperature=temperature)
        target_seq = self.from_tokenizer.batch_decode(from_model_outputs, skip_special_tokens=True)

        tgt_input_ids = self.to_tokenizer(target_seq, padding=True, truncation=True, return_tensors="pt").input_ids
        to_model_outputs = self.to_model.generate(tgt_input_ids, num_beams=5, num_return_sequences=1, temperature=temperature)
        back_translated_seq = self.to_tokenizer.batch_decode(to_model_outputs, skip_special_tokens=True)
        return list(map(self.clean_up_tokenization, back_translated_seq))

    @staticmethod
    def instantiate_machine_translation_model(from_model: str, device: str = "cpu") -> Tuple[FSMTModel, FSMTTokenizer]:
        model = FSMTForConditionalGeneration.from_pretrained(from_model).to(device)
        tokenizer = FSMTTokenizer.from_pretrained(from_model)
        return model, tokenizer


class RandomAugmenter(TweetsAugmenter):
    def __init__(self):
        self.swap_augmenter = self.instantiate_augmenter("swap", 0.1, aug_max=2)
        self.delete_augmenter = self.instantiate_augmenter("delete", 0.1, aug_max=2)
        self.substitute_augmenter = self.instantiate_augmenter("substitute", 0.1, aug_max=2)

    def augment(self, data: Union[str, list], num_return_sequences: int = 1, num_thread: int = 1) -> List:
        normalized_data = self.normalize_tweets(data)
        if isinstance(data, str) or (isinstance(data, list) and len(data) == 1):
            augmented_text = self.swap_augmenter.augment(normalized_data[0], n=num_return_sequences, num_thread=num_thread)
            augmented_text = self.delete_augmenter.augment(augmented_text, n=num_return_sequences, num_thread=num_thread)
            augmented_text = self.swap_augmenter.augment(augmented_text, n=num_return_sequences, num_thread=num_thread)
            augmented_text = list(augmented_text)
        else:
            augmented_text = self.swap_augmenter.augment(normalized_data)
            augmented_text = self.delete_augmenter.augment(augmented_text)
            augmented_text = self.substitute_augmenter.augment(augmented_text)
        return list(map(self.clean_up_tokenization, augmented_text))

    @staticmethod
    def instantiate_augmenter(action: str, aug_p: float, aug_max: Optional[int] = None):
        """actions: swap, delete, substitute or crop"""
        return naw.RandomWordAug(action=action, aug_p=aug_p, aug_max=aug_max)


class DropoutAugmenter(TweetsAugmenter):
    def augment(self, data: Union[str, list], num_return_sequences: int = 1) -> List:
        normalized_data = self.normalize_tweets(data)
        augmented_text = []
        for n in range(num_return_sequences):
            augmented_text += normalized_data
        return list(map(self.clean_up_tokenization, augmented_text))


if __name__ == "__main__":
    from sequence_classifier.engines.environment import StaticEnvironment
    from sequence_classifier.validate import ConfigValidator

    config_path = "./sequence_classifier/configs/test/emotion_scl.yaml"
    cfg = instantiate_config(config_path)
    validator = ConfigValidator(cfg)
    config = validator()
    env = StaticEnvironment(cfg)
    data_loader = env.load_environment("train", "batch_training")
    augmenter = RandomAugmenter()
    for i, batch in enumerate(tqdm(data_loader)):
        normalized_tweets = list(map(normalizeTweet, batch['text']))
        augmented_text = augmenter.augment(normalized_tweets)
        print(f"original text: "
              f"{batch['text']}\n"
              f"normalized: "
              f"{normalized_tweets}\n"
              f"augmented_text: "
              f"{augmented_text}")
        break