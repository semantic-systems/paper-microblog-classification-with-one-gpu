import abc
import copy
from pathlib import Path

import torch
from typing import Union, Dict, List, Optional
from omegaconf import DictConfig
from torch.nn import Module, ModuleList
from transformers import PreTrainedModel
from data_augmenters.tweet_normalizer import clean_up_tokenization, normalizeTweet, tweeteval_preprocess
from sequence_classifier.schema import InputFeature


class SequenceClassification(Module):
    def __init__(self, cfg: DictConfig, class_weights: Optional[list] = None):
        super(SequenceClassification, self).__init__()
        self.cfg = cfg
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.encoder = self.instantiate_encoder()
        self.classification_head = self.instantiate_classification_head()
        if cfg.model.load_ckpt is not None:
            checkpoint = torch.load(cfg.model.load_ckpt, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        steps:
        x = self.encoder(x)
        x = self.classification_head(x)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_encoder(self):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate_classification_head(self):
        raise NotImplementedError

    @staticmethod
    def trim_encoder_layers(encoder: PreTrainedModel, num_layers_to_keep: Union[int, str]) -> PreTrainedModel:
        if num_layers_to_keep == "full":
            return encoder
        else:
            full_layers = encoder.encoder.layer
            trimmed_layers = ModuleList()

            # Now iterate over all layers, only keeping only the relevant layers.
            for i in range(num_layers_to_keep):
                trimmed_layers.append(full_layers[i])

            # create a copy of the model, modify it with the new list, and return
            trimmed_encoder = copy.deepcopy(encoder)
            trimmed_encoder.encoder.layer = trimmed_layers
            return trimmed_encoder

    @staticmethod
    def freeze_encoder(encoder: PreTrainedModel, layers_to_freeze: Union[str, int]) -> PreTrainedModel:
        if layers_to_freeze == "none":
            return encoder
        elif layers_to_freeze == "all":
            for param in encoder.parameters():
                param.requires_grad = False
            return encoder
        elif isinstance(layers_to_freeze, int) and layers_to_freeze <= len(encoder.encoder.layer) - 1:
            modules = [encoder.embeddings, *encoder.encoder.layer[:layers_to_freeze]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            return encoder
        else:
            raise ValueError(f"Currently, only 'all', 'none' and integer (<=num_transformer_layers) "
                             f"are valid value for freeze_transformer_layer")

    def normalize(self, text: List[str]) -> List[str]:
        normalized_text = text
        if self.cfg.data.name in ["tweet_eval"]:
            normalized_text: List[str] = [tweeteval_preprocess(tweet) for tweet in text]
            # normalized_text = [clean_up_tokenization(tweet) for tweet in normalized_text]
        return normalized_text

    def preprocess(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        # remove max_length arg to use the model's original value for that

    def save_model(self, path: Path, index_label_map: Dict):
        torch.save({
            'config': self.cfg,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'index_label_map': index_label_map
        }, path)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def inference(self, input_feature: InputFeature, mode: str):
        if self.cfg.augmenter.name == "dropout" and mode == "train":
            num_instances = int(input_feature.input_ids.shape[0] / (self.cfg.augmenter.num_samples + 1))
            output = self.encoder(input_ids=input_feature.input_ids[:num_instances],
                                  attention_mask=input_feature.attention_mask[:num_instances]).pooler_output
            original_dropout_rate = self.encoder.encoder.layer[0].output.dropout.p
            for i, dropout_rate in enumerate(self.cfg.augmenter.dropout):
                for n in range(len(self.encoder.encoder.layer)):
                    self.encoder.encoder.layer[n].output.dropout.p = dropout_rate
                augmented_output = self.encoder(input_ids=input_feature.input_ids[(i+1)*num_instances:(i+2)*num_instances],
                                                attention_mask=input_feature.attention_mask[
                                                                 (i+1)*num_instances:(i+2)*num_instances]).pooler_output
                output = torch.cat((output, augmented_output), dim=0)
            for n in range(len(self.encoder.encoder.layer)):
                self.encoder.encoder.layer[n].output.dropout.p = original_dropout_rate
        else:
            output = self.encoder(input_ids=input_feature.input_ids,
                                  attention_mask=input_feature.attention_mask).pooler_output
        return output

    def get_multiview_batch(self, features, labels):
        # no augmentation
        if self.cfg.augmenter.name is None and self.cfg.model.contrastive.contrastive_loss_ratio > 0:
            contrastive_features = features[:, None, :]
            contrastive_labels = labels
        elif self.cfg.augmenter.name is not None:
            multiview_shape = (
                int(features.shape[0] / (self.cfg.augmenter.num_samples + 1)),
                self.cfg.augmenter.num_samples + 1,
                features.shape[-1]
            )
            contrastive_features = features.reshape(multiview_shape)
            contrastive_labels = labels[:int(features.shape[0]/(self.cfg.augmenter.num_samples+1))]
        else:
            raise ValueError
        return contrastive_features, contrastive_labels
