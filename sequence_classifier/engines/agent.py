from typing import Dict, Union, Type, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from dataclasses import dataclass
from copy import deepcopy
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_augmenters.data_augmenter import FSMTBackTranslationAugmenter, RandomAugmenter, DropoutAugmenter
from sequence_classifier.models import SingleLabelSequenceClassification, \
    SingleLabelContrastiveSequenceClassification
from sequence_classifier.schema import InputFeature, SingleLabelClassificationForwardOutput, \
    AgentPolicyOutput, TSNEFeature

PolicyClasses = Union[SingleLabelSequenceClassification,
                      SingleLabelContrastiveSequenceClassification
                      ]

ClassificationForwardOutput = Union[SingleLabelClassificationForwardOutput]


@dataclass
class AgentState(object):
    done: bool = False
    early_stopping: bool = False


class Agent(object):
    def __init__(self, config: DictConfig, device: torch.device, class_weights: Optional[list] = None):
        self.config = config
        self.state = AgentState()
        self.device = device
        self.class_weights = class_weights

    def act(self, **kwargs) -> AgentPolicyOutput:
        raise NotImplementedError

    def instantiate_policy(self):
        raise NotImplementedError

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError


class BatchLearningAgent(Agent):
    def __init__(self, config: DictConfig, device: torch.device, class_weights: Optional[list] = None):
        super(BatchLearningAgent, self).__init__(config, device, class_weights)
        self.policy = self.instantiate_policy()
        self.Augmenter = self.instantiate_augmenter(config.augmenter.name) if self.is_contrastive else None

    @property
    def policy_class(self) -> Type[PolicyClasses]:
        if self.is_contrastive:
            if self.config.model.type == "single-label":
                return SingleLabelContrastiveSequenceClassification
            elif self.config.model.type == "multi-label":
                return MultiLabelContrastiveSequenceClassification

        else:
            if self.config.model.type == "single-label":
                return SingleLabelSequenceClassification
            elif self.config.model.type == "multi-label":
                return MultiLabelSequenceClassification

    @property
    def is_contrastive(self) -> bool:
        return self.config.model.contrastive.contrastive_loss_ratio > 0

    def instantiate_policy(self):
        return self.policy_class(self.config, class_weights=self.class_weights)

    def log_something(self):
        raise NotImplementedError

    def update_state(self, state: Dict):
        self.state = AgentState(**state)

    def travel_back(self):
        raise NotImplementedError

    def act(self, data_loader: DataLoader, mode: str) -> AgentPolicyOutput:
        # action per time step - here it will be a batch
        y_predict, y_true = [], []
        loss = 0
        ce_loss = 0
        contrastive_loss = 0
        tsne_features: Dict = {"final_hidden_states": [], "encoded_features": [], "labels": []}
        test_input: List = []

        for i, batch in enumerate(tqdm(data_loader)):
            backward = False
            batch["text"] = self.policy.normalize(batch["text"])
            if mode == "train":
                self.policy.optimizer.zero_grad()
                if self.Augmenter is not None:
                    batch = self.augment(batch, self.config.augmenter.num_samples)
            if mode == "test":
                test_input.extend(batch["text"])
            if isinstance(batch["label"], list):
                batch["label"] = torch.stack(batch["label"]).T.float()
            labels: tensor = batch["label"].to(self.device)
            y_true.extend(labels)
            batch = self.policy.preprocess(batch)
            input_ids: tensor = batch["input_ids"].to(self.device)
            attention_masks: tensor = batch["attention_mask"].to(self.device)
            # convert labels to None if in testing mode.
            # labels = None if mode == "test" else labels
            input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            if ((i + 1) % self.config.data.gradient_accu_step == 0) or (i + 1 == len(data_loader)):
                backward = True
            outputs: ClassificationForwardOutput = self.policy(input_feature, mode=mode, backward=backward)
            prediction = self.get_prediction(outputs)
            y_predict.extend(prediction)
            if mode == "train":
                loss = (loss + outputs.loss)/(i+1)
                if self.is_contrastive:
                    ce_loss = (ce_loss + outputs.cross_entropy_loss) / (i + 1)
                    contrastive_loss = (contrastive_loss + outputs.contrastive_loss) / (i + 1)
            if mode in ["validation", "test"]:
                loss = (loss + outputs.loss)/(i+1)
            if "tsne" in self.config.visualizer and mode in ["validation", "test"]:
                tsne_features["encoded_features"].extend(outputs.encoded_features.tolist())
                tsne_features["final_hidden_states"].extend(outputs.prediction_logits.tolist())
        return AgentPolicyOutput(**{"y_predict": y_predict, "y_true": y_true, "loss": loss,
                                    "tsne_feature": TSNEFeature(**tsne_features), "test_input_text": test_input,
                                    "cross_entropy_loss": ce_loss, "contrastive_loss": contrastive_loss})

    def get_prediction(self, outputs: ClassificationForwardOutput):
        if self.config.model.type == "single-label":
            prediction = outputs.prediction_logits.argmax(1)
        elif self.config.model.type == "multi-label":
            prediction = torch.round(outputs.prediction_logits)
        else:
            raise NotImplementedError
        return prediction

    @staticmethod
    def instantiate_augmenter(name: str, **kwargs):
        if name is None:
            augmenter = None
        elif name == "dropout":
            augmenter = DropoutAugmenter()
        elif name == "random":
            augmenter = RandomAugmenter()
        elif name == "back_translation":
            augmenter = FSMTBackTranslationAugmenter(device=kwargs.get("device"),
                                                     from_model=kwargs.get("from_model"),
                                                     to_model=kwargs.get("to_model"))
        else:
            raise NotImplementedError
        return augmenter

    def augment(self, batch: Dict, num_augmented_samples: int) -> Dict:
        augmented_text = self.Augmenter.augment(batch["text"], num_return_sequences=num_augmented_samples)
        augmented_batch = deepcopy(batch)
        augmented_batch["text"].extend(augmented_text)
        # label need to repeat n times + the original copy
        # if isinstance(batch["label"], list):
        if self.config.model.type == "multi-label":
            batch["label"] = torch.stack(batch["label"]).T.float()
            augmented_batch["label"] = batch["label"].repeat(num_augmented_samples + 1, 1)
        else:
            augmented_batch["label"] = batch["label"].repeat(num_augmented_samples+1)
        return augmented_batch
