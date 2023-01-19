import json
from collections import Counter
from pathlib import Path
from typing import Dict, Union, List, Optional
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import emoji
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, asdict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, \
    precision_score, accuracy_score, silhouette_score
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Sampler

from sequence_classifier.data_generators import DataGenerator, DataGeneratorSubSample
from sequence_classifier.data_generators.samplers import FixedSizeCategoricalSampler
from sequence_classifier.evaluators.Visualizer import TSNEVisualizer
from sequence_classifier.helper import log_metrics
from sequence_classifier.schema import ClassificationResult, TSNEFeature, FeatureToVisualize


@dataclass
class EnvironmentState(object):
    config: Dict


class Environment(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.state = EnvironmentState({"config": self.config})
        self.environment = self.instantiate_environment()

    @abstractmethod
    def instantiate_environment(self) -> DataGenerator:
        # in single sequence classification task, the environment is where the dataloader is located
        # in RL/multiple sequence classification task, the environment is where the simulator is located
        # an environment in a classification task is a simplied version of the RL environment,
        # a static environment is expecting to receive an action from the agent(s) at each time step.
        # a dynamic environment is expecting to receive an action after t time step.
        raise NotImplementedError

    @abstractmethod
    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, state: EnvironmentState):
        raise NotImplementedError

    def return_state_as_dict(self):
        return asdict(self.state)

    def get_path_to_plot(self, name: str) -> str:
        return str(Path(self.path_to_output, "plots", f"{name}").absolute())

    @property
    def path_to_output(self):
        return Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}").absolute()

    def dump_config(self):
        with open(Path(self.path_to_output, f"config.yaml"), "w") as final:
            OmegaConf.save(config=self.config, f=final)

    def dump_result(self, result: List, mode: str):
        with open(Path(self.path_to_output, f"{mode}_result.json"), "w") as final:
            json.dump(result, final, indent=2)

    def dump_csv(self, data: Dict):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(Path(self.path_to_output, 'test_result.csv'), index=True, header=True)


class StaticEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(StaticEnvironment, self).__init__(config)
        self.label_index_map = self.environment.label_index_map
        self.index_label_map = {str(value): key for key, value in self.label_index_map.items()}
        self.tsne_visualizer = self.instantiate_tsne_visualizer()
        labels = [row["label"] for row in self.environment.training_dataset]
        self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels),
                                                               y=labels)

    def instantiate_environment(self) -> DataGenerator:
        if "subset" in self.config.data:
            return DataGeneratorSubSample(self.config)
        else:
            return DataGenerator(self.config)

    def instantiate_tsne_visualizer(self) -> Union[TSNEVisualizer, None]:
        if "tsne" in self.config.visualizer:
            return TSNEVisualizer()
        else:
            return None

    def instantiate_sampler(self, mode: str, training_type: str) -> Union[Sampler, None]:
        data_source = self.environment.training_dataset if mode == "train" else self.environment.testing_dataset
        if training_type == "episodic_training":
            return FixedSizeCategoricalSampler(data_source=data_source,
                                               n_way=self.config.episode.n_way,
                                               k_shot=self.config.episode.k_shot,
                                               iterations=self.config.episode.iteration,
                                               n_query=self.config.episode.n_query,
                                               replacement=self.config.episode.replacement)
        elif training_type == "batch_training":
            return None

    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        # mode -> "train" or "test"
        sampler = self.instantiate_sampler(mode, training_type)
        return self.environment(mode=mode, sampler=sampler)

    def update_state(self, state: Dict):
        raise NotImplementedError

    @property
    def num_labels(self):
        return self.environment.num_labels

    @property
    def labels_list(self) -> List[str]:
        labels = [*self.environment.label_index_map]
        # check if labels are emoji
        if all(emoji.is_emoji(label) for label in labels):
            normalized_labels = [emoji.demojize(label)[1:-1] for label in labels]
        else:
            normalized_labels = labels
        return normalized_labels

    def visualize_embedding(self, tsne_feature: TSNEFeature, epoch: Optional[int] = None):
        encoded_features: FeatureToVisualize = FeatureToVisualize(**{"feature": tsne_feature.encoded_features,
                                                                     "labels": tsne_feature.labels})
        final_hidden_states: FeatureToVisualize = FeatureToVisualize(**{"feature": tsne_feature.final_hidden_states,
                                                                        "labels": tsne_feature.labels})
        if epoch is not None:
            self.tsne_visualizer.visualize(data=encoded_features,
                                           path_to_save=self.get_path_to_plot(f'tsne_validation_encoder_output_epoch_{epoch}.png'))
            self.tsne_visualizer.visualize(data=final_hidden_states,
                                           path_to_save=self.get_path_to_plot(f'tsne_validation_head_output_epoch_{epoch}.png'))

        else:
            self.tsne_visualizer.visualize(data=encoded_features,
                                           path_to_save=self.get_path_to_plot(f'tsne_test_encoder_output.png'))
            self.tsne_visualizer.visualize(data=final_hidden_states,
                                           path_to_save=self.get_path_to_plot(f'tsne_test_head_output.png'))

    def clustering_score(self, features, labels) -> float:
        features = torch.tensor(features)
        features = features.cpu().detach().numpy()
        labels = torch.tensor(labels)
        labels = labels.cpu().detach().numpy()
        if self.config.model.type == "single-label":
            try:
                silhouette = silhouette_score(features, labels)
            except ValueError:
                silhouette = 0
        else:
            silhouette = 0
        return float(silhouette)

    @log_metrics
    def evaluate(self,
                 y_predict: List,
                 y_true: List,
                 loss: int,
                 mode: str,
                 num_epoch: Optional[int] = None) -> ClassificationResult:
        try:
            y_predict = torch.tensor(y_predict)
            y_true = torch.tensor(y_true)
        except ValueError:
            y_predict = torch.stack(y_predict)
            y_true = torch.stack(y_true)
            y_predict = y_predict.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
        acc = accuracy_score(y_true=y_true, y_pred=y_predict)
        f1_micro = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='micro')
        f1_macro = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='macro')
        recall_macro = recall_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='macro')
        precision_macro = precision_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average='macro')
        f1_per_class = f1_score(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels), average=None)
        if self.config.model.type == "single-label":
            cm = confusion_matrix(y_true=y_true, y_pred=y_predict, labels=range(self.num_labels))
            cmd_obj = ConfusionMatrixDisplay(cm, display_labels=self.labels_list)
            fig, ax = plt.subplots(figsize=(10, 10))
            cmd_obj.plot(xticks_rotation=30, ax=ax)
            cmd_obj.ax_.set(
                title='Confusion Matrix',
                xlabel='Predicted Labels',
                ylabel='True Labels')

            plt.axis('scaled')
            if num_epoch is not None:
                path_to_plot: str = str(
                    Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "plots",
                         f'confusion_matrix_{mode}_epoch_{num_epoch}.png').absolute())
            else:
                path_to_plot = str(
                    Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "plots",
                         f'confusion_matrix_{mode}.png').absolute())
            plt.savefig(path_to_plot, dpi=100)
            plt.close()
        else:
            if num_epoch is not None:
                path_to_plot: str = str(
                    Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "plots",
                         f'confusion_matrix_{mode}_epoch_{num_epoch}.png').absolute())
            else:
                path_to_plot = str(
                    Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "plots",
                         f'confusion_matrix_{mode}.png').absolute())
        f1_per_class = {label: f1_per_class[i] for i, label in enumerate(self.labels_list)}
        if self.config.data.config is not None and "stance" in self.config.data.config:
            other = (f1_per_class["against"] + f1_per_class["favor"])/2
        else:
            other = None
        results = ClassificationResult(**{"acc": acc,
                                          "loss": loss,
                                          "f1_micro": f1_micro,
                                          "f1_macro": f1_macro,
                                          "recall_macro": recall_macro,
                                          "precision_macro": precision_macro,
                                          "f1_per_class": f1_per_class,
                                          "path_to_plot": path_to_plot,
                                          "other": other,
                                          "encoded_feature_silhouette": 0,
                                          "final_output_silhouette": 0
                                          }
                                       )

        return results


class DynamicEnvironment(Environment):
    def __init__(self, config: DictConfig):
        super(DynamicEnvironment, self).__init__(config)

    def load_environment(self, mode: str, training_type: str) -> DataLoader:
        raise NotImplementedError

    def update_state(self, state: Dict):
        raise NotImplementedError

