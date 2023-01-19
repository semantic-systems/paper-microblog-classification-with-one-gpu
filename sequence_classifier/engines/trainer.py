import json
import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict
from abc import abstractmethod

import torch
from omegaconf import DictConfig
from torch import tensor

from sequence_classifier.engines.agent import Agent, BatchLearningAgent
from sequence_classifier.engines.environment import Environment, StaticEnvironment
from sequence_classifier.helper import fill_config_with_num_classes, get_data_time, set_run_training, set_run_testing
from sequence_classifier.schema import ClassificationResult, AgentPolicyOutput
from utils import set_seed
from sequence_classifier.validate import ConfigValidator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.previous_loss = 9999
        self.best_score = None

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup()
        self.environment = self.instantiate_environment()
        # update config according to envrionment instance - num_labels
        self.config.model.layers = fill_config_with_num_classes(self.config.model.layers,
                                                                self.environment.num_labels)
        self.agent = self.instantiate_agent()
        self.best_agent = None
        self.print_trainer_info()

    def run(self):
        self.train()
        self.test()

    def setup(self):
        set_seed(self.config.seed)
        validator = ConfigValidator(self.config)
        self.config = validator()

    @property
    def training_type(self):
        if "episode" in self.config:
            return "episodic_training"
        else:
            return "batch_training"

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def instantiate_environment(self):
        raise NotImplementedError

    @abstractmethod
    def instantiate_agent(self):
        raise NotImplementedError

    def print_trainer_info(self):
        logger.warning(f"Training Info: \n"
                       f"    Trainer: {self.__class__.__name__}\n"
                       f"    Agent: {self.agent.__class__.__name__}\n"
                       f"    Environment: {self.environment.__class__.__name__}\n"
                       f"    Policy: {self.agent.policy.__class__.__name__}\n"
                       f"    OutputPath: {self.config.model.output_path}")

    @staticmethod
    def log_result(result_per_epoch: ClassificationResult, final_result: List, epoch: Optional[int] = None):
        result_this_epoch = result_per_epoch.__dict__
        if epoch:
            result_this_epoch.update({"epoch": epoch})
        result_this_epoch.pop('path_to_plot', None)
        final_result.append(result_this_epoch)

    def save_best_model(self, best_validation_metric: int, result_per_epoch: ClassificationResult):
        if result_per_epoch.f1_macro > best_validation_metric:
            #label_index_map = dict([(str(value), key) for key, value in self.environment.label_index_map.items()])
            #self.agent.policy.save_model(
            #    Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "pretrained_models",
            #         f"{self.config.name}_best_model.pt").absolute(),
            #    index_label_map=label_index_map)
            self.best_agent = deepcopy(self.agent)

    def save_final_model(self):
        label_index_map = dict([(str(value), key) for key, value in self.environment.label_index_map.items()])
        self.agent.policy.save_model(
            Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}", "pretrained_models",
                 f"{self.config.name}_{get_data_time()}.pt").absolute(),
            index_label_map=label_index_map)

    def convert_tensor_index_to_label(self, labels: List[tensor]) -> List:
        if self.config.model.type == "single-label":
            labels = [label.item() for label in labels]
            return list(map(lambda index: self.environment.index_label_map[str(index)], labels))
        elif self.config.model.type == "multi-label":
            final_labels = []
            for label in labels:
                index_label = [i for i, l in enumerate(label) if l == 1]
                text_label = list(map(lambda index: self.environment.index_label_map[str(index)], index_label))
                final_labels.append(text_label)
            return final_labels
        else:
            raise NotImplementedError


class SingleAgentTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super(SingleAgentTrainer, self).__init__(config)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> Agent:
        raise NotImplementedError

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


class MultiAgentTrainer(Trainer):
    def __init__(self, config: DictConfig):
        super(MultiAgentTrainer, self).__init__(config)

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def instantiate_environment(self) -> Environment:
        return Environment(self.config)

    def instantiate_agent(self) -> List[Agent]:
        raise NotImplementedError


class BatchLearningTrainer(SingleAgentTrainer):
    def __init__(self, config: DictConfig):
        super(BatchLearningTrainer, self).__init__(config)
        if self.config.data.validation:
            self.early_stopping = EarlyStopping(tolerance=self.config.early_stopping.tolerance,
                                                min_delta=self.config.early_stopping.delta)

    def instantiate_environment(self) -> Environment:
        return StaticEnvironment(self.config)

    def instantiate_agent(self) -> Agent:
        return BatchLearningAgent(self.config, self.device, self.environment.class_weights)

    @set_run_training
    def train(self):
        # run per batch
        data_loader = self.environment.load_environment("train", self.training_type)
        if self.config.data.validation:
            validation_data_loader = self.environment.load_environment("validation", self.training_type)
        self.agent.policy.to(self.agent.policy.device)
        self.agent.policy.train()
        self.agent.policy.optimizer.zero_grad()
        train_result = []
        validation_result = []
        best_validation_metric = 0
        encoded_feature_silhouette = 0
        final_output_silhouette = 0
        # start new run
        for n in range(self.config.model.epochs):
            # training
            self.agent.policy.lr_scheduler.step()
            agent_output: AgentPolicyOutput = self.agent.act(data_loader, mode="train")
            y_predict, y_true, train_loss = agent_output.y_predict, agent_output.y_true, agent_output.loss
            ce_loss, contrastive_loss = agent_output.cross_entropy_loss, agent_output.contrastive_loss
            train_result_per_epoch: ClassificationResult = self.environment.evaluate(y_predict,
                                                                                     y_true,
                                                                                     train_loss,
                                                                                     mode="train",
                                                                                     num_epoch=n)
            logger.warning(f"Training results:")
            logger.warning(f"Epoch: {n}, Average loss: {train_loss}, CE loss: {ce_loss}, Contrastive loss: {contrastive_loss}, "
                           f"Average acc: {train_result_per_epoch.acc}, "
                           f"F1 macro: {train_result_per_epoch.f1_macro},"
                           f"F1 micro: {train_result_per_epoch.f1_micro}, "
                           f"F1 per class: {train_result_per_epoch.f1_per_class}, "
                           f"Precision macro: {train_result_per_epoch.precision_macro}, "
                           f"Recall macro: {train_result_per_epoch.recall_macro}, "
                           f"Other: {train_result_per_epoch.other},"
                           f"lr: {self.agent.policy.optimizer.param_groups[0]['lr']}")
            self.log_result(result_per_epoch=train_result_per_epoch, final_result=train_result, epoch=n)

            # validation
            if self.config.data.validation:
                agent_output: AgentPolicyOutput = self.agent.act(validation_data_loader, mode="validation")
                y_predict, y_true, validation_loss = agent_output.y_predict, agent_output.y_true, agent_output.loss
                # if "tsne" in self.config.visualizer:
                #     tsne_feature = agent_output.tsne_feature
                #     tsne_feature.labels = self.convert_tensor_index_to_label(y_true)
                #     feature_to_store = tsne_feature.__dict__
                #     feature_to_store.update({"epoch": n})
                #     feature_to_store.update({"y_true": y_true})
                #     with open(Path(self.config.model.output_path, self.config.name, f"seed_{self.config.seed}",
                #                    "pretrained_models", "validation_encoded_features.pkl").absolute(), 'ab') as f:
                #         pickle.dump(feature_to_store, f)
                    # self.environment.visualize_embedding(tsne_feature=tsne_feature, epoch=n)
                    # encoded_feature_silhouette = self.environment.clustering_score(tsne_feature.encoded_features,
                    #                                                                y_true)
                    # final_output_silhouette = self.environment.clustering_score(tsne_feature.final_hidden_states,
                    #                                                             y_true)

                validation_result_per_epoch: ClassificationResult = self.environment.evaluate(y_predict,
                                                                                              y_true,
                                                                                              validation_loss,
                                                                                              mode="validation",
                                                                                              num_epoch=n)
                validation_result_per_epoch.encoded_feature_silhouette = encoded_feature_silhouette
                validation_result_per_epoch.final_output_silhouette = final_output_silhouette
                logger.warning(f"Validation results:")
                logger.warning(
                    f"Epoch: {n}, Average loss: {validation_loss}, Average acc: {validation_result_per_epoch.acc}, "
                    f"F1 macro: {validation_result_per_epoch.f1_macro},"
                    f"F1 micro: {validation_result_per_epoch.f1_micro}, "
                    f"F1 per class: {validation_result_per_epoch.f1_per_class}, "
                    f"Precision macro: {validation_result_per_epoch.precision_macro}, "
                    f"Recall macro: {validation_result_per_epoch.recall_macro}, "
                    f"Other: {validation_result_per_epoch.other}"
                    f"encoded_feature_silhouette: {validation_result_per_epoch.encoded_feature_silhouette}, "
                    f"final_output_silhouette: {validation_result_per_epoch.final_output_silhouette}")
                self.log_result(result_per_epoch=validation_result_per_epoch, final_result=validation_result, epoch=n)
                self.save_best_model(best_validation_metric, validation_result_per_epoch)
                # early stopping
                self.early_stopping(validation_loss)
                if self.early_stopping.early_stop:
                    logger.warning(f"Early stopping reached at epoch: {n}")
                    break

        self.environment.dump_result(train_result, mode='train')
        self.environment.dump_config()
        if self.config.data.validation:
            self.environment.dump_result(validation_result, mode='validation')
        else:
            # save the final trained model -> not recommended
            self.save_final_model()

    @set_run_testing
    def test(self):
        if self.best_agent is not None:
            self.agent = self.best_agent
            self.best_agent = None
        data_loader = self.environment.load_environment("test", self.training_type)
        self.agent.policy.eval()
        test_result = []
        encoded_feature_silhouette = 0
        final_output_silhouette = 0
        with torch.no_grad():
            agent_output: AgentPolicyOutput = self.agent.act(data_loader, mode="test")
            y_predict, y_true, loss = agent_output.y_predict, agent_output.y_true, agent_output.loss
            test_data: Dict = {"text": agent_output.test_input_text,
                               "label": self.convert_tensor_index_to_label(y_true),
                               "prediction": self.convert_tensor_index_to_label(y_predict)}
            if "tsne" in self.config.visualizer:
                tsne_feature = agent_output.tsne_feature
                labels = [label.item() for label in y_true]
                tsne_feature.labels = list(map(lambda index: self.environment.index_label_map[str(index)], labels))
                self.environment.visualize_embedding(tsne_feature=tsne_feature)
                encoded_feature_silhouette = self.environment.clustering_score(tsne_feature.encoded_features, y_true)
                final_output_silhouette = self.environment.clustering_score(tsne_feature.final_hidden_states, y_true)
            result = self.environment.evaluate(y_predict, y_true, loss, mode="test")
            result.encoded_feature_silhouette = encoded_feature_silhouette
            result.final_output_silhouette = final_output_silhouette
            logger.warning(f"Testing Accuracy: {result.acc}, Loss: {result.loss}, F1 micro: {result.f1_micro},"
                           f"F1 macro: {result.f1_macro}, F1 per class: {result.f1_per_class}, "
                           f"Precision macro: {result.precision_macro}, "
                           f"Recall macro: {result.recall_macro}, "
                           f"Other: {result.other}, "
                           f"encoded_feature_silhouette: {result.encoded_feature_silhouette}, "
                           f"final_output_silhouette: {result.final_output_silhouette}")
            self.log_result(result_per_epoch=result, final_result=test_result)
            self.environment.dump_result(test_result, mode='test')
            self.environment.dump_csv(test_data)
