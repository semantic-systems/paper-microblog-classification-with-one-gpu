from itertools import chain
from typing import Optional

from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AdamW, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, \
    get_linear_schedule_with_warmup
from sequence_classifier.models import SequenceClassification
from sequence_classifier.models.heads import DenseLayerHead
from sequence_classifier.schema import SingleLabelClassificationForwardOutput, InputFeature, EncodedFeature
from sequence_classifier.losses.supervised_contrastive_loss import SupervisedContrastiveLoss


class SingleLabelSequenceClassification(SequenceClassification):
    def __init__(self, cfg: DictConfig, class_weights: Optional[list] = None):
        super(SingleLabelSequenceClassification, self).__init__(cfg, class_weights)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=False)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 10, 2*cfg.model.epochs)
        self.loss = CrossEntropyLoss(weight=self.class_weights, reduction="mean")

    def forward(self,
                input_feature: InputFeature,
                mode: str,
                backward: Optional[bool] = False) -> SingleLabelClassificationForwardOutput:
        output = self.inference(input_feature, mode)
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature, mode=mode)

        if mode == "train":
            loss = self.loss(head_output.output, input_feature.labels)
            loss = loss / self.cfg.data.gradient_accu_step
            loss.backward()
            if backward:
                self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "validation":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "test":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        else:
            raise ValueError(f"mode {mode} is not one of train, validation or test.")

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        encoder = self.freeze_encoder(encoder, self.cfg.model.freeze_transformer_layers)
        return encoder

    def instantiate_classification_head(self) -> DenseLayerHead:
        return DenseLayerHead(self.cfg)


class SingleLabelContrastiveSequenceClassification(SingleLabelSequenceClassification):
    def __init__(self, cfg: DictConfig, class_weights: Optional[list] = None):
        super(SingleLabelContrastiveSequenceClassification, self).__init__(cfg, class_weights)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=True)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 10, 2*cfg.model.epochs)
        self.loss = CrossEntropyLoss(weight=self.class_weights, reduction="mean")
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=cfg.model.contrastive.temperature,
                                                          base_temperature=cfg.model.contrastive.base_temperature,
                                                          contrast_mode=cfg.model.contrastive.contrast_mode)
        self.contrastive_loss_ratio = cfg.model.contrastive.contrastive_loss_ratio

    def forward(self,
                input_feature: InputFeature,
                mode: str,
                backward: Optional[bool] = False) -> SingleLabelClassificationForwardOutput:
        output = self.inference(input_feature, mode)
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature, mode=mode)

        if mode == "train":
            loss = self.loss(head_output.output, input_feature.labels)
            # reshape to compute the loss
            contrastive_features, contrastive_labels = self.get_multiview_batch(head_output.output, input_feature.labels)
            contrastive_loss = self.contrastive_loss(contrastive_features, contrastive_labels)
            total_loss = (1 - self.contrastive_loss_ratio) * loss + self.contrastive_loss_ratio * contrastive_loss
            total_loss = total_loss / self.cfg.data.gradient_accu_step
            total_loss.backward()
            if backward:
                self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=total_loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature,
                                                          cross_entropy_loss=loss.item(),
                                                          contrastive_loss=contrastive_loss.item())
        elif mode == "validation":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "test":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        else:
            raise ValueError(f"mode {mode} is not one of train, validation or test.")



