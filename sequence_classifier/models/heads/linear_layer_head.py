from typing import Optional
from omegaconf import DictConfig
from torch import tensor
from torch.nn import Linear, ModuleList, Dropout

from sequence_classifier.models.heads import Head
from sequence_classifier.schema import EncodedFeature, HeadOutput
import torch.nn.functional as F


class DenseLayerHead(Head):
    def __init__(self, cfg: DictConfig):
        super(DenseLayerHead, self).__init__()
        layer_stacks = [Linear(layer.n_in, layer.n_out) for layer in cfg.model.layers.values()]
        self.head_type = "mlp" if len(layer_stacks) > 1 else "linear"
        self.classification_layer = ModuleList(layer_stacks)
        self.dropout = Dropout(p=cfg.model.dropout_rate)
        self.l2_normalize_encoded_feature = cfg.model.L2_normalize_encoded_feature
        self.l2_normalize_logits = cfg.model.L2_normalize_logits

    def forward(self, encoded_features: EncodedFeature, mode: str) -> HeadOutput:
        encoded_feature: tensor = encoded_features.encoded_feature
        output = encoded_feature
        if self.head_type == "linear":
            if self.l2_normalize_encoded_feature:
                norm = output.norm(p=2, dim=1, keepdim=True)
                output = output.div(norm.expand_as(output))
            output = self.classification_layer[0](output)
            if self.l2_normalize_logits:
                norm = output.norm(p=2, dim=1, keepdim=True)
                output = output.div(norm.expand_as(output))
        else:
            for i, layer in enumerate(self.classification_layer):
                if i < len(self.classification_layer) - 1:
                    if mode == "train":
                        output = F.relu(self.dropout(layer(output)))
                    elif mode in ["validation", "test"]:
                        output = F.relu(layer(output))
                else:
                    if self.l2_normalize_encoded_feature:
                        norm = output.norm(p=2, dim=1, keepdim=True)
                        output = output.div(norm.expand_as(output))
                    output = layer(output)
                    if self.l2_normalize_logits:
                        norm = output.norm(p=2, dim=1, keepdim=True)
                        output = output.div(norm.expand_as(output))
        return HeadOutput(output)
