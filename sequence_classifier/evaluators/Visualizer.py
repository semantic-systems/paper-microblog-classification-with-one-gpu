from typing import Optional, Union, Dict, Tuple

import numpy as np
from abc import abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from torch import tensor

from sequence_classifier.schema import FeatureToVisualize, InputFeature, SingleLabelClassificationForwardOutput
# from mayavi import mlab
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sequence_classifier.models.SingleLabelSequenceClassification import SingleLabelSequenceClassification, SingleLabelContrastiveSequenceClassification


Model = Union[SingleLabelSequenceClassification, SingleLabelContrastiveSequenceClassification]


class Visualizer(object):
    def __init__(self):
        self.name = self.__class__

    @abstractmethod
    def visualize(self, data: FeatureToVisualize, path_to_save: Optional[str]=None):
        raise NotImplementedError


class SphericalVisualize(Visualizer):
    def __init__(self):
        super(SphericalVisualize, self).__init__()

    def visualize(self, data: FeatureToVisualize, path_to_save: Optional[str]=None):
        # Create a sphere
        r = 1.0
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)

        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
        mlab.clf()
        normalized_catesian = self.normalize(data.feature)
        mlab.mesh(x, y, z, color=(0.0, 0.5, 0.5), opacity=0.5)
        mlab.points3d(normalized_catesian[:, 0], normalized_catesian[:, 1], normalized_catesian[:, 2],
                      [int(n) for n in data.labels], scale_mode='none',
                      scale_factor=0.05)
        mlab.show()

    def normalize(self, x: np.array):
        return normalize(x, norm="l2")


class TSNEVisualizer(Visualizer):
    def __init__(self, n_components=2, perplexity=15, n_iter=1000):
        super(TSNEVisualizer, self).__init__()
        self.model = TSNE(n_components=n_components, verbose=0, perplexity=perplexity, n_iter=n_iter)

    def visualize(self, data: FeatureToVisualize, path_to_save: Optional[str] = None):
        tsne_results = self.model.fit_transform(np.asarray(data.feature))
        df = pd.DataFrame()
        df["y"] = data.labels
        df["comp-1"] = tsne_results[:, 0]
        df["comp-2"] = tsne_results[:, 1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", len(list(set(data.labels)))),
                        data=df).set(title="T-SNE projection")
        if path_to_save is not None:
            plt.savefig(path_to_save)
        plt.clf()


def load_model(path_to_pretrained_model: str, model_type: str) -> Tuple[Model, DictConfig, dict]:
    if model_type == "sl":
        model_class = SingleLabelSequenceClassification
    elif model_type == "scl":
        model_class = SingleLabelContrastiveSequenceClassification
    else:
        raise NotImplementedError

    checkpoint = torch.load(path_to_pretrained_model, map_location=torch.device('cpu'))
    model = model_class(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['config'], checkpoint['index_label_map']


def get_feature(batch: Dict, model: Model, device: Optional[str] = "cpu") -> FeatureToVisualize:
    tokenized_text = model.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
    input_ids: tensor = tokenized_text["input_ids"].to(device)
    attention_masks: tensor = tokenized_text["attention_mask"].to(device)
    labels = None
    input_feature: InputFeature = InputFeature(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
    output: SingleLabelClassificationForwardOutput = model.forward(input_feature, mode="test")
    features = output.prediction_logits.cpu().detach().numpy()
    labels = batch["label"].cpu().detach().numpy()
    return FeatureToVisualize(**{"feature": features, "labels": labels})



if __name__ == "__main__":
    # data = FeatureToVisualize(feature=np.random.random((100, 3)), labels=[str(n) for n in list(range(0, 50))+list(range(0, 50))])
    path_to_model = "./outputs/crisis/crisis_bert_base_uncased/seed_0/pretrained_models/crisis_bert_base_uncased_30_08_2022_05_28_06.pt"
    visualizer = TSNEVisualizer()
    model, config, index_label_map = load_model(path_to_model, model_type="sl")
    config.data.batch_size = 200
    from sequence_classifier.engines.environment import StaticEnvironment
    env = StaticEnvironment(config)
    data_loader = env.load_environment("test", "batch_training")
    features = None
    for i, batch in enumerate(tqdm(data_loader)):
        features = get_feature(batch, model)
        if features is not None:
            break
    def convert_index_to_label(index: int) -> str:
        return index_label_map.get(str(index), None)
    labels = list(map(convert_index_to_label, features.labels))
    print(labels)
    features.labels = labels
    visualizer.visualize(features, "./tsne.png")

