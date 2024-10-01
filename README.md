# Revisiting Supervised Contrastive Learning for Microblog Classification
## Requirement
- python>=3.7.*
- torch>=1.10.0
- transformers>=4.12.5
- hydra==1.1.1
- mlflow==1.21.0

The required packages are defined in the `requirements.txt`, which uses the cpu version of pytorch.
To use the gpu version, please visit follow the official [installation guideline](https://pytorch.org/get-started/locally/).
Please install the packages in your own virtual environment with the following command:
```
pip install -r requirements.txt
```


## Overview
This repository contains source code for the paper submitted to EMNLP 2024 - Revisiting 
Supervised Contrastive Learning for Microblog Classification - under Apache 2.0 License.

### How to write a configuration file?
The configuration file is utilized by hydra with a .yaml file. To set up a configuration, one can refer to one of the example 
configuration under `configs/`. Note that there is no predefined schema for the config. One can freely add arguments in the 
yaml file (If you do so, some tests might fail). To use the arguments defined in the config, one has two options:
- attribute style access: `config.model.from_pretrained`
- dictionary style access: `config["model"]["from_pretrained"]`

### How to observe experiments in your localhost
Before running an experiment, you must open mlflow locally. (not nice, i know.)
```
mlflow ui
```
The default host is set to be http://127.0.0.1:5000 . Simply open it in your browser.

### How to run an experiment?
```
python main.py <path/to/config>
```
So far you need to define the configuration file within the `main.py`. 
Configurations for both TweetEval and Tweet Topic Classification are included under `./sequence_classifier/configs/` 

## Pipeline
The pipeline of training or fine-tuning a language model with a downstream classifier contains the following parts.
1. a Dataset class, which inherits the implementation of a [huggingface dataset](https://huggingface.co/datasets).
2. a Generator class, which the model class uses for different training schemes. For example, for regular supervised batch training, one might want a Generator to loop over the dataset batch-wise.
While for few shot learning, one might desire an episodic sampler within the Generator.
3. a Validator class, which takes the config instances as input and validate the value type, defined in the validator. This is because hydra does not support validation of values.
4. a Model class, which defines the training pipeline, given a downstream classification task.
5. an Evaluator class, which reduces the evaluation process depending on the type of output data given by the model object.
6. an Agent class, which interacts with its environment and makes decision and gets evaluated by the environment. For RL problems, this is intuitive. A classification problem can be viewed as an instance of the RL problem, where the evaluation happens after each action. This is to say, a classification task can be viewed as a one-step RL problem.
7. an Environment class, where the datasets and dataloaders are stored, as well as the evaluation scripts (Evaluator). 
8. a Trainer class, which is an Injector, which utilizes all the other classes in a centralized manner. Logging, training, instantiating.

## Dataset
The huggingface Dataset class contains three methods to be customized. 
```
def _info(self):
    return datasets.DatasetInfo(...)
```
The `_info()` method defines the meta information of this dataset. This includes:
- the description of the dataset
- the class label of the dataset
- the value type for each column
- the homepage if it exists
- the citation
```
def split_generators(self):
    ## do something to get the data from a URL or a local file(s)
    ## create training/validation/testing data placeholder by assigning the path to the data
```
The `_split_generators()` method is in charge of downloading from a URL or referring to local files, 
as well as split them into various type of data, training, validation or testing.

```
def _generate_examples(self):
    ## read the data
    ## select information that you want to keep in the data 
```
The `_generate_examples()` method already has the information of the downloaded/referred data from the web/local machine.
In this method, the information selection process of reading a data is happening. For example, if your dataset has various columns
storing lots of different information, but you are only interested in some of those. You should define the information of interet here.

## Generator
The Generator class is instantiated with the config of type DictConfig defined in hydra. 

In the constructor, number of labels (`num_labels`) must be defined, as well as a `label_index_map` of type dictionary.

There are two properties in this class, which is the `training_dataset` and `testing_dataset` (also `validation_dataset` in the future).
The property loads the dataset from remote of locally, which is used in the constructor (typically) to define number of labels.

During the loading process of the dataset, a renaming of the label column will happen. This is because the pipeline expects 
the label column of all dataset to be the same. Such renaming allows the pipeline to be functioning, invariant to datasets.

The last step of customizing a Genarator to the implementation of the `__call__` method, which creates a pytorch Dataloader.
Note that, if you wish to change a sampler, the instantiated sampler object must be given as an argument (`sampler=Optional[Sampler]`).

## Validator
The validator class is responsible for validating the value in the config file. This project uses
hydra as a configuration manager. One drawback of hydra is that it does not support automatic validation of values.
This creates trouble because I want to make the error within each component to stay only in that component. Therefore a validator should 
be implemented for each type of config scheme. This component is highly customizable.

## Model
The Model class is the main blood of this repository. It constructs an `encoder` (pre-trained language model), 
a `feature_transformer` (representation learning after the encoded feature in the latent feature space, or projecting 
the features onto another space such as hyperbolic space) and a `classification_head` (whose job is to construct the classifier)

One must implement the forward method as in a pytorch fashion. The general steps will be as follows:
1. `encoder` encodes the raw text features;
2. `feature_transformer` transforms the encoded features;
3. `classification_head` makes prediction based on the transformed features.

Note that the `classification_head` by itself is also a `torch.nn.Module`, which process information per training batch/episode/a set of data.

### Methods
1. `trim_encoder_layers(encoder: PreTrainedModel, num_layers_to_keep: int) -> PreTrainedModel` 
- In some situation, it is desirable to use only part of the transformer layers in the pre-trained language model.
2. `freeze_encoder(encoder: PreTrainedModel, layers_to_freeze: Union[str, int]) -> PreTrainedModel`
- In some situation, it is desirable __NOT__ to train the encoder. For example it might help generalization in few-shot setting.
3. `preprocess(self, batch)`
- The preprocessing step does not happen within the Generator, but here. The reason for that is to have a consistent pipeline 
in both training and testing step. This is because for application, the input sentence is going directly into the Model class. 


## Agent
The Agent class represents the body where the classifier sits. It has its policy, where decisions are made; action space, which is, in classification, the set of labels.
In each training/time step, it receives raw (textual) features and processes it and makes prediction given the task.
It contains the following methods.
1. `policy_class(self)`
- where you select the policy/model that does the downstream prediction tasks. It can be, for example, a model where you define a language model and a prototypical networks as a classifier.
5. `act(self)`
- the within-batch/episode level inference with the model defined as a policy.

## Environment
The Environment class defines the world/environment where the agent is sitting. In classification, this will be where the datasets and their loaders are stored. 
The configuration of datasets and dataloaders are defined to be the dynamics of this environment.

## Trainer
The injector class Trainer instantiates all the higher-level classes. On top of that, it has the following methods.
1. `train(self)`
- This method defines the highest level training process - iterating over batches in batch learning, iterating over iterations in meta learning 
5. `test(self)`
- same thing as above but for testing.
