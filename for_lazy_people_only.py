import abc
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional


class Result(object):
    def __init__(self, path: str, root: str):
        self.root = root
        self.path = path
        self.result = self.read_json(path)
        self.encoded_feature_silhouette = self.result[0].get("encoded_feature_silhouette", 0)
        self.final_output_silhouette = self.result[0].get("final_output_silhouette", 0)
        config_path = path.replace("test_result.json", "config.yaml")
        self.config = self.read_yaml(config_path)
        self.task = self.get_task()
        self.seed = self.config.get("seed")
        self.name = self.config.get("name")
        self.include_oos = self.config.get("data").get("include_oos", False)
        self.batch_size = self.config.get("data").get("batch_size")
        self.early_stopping_patience = self.config.get("early_stopping").get("tolerance")
        self.l2_normalized_encoded_feature = self.config.get("model").get("L2_normalize_encoded_feature")
        self.epochs = self.config.get("model").get("epochs")
        self.freeze_transformer_layers = self.config.get("model").get("freeze_transformer_layers")
        self.model = self.get_model()
        self.head_type = "linear" if len(self.config.get("model").get("layers")) == 1 else "mlp"
        self.learning_rate = self.config.get("model").get("learning_rate")
        self.dropout = self.config.get("model").get("dropout")
        self.contrastive = self.is_contrastive()
        self.contrastive_loss_ratio = 0 if not self.contrastive else self.config.get("model").get("contrastive").get("contrastive_loss_ratio")
        self.base_temperature = 0 if not self.contrastive else self.config.get("model").get("contrastive").get("base_temperature")
        self.contrastive_temperature = 0 if not self.contrastive else self.config.get("model").get("contrastive").get("temperature")
        self.contrast_mode = np.nan if not self.contrastive else self.config.get("model").get("contrastive").get("contrast_mode")
        self.augmenter = self.get_augmenter()
        self.num_augmented_samples = self.config.get("augmenter").get("num_samples", None) if self.augmenter else np.nan
        self.augmenter_dropout = str(self.config.get("augmenter").get("dropout", None)) if self.augmenter else np.nan
        self.metric_name = self.get_metric_name(self.task)
        self.metric = self.get_metric(self.metric_name)

    @staticmethod
    def read_yaml(yaml_to_read: str) -> Dict:
        with open(yaml_to_read, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            d = json.load(f)
        return d

    def get_task(self) -> str:
        data = self.config.get("data").get("name")
        task = self.config.get("data").get("config", None)
        if task not in [None, "none"]:
            if task == "stance_abortion":
                task = "stance"
            return task
        else:
            task = data.split("/")[-1].split(".py")[0]
            if task == "TRECIS_event_type":
                return "Event Type"
            elif "sexism" in task:
                return task
            elif "tweet_topic_single" in task:
                return task
            else:
                raise ValueError("task not known.")

    def get_model(self) -> str:
        model_map = {"roberta-base": "Rob-bs",
                     "vinai/bertweet-base": "Bertweet",
                     "bert-base-uncased": "Bert-bs-uncased",
                     "language_models/CoyPu-CrisisLM-v1": "CrisisLM"}
        model = self.config.get("model").get("from_pretrained")
        return model_map[model]

    def is_contrastive(self) -> bool:
        if self.config.get("model").get("contrastive", None) is None:
            is_contrastive = False
        elif self.config.get("model").get("contrastive").get("contrastive_loss_ratio") == 0:
            is_contrastive = False
        else:
            is_contrastive = True
        return is_contrastive

    def get_augmenter(self) -> bool:
        if self.config.get("augmenter") is None:
            augmenter = None
        else:
            augmenter = self.config.get("augmenter").get("name")
        return augmenter

    @staticmethod
    def get_metric_name(task):
        raise NotImplementedError

    def get_metric(self, metric_name) -> float:
        if len(metric_name) == 1:
            metric = self.result[0].get(metric_name[0])
        elif len(metric_name) == 2:
            try:
                metric = self.result[0].get(metric_name[0]).get(metric_name[1])
            except AttributeError:
                metric
        else:
            raise ValueError
        return metric


class CrisisResult(Result):

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"Event Type": ["f1_macro"]}
        return metric_dict[task]


class SexismResult(Result):

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"sexism_level_three": ["f1_macro"],
                       "sexism_level_two": ["f1_macro"],
                       "sexism_level_one": ["f1_macro"]}
        return metric_dict[task]


class SemevalResult(Result):

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"subtask5.english": ["f1_macro"]}
        return metric_dict[task]


class TweetTopicResult(Result):

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"tweet_topic_single": ["f1_macro"]}
        return metric_dict[task]


class TweetEvalResult(Result):

    @staticmethod
    def get_metric_name(task):
        metric_dict = {"stance": ["other"],
                       "sentiment": ["recall_macro"],
                       "offensive": ["f1_macro"],
                       "irony": ["f1_per_class", "irony"],
                       "hate": ["f1_macro"],
                       "emotion": ["f1_macro"],
                       "emoji": ["f1_macro"]}

        task = "stance" if "stance" in task else task
        return metric_dict[task]


class Table(object):
    def __init__(self, result_df: pd.DataFrame):
        self.result_df = result_df

    @staticmethod
    def write_top(session_to_include: List, task_list: List):
        table_alignment = ""
        column_string = ""
        for i, col in enumerate(session_to_include+task_list):
            table_alignment += "c|"
            if col == "contrastive_loss_ratio":
                col = "ratio"
            if col == "head_type":
                col = "cls"
            if col == "contrastive_temperature":
                col = "tmp"
            if col == "contrast_mode":
                col = "mode"
            if col == "base_temperature":
                col = "bs_temp"
            if col == "augmenter_dropout":
                col = "dropout"
            column_string += f"{col.replace('_', '-')}&"

            if i == len(session_to_include+task_list)-1:
                table_alignment = table_alignment[:-1]
                column_string = column_string[:-1]
        top_string = "\\scalebox{0.75}{\n" \
                      "\\begin{center}\n" \
                      f"\\begin{{tabular}}{{{table_alignment}}}\n" \
                      "\\hline\n" \
                      f"{column_string}\\\ \n" \
                      "\\hline\\hline \n"
        return top_string

    def get_target_df(self, row: Tuple):
        df = self.result_df
        for (key, value) in row:
            df = df[(df[key] == value)]
        return df

    def write_row(self, row: Tuple, task_list: List, col_to_write: Optional[str]="metric_score"):
        tex = ""
        scores = {task: {"avg": 0, "std": 0, "max": 0} for task in task_list}
        for element in row:
            tex += f"{list(element)[1]}&"
        for i, task in enumerate(task_list):
            df = self.get_target_df(row)
            scores[task]["avg"] = df[(df["task"] == task)][col_to_write].mean()
            scores[task]["std"] = df[(df["task"] == task)][col_to_write].std()
            scores[task]["max"] = df[(df["task"] == task)][col_to_write].max()
        for i, task in enumerate(task_list):
            tex += f"{round(100*scores[task]['avg'], 1)}\small$\pm${round(100*scores[task]['std'], 1)}\\thinspace({round(100*scores[task]['max'], 1)})&"
            if i == len(task_list)-1:
                tex = tex[:-1]
        tex += "\\\ \n"
        return tex

    def write_end(self, session_to_include: List, task_list: List, result: Result) -> str:
        metric_string = ""
        for i, col in enumerate(session_to_include+task_list):
            if i <= len(session_to_include)-1:
                metric_string += "&"
            else:
                metric_string += f"{result.get_metric_name(col)[0].replace('_', '-')}&"
            if i == len(session_to_include+task_list):
                metric_string = metric_string[:-1]
        string = "\\hline\\hline\n" \
                 f"\\textbf{{Metric}}{metric_string}\n" \
                 "\\end{tabular}\n" \
                 "\\end{center}}"
        return string


class TweetEvalMainTable(Table):
    def __init__(self, result_df: pd.DataFrame):
        super(TweetEvalMainTable, self).__init__(result_df)

    @staticmethod
    def write_top(session_to_include: List, task_list: List):
        table_alignment = ""
        column_string = ""
        empty_grids = "&"*len(session_to_include)
        col_list = [task for task in task_list]
        col_order = {'emoji': 0, 'emotion': 1, 'hate': 2, 'irony': 3, 'offensive': 4, 'sentiment': 5, 'stance': 6}
        col_list.sort(key=lambda col: col_order[col])
        col_list.extend(["All"])
        for i, col in enumerate(session_to_include+col_list):
            table_alignment += "c|"
            if i == len(session_to_include+col_list)-2:
                table_alignment += "|"
            if col == "contrastive_loss_ratio":
                col = "ratio"
            if col == "head_type":
                col = "cls"
            if col == "contrastive_temperature":
                col = "tmp"
            if col == "contrast_mode":
                col = "mode"
            if col == "base_temperature":
                col = "bs_temp"
            column_string += f"{col.capitalize().replace('_', '-')}&"
            if i == len(session_to_include+col_list)-1:
                table_alignment = table_alignment[:-1]
                column_string = column_string[:-1]
        top_string = "\\scalebox{0.7}{\n" \
                      "\\begin{center}\n" \
                      f"\\begin{{tabular}}{{{table_alignment}}}\n" \
                      "\\hline\n" \
                      f"{column_string}\\\ \n" \
                      "\\hline\\hline \n" \
                     f"SVM{empty_grids} 29.3& 64.7&36.7&61.7&52.3&62.9&67.3&53.5\\\ \n" \
                     f"FastText{empty_grids} 25.8& 65.2&50.6&63.1&73.4&62.9&65.4&58.1\\\ \n" \
                     f"BLSTM{empty_grids} 24.7& 66.0&52.6&62.8&71.7&58.3&59.4&56.5\\\ \n" \
                     f"Rob-Bs{empty_grids} 30.9\small$\pm$0.2\\thinspace(30.8)& 76.1\small$\pm$0.5\\thinspace(76.6)& 46.6\small$\pm$2.5\\thinspace(44.9)&59.7\small$\pm$5.0\\thinspace(55.2)&79.5\small$\pm$0.7\\thinspace(78.7)& 71.3\small$\pm$1.1\\thinspace(72.0)&68.0\small$\pm$0.8\\thinspace(70.9)&61.3\\\ \n" \
                     f"Rob-RT{empty_grids} 31.4\small$\pm$0.4\\thinspace(31.6)& 78.5\small$\pm$1.2\\thinspace(79.8)& 52.3\small$\pm$0.2\\thinspace(55.5)&61.7\small$\pm$0.6\\thinspace(62.5)&80.5\small$\pm$1.4\\thinspace(81.6)& 72.6\small$\pm$0.4\\thinspace(72.9)&69.3\small$\pm$1.1\\thinspace(72.6)&65.2\\\ \n " \
                     f"Rob-Tw{empty_grids} 29.3\small$\pm$0.4\\thinspace(29.5)& 72.0\small$\pm$0.9\\thinspace(71.7)& 46.9\small$\pm$2.9\\thinspace(45.1)&65.4\small$\pm$3.1\\thinspace(65.1)&77.1\small$\pm$1.3\\thinspace(78.6)&69.1\small$\pm$1.2\\thinspace(69.3)&66.7\small$\pm$1.0\\thinspace(67.9)&61.0\\\ \n " \
                     f"XLM-R{empty_grids} 28.6\small$\pm$0.7\\thinspace(27.7)& 72.3\small$\pm$3.6\\thinspace(68.5)& 44.4\small$\pm$0.7\\thinspace(43.9)&57.4\small$\pm$4.7\\thinspace(54.2)&75.7\small$\pm$1.9\\thinspace(73.6)&68.6\small$\pm$1.2\\thinspace(69.6)&65.4\small$\pm$0.8\\thinspace(66.0)&57.6\\\ \n " \
                     f"XLM-Tw{empty_grids} 30.9\small$\pm$0.5\\thinspace(30.8)& 77.0\small$\pm$1.5\\thinspace(78.3)& 50.8\small$\pm$0.6\\thinspace(51.5)&69.9\small$\pm$1.0\\thinspace(70.0)&79.9\small$\pm$0.8\\thinspace(79.3)&72.3\small$\pm$0.2\\thinspace(72.3)&67.1\small$\pm$1.4\\thinspace(68.7)&64.4\\\ \n" \
                     "\\hline\n" \
                     f"Bertweet{empty_grids} 33.4&79.3&56.4&82.1&79.5&73.4&71.2&67.9\\\ \n" \
                     f"TimeLMs-2021{empty_grids} 34.0&80.2&55.1	&64.5&82.2&73.7&72.9&66.2\\\ \n" \
                     "\\hline\n"
        return top_string

    def write_row(self, row: Tuple, task_list: List, col_to_write: Optional[str]="metric_score"):
        coefict = 100 if col_to_write == "metric_score" else 1
        decimal = 1 if col_to_write == "metric_score" else 3
        tex = ""
        scores = {task: {"avg": 0, "std": 0, "max": 0} for task in task_list }
        col_list = [task for task in task_list]
        col_order = {'emoji': 0, 'emotion': 1, 'hate': 2, 'irony': 3, 'offensive': 4, 'sentiment': 5, 'stance': 6}
        col_list.sort(key=lambda col: col_order[col])
        col_list.extend(["all"])
        stance_avg_score = {}
        for element in row:
            tex += f"{list(element)[1]}&"
        for i, task in enumerate(task_list):
            df = self.get_target_df(row)
            if df.empty:
                return ""
            else:
                scores[task]["avg"] = df[(df["task"] == task)][col_to_write].mean()
                scores[task]["std"] = df[(df["task"] == task)][col_to_write].std()
                scores[task]["max"] = df[(df["task"] == task)][col_to_write].max()
        scores["all"] = np.mean([task_dict["avg"] for task_dict in scores.values()])
        scores = {k: v for k, v in scores.items()}
        for i, task in enumerate(col_list):
            if i < len(col_list) - 1:
                tex += f"{round(coefict * scores[task]['avg'], decimal)}\small$\pm${round(coefict * scores[task]['std'], decimal)}\\thinspace({round(coefict * scores[task]['max'], decimal)})&\n"
            else:
                tex += str(round(coefict * scores[task], decimal))
                tex += "\\\ \n"
        return tex

    def write_end(self, session_to_include: List, task_list: List, result: Result) -> str:
        metric_string = ""
        for i, col in enumerate(session_to_include + task_list):
            if i <= len(session_to_include) - 1:
                metric_string += "&"
        string = "\\hline\\hline\n" \
                 f"\\textbf{{Metric}}{metric_string}M-F1&M-F1&M-F1&F$^{{(i)}}$&M-F1&M-Rec&AVG(F$^{{(a)}}$, F$^{{(f)}}$)&TE\n" \
                 "\\end{tabular}\n" \
                 "\\end{center}}"
        return string


class LatexTableWriter(object):
    def __init__(self, output_path: str, result_class: type(Result), table: Optional[Table] = Table):
        self.output_path = output_path
        self.result_class = result_class
        test_result_path = self.fetch_test_results_from_dir(output_path)
        self.result_instances = self.retrieve_result_instance(test_result_path)
        self.task_list = list(set(result.task for result in self.result_instances))
        self.result_df = self.get_result_df(self.result_instances)
        self.table = table(self.result_df)
        self.write_to_csv(self.result_df, str(Path(self.output_path, "results.csv").absolute()))

    def get_model_list(self, path_list: list) -> list:
        return list(set([re.search(f"{self.output_path}(.*)/", path).group(1).split("/")[0] for path in path_list]))

    @staticmethod
    def write_to_csv(df: pd.DataFrame, path: str):
        df.to_csv(path)

    def fetch_test_results_from_dir(self, root: str):
        return [fname for fname in self.walk_through_files(root, "test_result.json")]

    def retrieve_result_instance(self, result_path: List[str]) -> List[Result]:
        return [self.result_class(file, root=self.output_path) for file in result_path]

    @staticmethod
    def walk_through_files(path, file_extension='.csv'):
        for (dirpath, dirnames, filenames) in os.walk(path):
            for filename in filenames:
                if filename.endswith(file_extension):
                    yield os.path.join(dirpath, filename)

    @staticmethod
    def get_result_df(result_instances: List[Result]) -> pd.DataFrame:
        metric_name = []
        for metric in [result.metric_name for result in result_instances]:
            if len(metric) == 1:
                if metric[0] == "other":
                    metric_name.append("f1_macro (averaging of favor and against class)")
                else:
                    metric_name.append(metric[0])
            elif len(metric) == 2:
                metric_name.append(f"{metric[0]}: {metric[1]}")
            else:
                raise NotImplementedError
        data = {'seed': [result.seed for result in result_instances],
                'task': [result.task for result in result_instances],
                'model': [result.model for result in result_instances],
                'name': [result.name for result in result_instances],
                'include_oos': [result.include_oos for result in result_instances],
                'batch_size': [result.batch_size for result in result_instances],
                'early_stopping_patience': [result.early_stopping_patience for result in result_instances],
                'dropout': [result.dropout for result in result_instances],
                'L2_normalize_encoded_feature': [result.l2_normalized_encoded_feature for result in result_instances],
                'epochs': [result.epochs for result in result_instances],
                'freeze_transformer_layers': [result.freeze_transformer_layers for result in result_instances],
                'head_type': [result.head_type for result in result_instances],
                'learning_rate': [result.learning_rate for result in result_instances],
                'contrastive': [result.contrastive for result in result_instances],
                'contrastive_loss_ratio': [result.contrastive_loss_ratio for result in result_instances],
                'contrastive_temperature': [result.contrastive_temperature for result in result_instances],
                'base_temperature': [result.base_temperature for result in result_instances],
                'contrast_mode': [result.contrast_mode for result in result_instances],
                'augmenter': [result.augmenter for result in result_instances],
                'num_augmented_samples': [result.num_augmented_samples for result in result_instances],
                'encoded_feature_silhouette': [result.encoded_feature_silhouette for result in result_instances],
                'final_output_silhouette': [result.final_output_silhouette for result in result_instances],
                'augmenter_dropout': [result.augmenter_dropout for result in result_instances],
                'metric_name': metric_name,
                'metric_score': [result.metric for result in result_instances]
                }
        return pd.DataFrame(data)

    def write_to_tex(self, name: str, session_to_include: List, col_to_write: Optional[str]="metric_score"):
        rows = self.get_rows(session_to_include)
        with open(str(Path(self.output_path, f"{name}_latex_table.tex").absolute()), "w") as f:
            f.write(self.table.write_top(session_to_include=session_to_include, task_list=self.task_list))
            for row in rows:
                f.write(self.table.write_row(row=row, task_list=self.task_list, col_to_write=col_to_write))
            f.write(self.table.write_end(session_to_include=session_to_include, task_list=self.task_list, result=self.result_instances[0]))

    def write_silhouette_to_tex(self, name: str, session_to_include: List):
        rows = self.get_rows(session_to_include)
        with open(str(Path(self.output_path, f"{name}_latex_table.tex").absolute()), "w") as f:
            f.write(self.table.write_top(session_to_include=session_to_include, task_list=self.task_list))
            for row in rows:
                f.write(self.table.write_row(row=row, task_list=self.task_list, col_to_write="encoded_feature_silhouette"))
            f.write(self.table.write_end(session_to_include=session_to_include, task_list=self.task_list, result=self.result_instances[0]))

    def get_rows(self, session_to_include: List):
        session_dict = {key: self.result_df[key].unique() for key in session_to_include}
        a = []
        for key, values in session_dict.items():
            session_list = []
            for value in values:
                session_list.append([key, value])
            a.append(session_list)
        rows = list(itertools.product(*a))
        return rows


class ConfigWriter(object):
    @staticmethod
    def write_from_dict(dict_to_dump: Dict, path_to_dump: str):
        with open(path_to_dump, 'w') as f:
            yaml.dump(dict_to_dump, f, default_flow_style=False)

    @staticmethod
    def read_yaml(yaml_to_read: str) -> Dict:
        with open(yaml_to_read, 'r') as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def change_field_of_all(dir: str):
        files = [os.path.join(path, name) for path, subdirs, files in os.walk(dir) for name in files if name.endswith(".yaml")]
        updated_dicts: List[Dict] = []
        for file in files:
            config = ConfigWriter.read_yaml(file)
            config["seed"] = [0, 1, 2]
            # config["model"]["layers"] = {"layer1": {"n_in": 768, "n_out": 768}, "layer2": {"n_in": 768, "n_out": 20}}
            # output = config["model"]["output_path"]
            # updated_output = output.replace("/contrastive_loss_ratio/", "/base_temp/")
            # config["model"]["output_path"] = updated_output
            config["early_stopping"]["tolerance"] = 5
            # config["model"]["epochs"] = 100
            # config["data"]["gradient_accu_step"] = 1
            config["data"]["batch_size"] = 32
            # config["model"]["output_path"] = "./outputs/tweeteval/experiments/cohort7/09/bertweet/"
            # config["model"]["from_pretrained"] = "roberta-base"
            config["model"]["L2_normalize_encoded_feature"] = True
            config["model"]["L2_normalize_logits"] = True
            # config["model"]["learning_rate"] = 1.0e-05
            # config["augmenter"]["name"] = "dropout"
            # config["augmenter"]["dropout"] = [0.1, 0.1]
            # config["augmenter"]["num_samples"] = 2
            config["model"]["contrastive"]["contrastive_loss_ratio"] = 0.95
            # config["model"]["contrastive"]["base_temperature"] = 0.3
            config["model"]["contrastive"]["temperature"] = 0.3
            # config["model"]["freeze_transformer_layers"] = "all"
            output = config["model"]["output_path"]
            # updated_output = output.replace("aug/05/01/roberta_base/", "best/095/03/roberta_base/")
            # config["model"]["output_path"] = updated_output
            # if updated_output.endswith("/"):
            #     updated_output = updated_output[:-1]
            # path_to_ckpt = f"{updated_output}/{config['name']}/seed_{config['seed'][0]}/pretrained_models/{config['name']}_best_model.pt"
            # config["model"]["load_ckpt"] = path_to_ckpt
            updated_dicts.append(config)
            ConfigWriter.write_from_dict(config, file)


if __name__ == "__main__":
    # ConfigWriter.change_field_of_all("sequence_classifier/configs/tweeteval/final/cohort10/best/temperature/03/")
    # ConfigWriter.change_field_of_all("sequence_classifier/configs/tweeteval/final/cohort7/")
    writer = LatexTableWriter("./tables/tweeteval/contrastive_learning_tweeteval/best/", TweetEvalResult, table=TweetEvalMainTable)
    # writer.write_to_tex(name="encoded_feature_silhouette", session_to_include=["model", "contrastive_loss_ratio"], col_to_write="encoded_feature_silhouette")
    # writer.write_to_tex(name="final_output_silhouette", session_to_include=["model", "contrastive_loss_ratio"], col_to_write="final_output_silhouette")
    # writer.write_to_tex(name="tweeteval", session_to_include=["model", "contrastive_loss_ratio" , "contrastive_temperature"])
    # writer.write_to_tex(name="tweeteval", session_to_include=["model", "contrastive_loss_ratio"])
    # writer = LatexTableWriter("./tables/tweeteval/paper/cohort2", TweetEvalResult, table=TweetEvalMainTable)
    writer.write_to_tex(name="tweeteval", session_to_include=["model"])
    # writer = LatexTableWriter("./tables/tweet_topic_single/experiments/", TweetTopicResult)
    # writer.write_to_tex(name="tweeteval", session_to_include=["model", "contrastive_loss_ratio"])

    # writer = LatexTableWriter("./tables/tweeteval/0311/", TweetEvalResult, table=TweetEvalMainTable)
    # writer.write_to_tex(name="tweeteval", session_to_include=["model", "augmenter_dropout"])
    # writer = LatexTableWriter("./tables/crisis/experiments/", CrisisResult)
    # writer.write_to_tex(name="crisis", session_to_include=["model", "contrastive", "head_type"])
    # writer = LatexTableWriter("./tables/sexism/", SexismResult)
    # writer.write_to_tex(name="sexism", session_to_include=["model", "contrastive"])
    # writer = LatexTableWriter("./tables/semeval18/100epochs/sl/", SemevalResult)
    # writer.write_to_tex(name="semeval18", session_to_include=["model"])
    # writer.write_to_tex(name="semeval18", session_to_include=["model", "contrastive_loss_ratio", "contrastive_temperature"])



