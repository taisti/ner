import os
import torch
import json

from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
from utils import tokenize_and_align_labels, compute_metrics, token_to_entity_predictions
from datasets import Dataset
from nervaluate import Evaluator


class NERTaisti:
    """
    It uses Huggingface's Trainer API. For more details see:
    https://huggingface.co/docs/transformers/v4.16.2/en/custom_datasets#token-classification-with-wnut-emerging-entities
    """
    def __init__(self, config):

        if isinstance(config, str):
            with open(config, "r") as json_file:
                self.config = json.load(json_file)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"{config} is not a config!")

        if self.config["model_pretrained_path"] and os.path.exists(self.config["model_pretrained_path"]):
            model_name_or_path = self.config["model_pretrained_path"]
            print(f"Loaded pretrained model from {os.path.abspath(model_name_or_path)}!!!")
        else:
            model_name_or_path = self.config["_name_or_path"]  # huggingface's variable name for model type
            print(f"Loaded huggingface checkpoint: {model_name_or_path}")

        possible_tokenizer_path = os.path.join(self.config["model_pretrained_path"], "tokenizer_config.json")
        if self.config["model_pretrained_path"] and os.path.exists(possible_tokenizer_path):
            tokenizer_name_or_path = possible_tokenizer_path
            print(f"Loaded tokenizer from {os.path.abspath(tokenizer_name_or_path)}!!!")
        else:
            tokenizer_name_or_path = self.config["_name_or_path"]
            print(f"Loaded huggingface tokenizer: {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        label2id = {k: int(v) for k, v in self.config["label2id"].items()}
        id2label = {v: k for k, v in label2id.items()}
        torch.manual_seed(self.config["training_args"]["seed"])
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, num_labels=len(self.config["label2id"]), ignore_mismatched_sizes=True,
            label2id=label2id, id2label=id2label
        )

        training_args = TrainingArguments(
            **self.config["training_args"]
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, max_length=self.config["max_length"], padding="max_length"
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

    def train(self, train_recipes, train_entities, val_recipes, val_entities):

        _, train_dataset = self.prepare_data(train_recipes, train_entities)
        _, val_dataset = self.prepare_data(val_recipes, val_entities)

        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = val_dataset

        self.trainer.train()

        self.save_model()

    def evaluate(self, recipes, entities):

        pred_entities = self.predict(recipes)

        entities_types = list(set(list(self.trainer.model.config.label2id.keys())))
        entities_types.remove("O")

        evaluator = Evaluator(
            entities, pred_entities, tags=entities_types, loader="list"
        )

        evaluation_all, evaluation_by_tag = evaluator.evaluate()

        results = {"all": {}}

        for k, v in evaluation_all["strict"].items():
            results["all"][k] = v

        for entity in entities_types:
            results[entity] = {}
            for k, v in evaluation_by_tag[entity]["strict"].items():
                results[entity][k] = v

        return results

    def predict(self, recipes):

        data, dataset = self.prepare_data(recipes, [])
        preds = self.trainer.predict(dataset)

        token_probs = preds[0]
        token_labels = token_probs.argmax(axis=2)

        pred_entities = []

        num_of_recipes = dataset.num_rows

        for recipe_idx in range(num_of_recipes):
            text_split_words = recipes[recipe_idx]
            text_split_tokens = self.tokenizer.convert_ids_to_tokens(data["input_ids"][recipe_idx])

            pred_entities.append(token_to_entity_predictions(
                text_split_words, text_split_tokens, token_labels[recipe_idx], self.config['id2label']
            ))

        return pred_entities

    def prepare_data(self, recipes, entities):
        data = tokenize_and_align_labels(
            recipes=recipes, entities=entities, tokenizer=self.tokenizer,
            label2id=self.trainer.model.config["label2id"],
            max_length=self.config["max_length"],
            only_first_token=self.config["only_first_token"]
        )

        dataset = Dataset.from_dict(data)

        return data, dataset

    def save_model(self):
        save_dir = self.config["save_dir"] if self.config["save_dir"] else "taisti_ner_model"

        print(f"Model with configs saved in {os.path.abspath(save_dir)}!!!")

        with open(os.path.join(save_dir, "training_args.json"), "w") as json_file:
            json.dump(self.trainer.args, json_file, indent=4)

        self.trainer.save_model(save_dir)
