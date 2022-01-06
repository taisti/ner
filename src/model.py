import torch

from transformers import (AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
from mappings import ENTITY_TO_LABEL, ENTITIES_MAP
from utils import tokenize_and_align_labels, compute_metrics, token_to_entity_predictions
from datasets import Dataset
from nervaluate import Evaluator


class NERTaisti:
    def __init__(self, config):

        self.config = config
        self.bert_type = self.config["bert_type"]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        if not self.config.get("model_pretrained_path", ""):
            model_pretrained_path = self.bert_type
        else:
            model_pretrained_path = self.config["model_pretrained_path"]

        torch.manual_seed(self.config["training_args"]["seed"])
        model = AutoModelForTokenClassification.from_pretrained(
            model_pretrained_path, num_labels=len(ENTITY_TO_LABEL)
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

    def evaluate(self, recipes, entities):

        pred_entities = self.predict(recipes)

        entities_types = list(set(list(ENTITIES_MAP.values())))
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
                text_split_words, text_split_tokens, token_labels[recipe_idx]
            ))

        return pred_entities

    def prepare_data(self, recipes, entities):
        data = tokenize_and_align_labels(
            recipes=recipes, entities=entities, tokenizer=self.tokenizer, max_length=self.config["max_length"],
            only_first_token=self.config["only_first_token"]
        )

        dataset = Dataset.from_dict(data)

        return data, dataset
