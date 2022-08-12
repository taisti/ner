#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re

import prepare_data_utils
from model import NERTaisti


ALL_ENTITIES_HIERARCHY = [
    "quantity",
    "unit",
    "food_product_with_unit",
    "food_product_without_unit_countable",
    "food_product_without_unit_uncountable",
    "food_product_whole",
    "process",
    "physical_quality",
    "color",
    "trade_name",
    "example",
    "taste",
    "purpose",
    "diet",
    "part",
    "possible_substitute",
    "excluded",
    "exclusive"
]


ENTITIES_MAP = {
    entity: "O" if "food" not in entity else "FOOD"
    for entity in ALL_ENTITIES_HIERARCHY
}

for entity in ["quantity", "unit", "color", "physical_quality", "process"]:
    ENTITIES_MAP[entity] = entity.upper()

# TODO: speed-up the process with multiprocessing
choose_span_func = prepare_data_utils.choose_food_span
entities_map = ENTITIES_MAP
entity_hierarchy = ALL_ENTITIES_HIERARCHY

train_indices = list(range(240)) + list(range(300, 400)) + list(range(500, 600))
val_indices = list(range(240, 300))
missing_indices = [417, 443]
test_indices = [idx for idx in range(400, 500) if idx not in missing_indices]

train_recipe_paths = [f"annotations/{idx}.txt" for idx in train_indices]
train_ann_paths = [f"annotations/{idx}.ann" for idx in train_indices]
val_recipe_paths = [f"annotations/{idx}.txt" for idx in val_indices]
val_ann_paths = [f"annotations/{idx}.ann" for idx in val_indices]
test_recipe_paths = [f"annotations/{idx}.txt" for idx in test_indices]
test_ann_paths = [f"annotations/{idx}.ann" for idx in test_indices]

train_recipes, train_entities = prepare_data_utils.collect_recipes_with_annotations(
    annotations_paths=train_ann_paths, recipes_paths=train_recipe_paths,
    scheme_func=prepare_data_utils.bio_scheme,
    map_entity_func=prepare_data_utils.map_entity,
    entities_map=entities_map,
    choose_span_func=choose_span_func,
    entity_hierarchy=entity_hierarchy
)

val_recipes, val_entities = prepare_data_utils.collect_recipes_with_annotations(
    annotations_paths=val_ann_paths, recipes_paths=val_recipe_paths,
    scheme_func=prepare_data_utils.bio_scheme,
    map_entity_func=prepare_data_utils.map_entity,
    entities_map=entities_map,
    choose_span_func=choose_span_func,
    entity_hierarchy=entity_hierarchy
)


test_recipes, test_entities = prepare_data_utils.collect_recipes_with_annotations(
    annotations_paths=test_ann_paths, recipes_paths=test_recipe_paths,
    scheme_func=prepare_data_utils.bio_scheme,
    map_entity_func=prepare_data_utils.map_entity,
    entities_map=entities_map,
    choose_span_func=choose_span_func,
    entity_hierarchy=entity_hierarchy
)




def train():
    # cross-validation was used hence we can train on everything
    recipes = train_recipes + val_recipes + test_recipes
    entities = train_entities + val_entities + test_entities


    label2id = {"O": 0}
    idx = 1

    for entity in set(list(entities_map.values())):
        if entity == "O":
            continue
        label2id[f"B-{entity}"] = idx
        idx += 1
        label2id[f"I-{entity}"] = idx
        idx += 1

    label2id = {k: v for k, v in sorted(label2id.items(), key=lambda item: item[1])}

    CONFIG = {
        "_name_or_path": "bert-base-cased",
        "model_pretrained_path": "",
        "save_dir": "../res/ner_model",  # or any choice
        "num_of_tokens": 128,
        "only_first_token": True,

        # for more details see https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#transformers.TrainingArguments
        "training_args": {
            "output_dir": '../checkpoints',
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "num_train_epochs": 10,
            "weight_decay": 0.01,
            "load_best_model_at_end": True,
            "seed": 62
        },

        "label2id" : label2id
    }

    model = NERTaisti(config=CONFIG)


    model.train(train_recipes, train_entities, val_recipes, val_entities)


def predict_dataset_sample():
    model = NERTaisti(config="../res/ner_model/config.json")
    recipe_paths = [f"annotations/{idx}.txt" for idx in range(240, 300)]

    recipes = prepare_data_utils.collect_recipes_without_annotations(
        recipes=recipe_paths
    )

    pred_entities = model.predict(recipes)
    for recipe, predictions in zip(recipes, pred_entities):
        for token, label in zip(recipe, predictions):
            print(f"{token} {label}")

def predict_text(text):
    model = NERTaisti(config="../res/ner_model/config.json")
    tokens = prepare_data_utils.prepare_text(text)
    predictions = model.predict([tokens])
    for token, label in zip(tokens, predictions[0]):
        print(f"{token} {label}")

train()