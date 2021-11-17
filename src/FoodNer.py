import torch
import numpy as np
import itertools
import copy
import json
import argparse
from transformers import BertForTokenClassification, BertTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm

import prepare_data_utils


class FoodNer:
    # this is a model from https://github.com/ds4food/FoodNer.git

    def __init__(self, path_to_pretrained_model, tokenizer_path="bert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
        self.model = BertForTokenClassification.from_pretrained(path_to_pretrained_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.tag_values = ['B-FOOD', 'I-FOOD', 'O', 'PAD']

    def predict(self, sentences, true_labels):
        """
        This method is mostly copied from the aforementioned repo. It is not a good code.
        :param sentences:
        :return:
        """

        ix = 0
        correct = copy.deepcopy(true_labels)
        predicted = []

        for s in tqdm(sentences, total=len(sentences)):
            test_sentence = " ".join(s)

            input_ids = self.tokenizer([test_sentence], return_tensors="pt").input_ids

            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                output = self.model(input_ids)

            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().numpy()[0])
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)

            pred = []
            for wi in range(0, len(true_labels[ix])):
                pred.append(new_labels[wi + 1])

            predicted.append(pred)
            ix += 1

        correct_flat = [item for sublist in correct for item in sublist]
        predicted_flat = [item for sublist in predicted for item in sublist]

        target_names = set([item for item in itertools.chain(correct_flat, predicted_flat)])

        with open("foodner" + '.log', 'w+') as log:
            log.write(classification_report(correct_flat, predicted_flat, target_names=target_names))

        return predicted


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--foodner-path', type=str, default="../res/foodner",
                        help='Path to FoodNer model')
    parser.add_argument('--annotations-folder', type=str, default="../data/annotations",
                        help='Path to folder with annotations')
    parser.add_argument('--recipes-folder', type=str, default="../data/annotations",
                        help='Path to folder with recipes')

    args = parser.parse_args()

    foodner = FoodNer(args.foodner_path)

    # FoodNer considers only food entities
    ENTITIES_MAP = {
        "food_product_with_unit": "FOOD",
        "food_product_without_unit_uncountable": "FOOD",
        "food_product_whole": "FOOD",
        "food_product_without_unit_countable": "FOOD",
        "quantity": "O",
        "unit": "O",
        "physical_quality": "O",
        "color": "O",
        "trade_name": "O",
        "purpose": "O",
        "taste": "O",
        "process": "O",
        "example": "O",
        "part": "O"
    }

    recipes, entities = prepare_data_utils.collect_annotations(
        annotations_folder=args.annotations_folder, recipes_folder=args.recipes_folder,
        scheme_func=prepare_data_utils.bio_scheme,
        map_entity_func=prepare_data_utils.map_entity,
        entities_map=ENTITIES_MAP)

    # see foodner.log
    entities_predicted = foodner.predict(recipes, entities)

    recipes_with_predictions = {
        i: {"recipe": " ".join(recipe),
            "entities": [recipe[token_id] for token_id in range(len(recipe)) if entities_predicted[i][token_id] != "O"]}
        for i, recipe in enumerate(recipes)}

    with open("predicted_entities.json", "w") as json_file:
        json.dump(recipes_with_predictions, json_file, indent=1)
