#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import spacy
from typing import List, Tuple
import prepare_data_utils
from model import NERTaisti
from spacy.training import biluo_tags_to_offsets, iob_to_biluo


model = NERTaisti(config="../res/ner_model/config.json")


def predict_text(text: str) -> Tuple[spacy.tokens.Doc, List[str]]:
    doc = prepare_data_utils.prepare_text(text)
    tokens = [token.text for token in doc]
    predictions = model.predict([tokens])[0]
    return doc, predictions


def main(output_path: str) -> None:
    all_queries = []
    while True:
        text = input("-" * 30 + "\nPlease type your text to process: ")
        doc, tags = predict_text(text)
        tags = iob_to_biluo(tags)
        entities = biluo_tags_to_offsets(doc, tags)
        all_queries.append({
            "text": text,
            "tokens": [{"text": t.text, "start": t.idx,
                        "end": t.idx + len(t)} for t in doc],
            "entities_list": [{"start": x[0],
                            "end": x[1],
                            "label": x[2],
                            "text": text[x[0]:x[1]]} for x in entities]
        })

        print("Found the following entities: ")
        for entity in entities:
            print(f"\t`{text[entity[0]:entity[1]]}` tagged as `{entity[2]}`")
        decision = input("Process next? Press N to exit the app or press anything else to continue. ")

        if decision.lower() == 'n' or decision.lower() == 'no':
            with open(output_path, 'w') as f:
                f.write(json.dumps(all_queries, indent=4))
                print(f"The NER output saved to {output_path}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', '--output_file_path',
                    help='Path to generated output JSON file',
                    type=str,
                    default='./output.json')

    args = parser.parse_args()
    main(args.output_file_path)
