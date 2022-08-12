#!/usr/bin/env python
# coding: utf-8

import prepare_data_utils
from model import NERTaisti

model = NERTaisti(config="../res/ner_model/config.json")
    
def predict_text(text):
    tokens = prepare_data_utils.prepare_text(text)
    predictions = model.predict([tokens])
    for token, label in zip(tokens, predictions[0]):
        print(f"{token} {label}")

while True:
    predict_text(input("Please type your text:"))
