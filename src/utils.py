import re
import numpy as np
from datasets import load_metric
import spacy
from spacy.training import biluo_tags_to_offsets
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

DISPLACY_OPTIONS = {
    "ents": ["FOOD", "UNIT", "QUANTITY", "PROCESS", "COLOR",
             "PHYSICAL_QUALITY"],
    "colors": {
        "FOOD": "#dbe9f6",
        "UNIT": "#bad6eb",
        "QUANTITY": "#89bedc",
        "PROCESS": "#539ecd",
        "COLOR": "#2b7bba",
        "PHYSICAL_QUALITY": "#0b559f"
    }
}


def check_if_entity_correctly_began(entity, prev_entity):
    """
    This function checks if "I-" entity is preceded with "B-" or "I-". For
    example, "I-FOOD" should not happen after "O" or after "B-QUANT".
    :param entity:
    :param prev_entity:
    :return: bool
    """
    if "I-" in entity and re.sub(r"[BI]-", "", entity) != \
            re.sub(r"[BI]-", "", prev_entity):
        return False
    return True


def token_to_entity_predictions(text_split_words, text_split_tokens,
                                token_labels, id2label):
    """
    Transform token (subword) predictions into word predictions.
    :param text_split_words: list of words from one recipe, eg. ["I", "eat",
    "chicken"] (the ones that go to tokenizer)
    :param text_split_tokens: list of tokens from one recipe, eg. ["I", "eat",
    "chic", "##ken"] (the ones that arise
    from input decoding)
    :param token_labels: list of labels associated with each token from
    text_split_tokens
    :param id2label: a mapping from ids (0, 1, ...) to labels ("B-FOOD",
    "I-FOOD", ...)
    :return: a list of entities associated with each word from text_split_words,
    ie. entities extracted from a recipe
    """

    word_idx = 0
    word_entities = []
    word_from_tokens = ""
    word_entity = ""
    prev_word_entity = ""

    for token_label, token in zip(token_labels, text_split_tokens):
        if token in ["[SEP]", "[CLS]"]:
            continue
        word_from_tokens += re.sub(r"^##", "", token)
        # take the entity associated with the first token (subword)
        word_entity = id2label[token_label] if word_entity == "" \
            else word_entity

        if word_from_tokens == text_split_words[word_idx] or\
                word_from_tokens == "[UNK]":
            word_idx += 1
            # replace entities containing "I-" that do not have a predecessor
            # with "B-"
            # TODO: perhaps it should be replaced with the next most probable
            #  entity, not with "O". Especially there are cases such as true:
            #  B-FOOD, I-FOOD, I-FOOD, and pred: B-FOOD, O, I-FOOD, for
            #  [confectioner, 's, sugar]
            word_entity = "O" if not \
                check_if_entity_correctly_began(word_entity, prev_word_entity) \
                else word_entity
            word_entities.append(word_entity)
            word_from_tokens = ""
            prev_word_entity = word_entity
            word_entity = ""

    return word_entities


# taken from: https://huggingface.co/docs/transformers/custom_datasets#token-classification-with-wnut-emerging-entities
def tokenize_and_align_labels(recipes, entities, tokenizer, max_length,
                              label2id, only_first_token=True):
    """
    Prepare recipes with/without entities for TaistiNER.
    :param recipes: list of lists of words from a recipe
    :param entities: list of lists of entities from a recipe
    :param tokenizer: tokenizer
    :param max_length: maximal tokenization length
    :param label2id: a mapping from labels ("B-FOOD", "I-FOOD", ...) to ids
    (0, 1, ...)
    :param only_first_token: whether to label only first subword of a word,
    eg. Suppose "chicken" is split into "chic", "##ken". Then if True, it will
    have [1, -100], if False [1, 1]. -100
    is omitted in Pytorch loss function
    :return: a dictionary with tokenized recipes with/without associated token
    labels
    """
    tokenized_data = tokenizer(recipes, truncation=True, max_length=max_length,
                               is_split_into_words=True)

    if entities:
        labels = []
        for i, entity in enumerate(entities):
            # Map tokens to their respective word.
            word_ids = tokenized_data.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    new_label = -100
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    new_label = label2id[entity[word_idx]]
                else:
                    new_label = -100 if only_first_token \
                        else label2id[entity[word_idx]]
                label_ids.append(new_label)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_data["labels"] = labels
    tokenized_data["recipes"] = recipes

    return tokenized_data


metric = load_metric("seqeval")


# TODO: these metrics are per token, not per entity, which would be preferable
#  instead this would require overriding huggigface's methods and classes
def compute_metrics(p):
    """
    Compute seqeval metrics for entities: precision, recall, f1-score, accuracy.
    These are used only during training.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions,
                             references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def bio_to_biluo(bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD", "U-PROCESS"]
    """
    biluo_entities = []

    for entity_idx in range(len(bio_entities)):
        cur_entity = bio_entities[entity_idx]
        next_entity = bio_entities[entity_idx + 1] if \
            entity_idx < len(bio_entities) - 1 else ""

        if cur_entity.startswith("B-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("B-", "U-", cur_entity))
        elif cur_entity.startswith("I-"):
            if next_entity.startswith("I-"):
                biluo_entities.append(cur_entity)
            else:
                biluo_entities.append(re.sub("I-", "L-", cur_entity))
        else:  # O
            biluo_entities.append(cur_entity)

    return biluo_entities


def biluo_to_span(recipe, biluo_entities):
    """
    :param biluo_entities: list of BILUO entities, eg. ["O", "B-FOOD", "L-FOOD",
    "U-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    doc = nlp(recipe)
    spans = biluo_tags_to_offsets(doc, biluo_entities)
    return spans


def bio_to_span(recipe, bio_entities):
    """
    :param bio_entities: list of BIO entities, eg. ["O", "B-FOOD", "I-FOOD",
    "B-PROCESS"]
    :return: list of span entities, eg. [(span_start, span_end, "FOOD"),
    (span_start, span_end, "PROCESS")]
    """
    biluo_entities = bio_to_biluo(bio_entities)
    spans = biluo_to_span(recipe, biluo_entities)
    return spans


def visualize_prediction(recipes, entities, in_jupyter=False,
                         options=DISPLACY_OPTIONS):
    """
    Visualize entities within a text. This function by default (
    in_jupyter=False) displays visualizations locally on port 5000.
    :param recipes: list of recipes or a single recipe
    :param entities: list of entities associated with recipe(s). Can be BIO,
    BILUO or spans
    :param in_jupyter: if True visualization can be displayed in Jupyter cell
    :param options: options for visualization. These include setting
    available entities and colors for visualizing them.
    """
    if isinstance(recipes, str):
        recipes = [recipes]
        entities = [entities]

    if isinstance(entities[0][0], str):
        if any([bool(re.match(r"[UL]-", entity)) for entity in
                set(entities[0])]):  # entities are in BILUO schema
            spans = [biluo_to_span(recipe, ents) for recipe, ents in
                     zip(recipes, entities)]
        else:  # entities are in BIO schema
            spans = [bio_to_span(recipe, ents) for recipe, ents in
                     zip(recipes, entities)]
    else:  # spans
        spans = entities

    docs = []
    for recipe, recipe_spans in zip(recipes, spans):
        doc = nlp(recipe)
        ents = []
        for span_start, span_end, label in recipe_spans:
            ent = doc.char_span(span_start, span_end, label=label)
            ents.append(ent)

        doc.ents = ents
        docs.append(doc)

    if in_jupyter:
        displacy.render(docs, style="ent", jupyter=True, options=options)
    else:
        displacy.serve(docs, style="ent", options=options)
