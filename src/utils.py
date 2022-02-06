import re
import numpy as np
from datasets import load_metric


def check_if_entity_correctly_began(entity, prev_entity):
    """
    This function checks if "I-" entity is preceded with "B-" or "I-". For example, "I-FOOD" should not happen after "O"
    or after "B-QUANT".
    :param entity:
    :param prev_entity:
    :return: bool
    """
    if "I-" in entity and re.sub(r"[BI]-", "", entity) != re.sub(r"[BI]-", "", prev_entity):
        return False
    return True


def token_to_entity_predictions(text_split_words, text_split_tokens, token_labels, id2label):
    """
    Transform token (subword) predictions into word predictions.
    :param text_split_words: list of words from one recipe, eg. ["I", "eat", "chicken"] (the ones that go to tokenizer)
    :param text_split_tokens: list of tokens from one recipe, eg. ["I", "eat", "chic", "##ken"] (the ones that arise
    from input decoding)
    :param token_labels: list of labels associated with each token from text_split_tokens
    :param id2label: a mapping from ids (0, 1, ...) to labels ("B-FOOD", "I-FOOD", ...)
    :return: a list of entities associated with each word from text_split_words, ie. entities extracted from a recipe
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
        word_entity = id2label[token_label] if word_entity == "" else word_entity

        if word_from_tokens == text_split_words[word_idx] or word_from_tokens == "[UNK]":
            word_idx += 1
            # replace entities containing "I-" that do not have a predecessor with "B-"
            # TODO: perhaps it should be replaced with the next most probable entity, not with "O". Especially there are
            #  cases such as true: B-FOOD, I-FOOD, I-FOOD, and pred: B-FOOD, O, I-FOOD, for [confectioner, 's, sugar]
            word_entity = "O" if not check_if_entity_correctly_began(word_entity, prev_word_entity) else word_entity
            word_entities.append(word_entity)
            word_from_tokens = ""
            prev_word_entity = word_entity
            word_entity = ""

    return word_entities


def tokenize_and_align_labels(recipes, entities, tokenizer, max_length, label2id, only_first_token=True):
    """
    Prepare recipes with/without entities for TaistiNER.
    :param recipes: list of lists of words from a recipe
    :param entities: list of lists of entities from a recipe
    :param tokenizer: tokenizer
    :param max_length: maximal tokenization length
    :param label2id: a mapping from labels ("B-FOOD", "I-FOOD", ...) to ids (0, 1, ...)
    :param only_first_token: whether to label only first subword of a word, eg. Suppose "chicken" is split into
    "chic", "##ken". Then if True, it will have [1, -100], if False [1, 1]. -100 is omitted in Pytorch
    loss function
    :return: a dictionary with tokenized recipes with/without associated token labels
    """
    tokenized_data = tokenizer(recipes, truncation=True, max_length=max_length, is_split_into_words=True)

    if entities:
        labels = []
        for i, entity in enumerate(entities):
            word_ids = tokenized_data.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100. -100 is omitted in PyTorch loss function.
                if word_idx is None:
                    new_label = -100
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    new_label = label2id[entity[word_idx]]
                else:
                    new_label = -100 if only_first_token else label2id[entity[word_idx]]
                label_ids.append(new_label)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_data["labels"] = labels
    tokenized_data["recipes"] = recipes

    return tokenized_data


metric = load_metric("seqeval")


# TODO: these metrics are per token, not per entity, which would be preferable instead this would require overriding
#  huggigface's methods and classes
def compute_metrics(p):
    """
    Compute seqeval metrics for entities: precision, recall, f1-score, accuracy. These are used only during training.
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

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
