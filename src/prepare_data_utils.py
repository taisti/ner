import spacy
import os
import string
from brat_parser import get_entities_relations_attributes_groups
from spacy.training import offsets_to_biluo_tags
from tqdm import tqdm

NO_ENTITY_TOKEN = "O"
nlp = spacy.load('en_core_web_sm')


def process_ann_file(path_to_ann_file):
    entities, relations, _, _ = get_entities_relations_attributes_groups(path_to_ann_file)
    return entities, relations


def bio_scheme(entity_token):
    return entity_token.replace("L-", "I-").replace("U-", "B-")


def biluo_scheme(entity_token):
    return entity_token


def map_entity(entity_token, entities_map_dict):
    position, entity = entity_token.split("-")
    entity = entities_map_dict[entity]
    if entity == NO_ENTITY_TOKEN:
        return entity
    else:
        return position + "-" + entity


def dont_map_entity(entity, entities_map_dict):
    return entity


def prepare_token_text(token):
    return token.replace("\n", ".")


def prepare_token_entity(entity_token, scheme_func, entity_map_func, entities_map_dict):
    if entity_token == NO_ENTITY_TOKEN:
        return entity_token
    else:
        entity_token = scheme_func(entity_token)
        entity_token = entity_map_func(entity_token, entities_map_dict)
        return entity_token


def correct_span(span_start, span_end, text):
    while span_start > 0 and text[span_start-1].strip() != "":
        # TODO: this is to avoid problems with cases such as 13kg, quantity + unit, but it is not a nice solution
        if not text[span_start].isnumeric() and text[span_start - 1].isnumeric():
            break
        span_start -= 1

    if text[span_start] in string.punctuation:
        span_start += 1

    while span_end < len(text) - 1 and text[span_end].strip() not in [""]:
        # TODO: this is to avoid problems with cases such as 13kg, quantity + unit, but it is not a nice solution
        if text[span_end - 1].isnumeric() and not text[span_end].isnumeric():
            break
        span_end += 1

    if text[span_end] in string.punctuation:
        span_end -= 1

    return span_start, span_end


def process_single_annotation(ann_path, recipe_path, scheme_func, map_entity_func, entities_map):
    entities_from_ann_file, relations_from_ann_file = process_ann_file(ann_path)
    with open(recipe_path, "r") as f:
        recipe = f.read()

    entities_with_spans = []
    for entity_id in entities_from_ann_file.keys():
        for start_of_span, end_of_span in entities_from_ann_file[entity_id].span:
            start_span, end_span = correct_span(start_of_span, end_of_span, recipe)
            if start_of_span != start_span:
                print(start_span, start_of_span, recipe[start_of_span:end_of_span], end_of_span, end_span)
            entities_with_spans.append((start_span, end_span, entities_from_ann_file[entity_id].type))

    doc = nlp.make_doc(recipe)

    biluo_entities = offsets_to_biluo_tags(doc, entities_with_spans)

    recipe_entities = [prepare_token_entity(entity_token, scheme_func, map_entity_func, entities_map)
                       for entity_token in biluo_entities]
    recipe_tokens = [prepare_token_text(token.text) for token in doc]

    assert len(recipe_tokens) == len(recipe_entities)

    return recipe_tokens, recipe_entities


def collect_annotations(annotations_folder, recipes_folder, scheme_func, map_entity_func, entities_map,
                        print_warnings=False):
    """

    :param annotations_folder: (str) path to folder with annotations
    :param recipes_folder: (str) path to folder with recipes
    :param scheme_func: (func) choose if you want BIO (bio_scheme) or BILUO (biluo_scheme)
    :param map_entity_func: (func) choose if you want to map entites in according to entities_map
    :param entities_map: (dict) a dictionary for entity mappins
    :param print_warnings: (bool) choose if to display annotation files that did not load (currently, mostly because of
    entites overlap)
    :return: all_recipes: (list) a list with recipes splitted to tokens
    :return: all_entities: (list) a list with entities for each token from all_recipes
    """
    ann_files = [file for file in os.listdir(annotations_folder) if ".ann" in file]

    all_recipes = []
    all_entities = []

    overlap_files = []

    print("Loading annotation files")
    for ann_file in tqdm(ann_files, total=len(ann_files)):
        ann_path = os.path.join(annotations_folder, ann_file)
        recipe_path = os.path.join(recipes_folder, ann_file.replace(".ann", ".txt"))

        try:
            recipe_tokens, recipe_entities = process_single_annotation(ann_path, recipe_path, scheme_func,
                                                                       map_entity_func, entities_map)
            all_recipes.append(recipe_tokens)
            all_entities.append(recipe_entities)

        except ValueError as e:
            if print_warnings:
                print(e)
                print(ann_file)
            overlap_files.append(ann_file)

    return all_recipes, all_entities
