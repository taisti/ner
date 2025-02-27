import spacy
import os
import re
import copy
import string
import warnings
import random as rd
from brat_parser import get_entities_relations_attributes_groups
from spacy.training import offsets_to_biluo_tags
from tqdm import tqdm


NO_ENTITY_TOKEN = "O"
INCOMPLETE_ENTITY_TOKEN = "-"
nlp = spacy.load('en_core_web_sm')


def process_ann_file(path_to_ann_file):
    entities, relations, _, _ = get_entities_relations_attributes_groups(
        path_to_ann_file)
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


def dont_map_entity(entity_token, entities_map_dict):
    return entity_token


def prepare_token_text(token):
    return token.replace("\n", ".")


def prepare_token_entity(entity_token, scheme_func, entity_map_func,
                         entities_map_dict):
    if entity_token == NO_ENTITY_TOKEN:
        return entity_token
    else:
        entity_token = scheme_func(entity_token)
        entity_token = entity_map_func(entity_token, entities_map_dict)
        return entity_token


def correct_span(span_start, span_end, prev_span_end, next_span_start, recipe):
    """
    Corrects incomplete entities from ann files. For example:
    cacao powde -> cacao powder
    """

    # TODO: /colored instead of color this is the only one problematic
    #  example and should be held properly
    if recipe[span_start:span_end] == "colored" and \
            recipe[span_start-1:span_end] == "/colored":
        return span_start-1, span_end

    while span_start > prev_span_end and recipe[span_start-1].strip() != "":
        span_start -= 1

    if recipe[span_start] in string.punctuation:
        span_start += 1

    while span_end < next_span_start and recipe[span_end].strip() != "":
        span_end += 1

    if recipe[span_end - 1] in ",.":
        span_end -= 1

    return span_start, span_end


def get_first_broken_span(biluo_entities):
    """
    Locate first incomplete entity
    """
    broken_span_idx = 0
    for idx in range(len(biluo_entities)):
        if "B-" in biluo_entities[idx] or "U-" in biluo_entities[idx]:
            broken_span_idx += 1
        elif biluo_entities[idx] == "-":
            break

    return broken_span_idx


def choose_food_span(span1, span2, hierarchy):
    """
    With two overlapping entities, choose the one which is a food entity.
    """
    if "food" in span1[2] and "food" not in span2[2]:
        return 0
    elif "food" not in span1[2] and "food" in span2[2]:
        return 1
    else:
        return rd.choice([0, 1])


def choose_span_according_to_hierarchy(span1, span2, hierarchy):
    """
    With two overlapping entities, choose the one which is higher in the
    hierarchy (order of elements in hierarchy, the first being the most
    important).
    """

    hierarchy_values = {entity: i for i, entity in enumerate(hierarchy)}

    if hierarchy_values[span1[2]] <= hierarchy_values[span2[2]]:
        return 0
    else:
        return 1


def remove_overlap_entities(entities_with_spans, choose_span_func, ann_path,
                            hierarchy):
    """
    Remove overlapping entities in according to choose_span_func.
    """
    entities_with_spans = copy.deepcopy(entities_with_spans)
    span_idx = 0
    while span_idx < len(entities_with_spans) - 1:
        if entities_with_spans[span_idx][1] > \
                entities_with_spans[span_idx + 1][0]:  # found overlapping

            print("=" * 50)
            print(f"There are some overlapping entities in {ann_path}")

            keep_span = choose_span_func(
                entities_with_spans[span_idx],
                entities_with_spans[span_idx + 1],
                hierarchy
            )

            if keep_span == 0:
                print(f"Discarded: {entities_with_spans[span_idx + 1]}")
                print(f"Chosen: {entities_with_spans[span_idx]}")
                entities_with_spans.pop(span_idx + 1)
            else:
                print(f"Discarded: {entities_with_spans[span_idx]}")
                print(f"Chosen: {entities_with_spans[span_idx + 1]}")
                entities_with_spans.pop(span_idx)

        else:
            span_idx += 1

    return entities_with_spans


def correct_incomplete_entities(entities_with_spans, biluo_entities, recipe,
                                doc):
    """
    Correct incomplete entities (INCOMPLETE_ENTITY_TOKEN in biluo_entities).
    """
    while INCOMPLETE_ENTITY_TOKEN in biluo_entities:
        first_broken_span_idx = get_first_broken_span(biluo_entities)
        broken_span = entities_with_spans[first_broken_span_idx]
        print(f"Incomplete span: {broken_span}",
              recipe[broken_span[0]:broken_span[1]])

        prev_span_end = entities_with_spans[first_broken_span_idx - 1][1]\
            if first_broken_span_idx > 0 else 0
        next_span_start = entities_with_spans[first_broken_span_idx + 1][0] \
            if first_broken_span_idx < len(entities_with_spans) - 1\
            else len(recipe)

        corrected_span_start, corrected_span_end = correct_span(
            broken_span[0],
            broken_span[1],
            prev_span_end,
            next_span_start,
            recipe
        )

        entities_with_spans[first_broken_span_idx] = (
            corrected_span_start, corrected_span_end, broken_span[2]
        )

        print(f"Corrected span: {entities_with_spans[first_broken_span_idx]}",
              recipe[corrected_span_start:corrected_span_end])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            biluo_entities = offsets_to_biluo_tags(doc, entities_with_spans)

    return biluo_entities, entities_with_spans


def process_single_annotation(ann_path, recipe_path, scheme_func,
                              map_entity_func, entities_map, choose_span_func,
                              entity_hierarchy):
    entities_from_ann_file, relations_from_ann_file = process_ann_file(ann_path)
    with open(recipe_path, "r") as f:
        recipe = f.read()

    entities_with_spans = []
    for entity_id in entities_from_ann_file.keys():
        for start_of_span, end_of_span in entities_from_ann_file[entity_id].span:
            entities_with_spans.append((
                start_of_span, end_of_span,
                entities_from_ann_file[entity_id].type
            ))

    entities_with_spans = sorted(entities_with_spans, key=lambda span: span[0])
    entities_with_spans = remove_overlap_entities(
        entities_with_spans,
        choose_span_func,
        ann_path,
        entity_hierarchy
    )

    doc = nlp.make_doc(recipe)
    # ignore warning for incomplete entities, as this issue is already handled
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        biluo_entities = offsets_to_biluo_tags(doc, entities_with_spans)

    if INCOMPLETE_ENTITY_TOKEN in biluo_entities:
        print("=" * len(f"There are some incomplete entities in {ann_path}"))
        print(f"There are some incomplete entities in {ann_path}")

        biluo_entities, entities_with_spans = correct_incomplete_entities(
            entities_with_spans, biluo_entities, recipe, doc)

    recipe_entities = [prepare_token_entity(entity_token, scheme_func,
                                            map_entity_func, entities_map)
                       for entity_token in biluo_entities]
    recipe_tokens = [prepare_token_text(token.text) for token in doc]

    assert len(recipe_tokens) == len(recipe_entities)

    return recipe_tokens, recipe_entities


def collect_recipes_with_annotations(annotations_paths, recipes_paths,
                                     scheme_func, map_entity_func, entities_map,
                                     choose_span_func, **kwargs):
    """
    :param annotations_paths: a path to folder with annotations, or a list of
    paths to annotation files
    :param recipes_paths: a path to folder with recipes, or a list of paths to
    recipe files
    :param scheme_func: (func) choose if you want BIO (bio_scheme) or BILUO
    (biluo_scheme)
    :param map_entity_func: (func) choose if you want to map entities in
    according to entities_map
    :param entities_map: (dict) a dictionary for entity mappings
    :param choose_span_func: (func) function that chooses one, from two
    overlapping entities
    :param kwargs: (dict), currently supports only:
        *  entity_hierarchy - a list indicating the importance of entities,
        used to choose a more important entity in situations when two overlap
    :return: all_recipes: (list) a list with recipes split to tokens
    :return: all_entities: (list) a list with entities for each token from
    all_recipes
    """

    if isinstance(annotations_paths, list):
        ann_files = annotations_paths
    elif isinstance(annotations_paths, str):
        ann_files = [os.path.join(annotations_paths, file) for file in
            os.listdir(annotations_paths) if ".ann" in file]

    if isinstance(recipes_paths, list):
        recipe_files = recipes_paths
    elif isinstance(recipes_paths, str):
        recipe_files = [os.path.join(annotations_paths,
                                     re.findall(r'\d+', ann_file)[-1] + ".txt")
                        for ann_file in ann_files]

    all_recipes = []
    all_entities = []

    print("Loading annotation files")
    for ann_path, recipe_path in tqdm(zip(ann_files, recipe_files),
                                      total=len(ann_files)):

        recipe_tokens, recipe_entities = process_single_annotation(
            ann_path, recipe_path, scheme_func, map_entity_func,
            entities_map, choose_span_func, kwargs["entity_hierarchy"]
        )
        all_recipes.append(recipe_tokens)
        all_entities.append(recipe_entities)

    return all_recipes, all_entities


def collect_recipes_without_annotations(recipes):
    """
    :param recipes: a path to folder with recipes, a path to a recipe, a list of
    paths to recipe files or a list of recipes
    :return: recipes_tokens: (list) a list with recipes split to tokens
    """

    list_with_recipes = []
    if isinstance(recipes, str):
        if os.path.isdir(recipes): # path to directory with recipes
            recipe_paths = \
                [os.path.join(recipes, recipe_file) for recipe_file in
                 recipes if recipe_file.endswith(".txt")]

            for recipe_path in recipe_paths:
                with open(recipe_path, "r") as f:
                    recipe = f.read()
                list_with_recipes.append(recipe)

        elif os.path.isfile(recipes):  # path to recipe
            with open(recipes, "r") as f:
                list_with_recipes = [f.read()]

        else:
            raise FileNotFoundError(f"{recipes} is neither a path to directory"
                                    f"nor a path to recipe!")

    if isinstance(recipes, list):
        if os.path.isfile(recipes[0]):  # list of paths to recipes
            for recipe_path in recipes:
                with open(recipe_path, "r") as f:
                    recipe = f.read()
                list_with_recipes.append(recipe)

        else:  # list of recipes
            list_with_recipes = recipes

    recipes_tokens = []
    print("Tokenizing recipes")
    for recipe in tqdm(list_with_recipes, total=len(list_with_recipes)):

        doc = nlp.make_doc(recipe)

        recipe_tokens = [prepare_token_text(token.text) for token in doc]

        recipes_tokens.append(recipe_tokens)

    return recipes_tokens

def prepare_text(text):
    doc = nlp.make_doc(text)
    return doc
