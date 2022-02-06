import os
import re
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from src.prepare_data_utils import process_ann_file
from src.mappings import ENTITIES_MAP


def count_entities_types(quantities_dict, entities, nlp, entity_type):

    for entity in entities.values():
        if entity.type in quantities_dict.keys():
            quantities_dict[entity.type] += 1
        else:
            quantities_dict[entity.type] = 1

    return quantities_dict


# TODO kwargs for nlp argument or default none
def count_entities_examples(quantities_dict, entities, nlp, entity_type):

    for entity in entities.values():
        if entity.type.startswith(entity_type):
            # print(entity.text)  #TODO print text: ...

            doc = nlp(entity.text)
            entity_lemmas = " ".join([token.lemma_ for token in doc])
            # print(entity_lemmas)

            if entity_lemmas in quantities_dict.keys():
                quantities_dict[entity_lemmas] += 1
            else:
                quantities_dict[entity_lemmas] = 1

    return quantities_dict


def calculate_quantities(ann_files_paths, count_func, nlp, entity_type):

    quantities = {}

    num_of_files = 0

    for path_to_ann_file in ann_files_paths:

        num_of_files += 1
        entities, relations = process_ann_file(path_to_ann_file)

        quantities = count_func(quantities, entities, nlp, entity_type)

    return quantities, num_of_files


def set_plotting_style():
    sns.set_style("whitegrid")


def plot_entities_quantities(quantities, num_of_categories, images_folder_path, fig_title, save_fig,
                             show_fig):

    quantities = dict(sorted(quantities.items(), key=lambda x: x[1], reverse=True))

    other_quantity = 0

    # check if the 'other' category is necessary
    if num_of_categories < len(quantities):
        quantities_with_other = {}
        for i, (k, v) in enumerate(quantities.items()):
            if i < num_of_categories - 1:
                quantities_with_other[k] = v
            else:
                other_quantity += v

        quantities_with_other["others"] = other_quantity
        quantities = quantities_with_other

    entities = [entity.replace("_", " ") for entity in quantities.keys()]

    fig, ax = plt.subplots()

    # shorten labels for visualization
    for idx in range(len(entities)):
        entities[idx] = entities[idx].replace("food product without unit uncountable",
                                              "food product without\nunit uncountable")
        entities[idx] = entities[idx].replace("food product with unit", "food product\nwith unit")
        entities[idx] = entities[idx].replace("food product without unit countable",
                                              "food product without\nunit countable")

    ax.bar(entities, quantities.values())
    plt.xticks(rotation=90, size=10)

    for i, v in enumerate(quantities.values()):
        ax.text(i, v + 10, str(v), color='blue', fontweight='bold', rotation=0, ha='center', size=10)

    plt.title(fig_title, size=20)
    plt.ylim(0, 1.2 * max(quantities.values()))
    plt.tight_layout()

    if save_fig:
        fig_path = os.path.join(images_folder_path, f"{fig_title}.png")
        plt.savefig(fig_path)

    if show_fig:
        plt.show()


def plot_entities_examples_quantities(quantities, num_of_files, num_of_categories, entity_type, images_folder_path,
                                      save_fig, show_fig):

    quantities = dict(sorted(quantities.items(), key=lambda x: x[1],
                             reverse=True))

    other_quantity = 0

    # check if the 'other' category is necessary
    if num_of_categories < len(quantities):
        quantities_with_other = {}
        for i, (k, v) in enumerate(quantities.items()):
            if i < num_of_categories - 1:
                quantities_with_other[k] = v
            else:
                other_quantity += v

        quantities_with_other["others"] = other_quantity
        quantities = quantities_with_other

    entities = list(quantities.keys())

    fig, ax = plt.subplots()

    ax.bar(entities, quantities.values())
    plt.xticks(rotation=90, size=10)

    for i, v in enumerate(quantities.values()):
        ax.text(i, v + max(quantities.values())*0.005, str(v), color='blue', fontweight='bold', rotation=0,
                ha='center', size=10)

    plt.title(f"{entity_type} entities quantities within {num_of_files} recipes", size=20)
    plt.ylim(0, 1.2 * max(quantities.values()))
    plt.tight_layout()

    if save_fig:
        fig_path = os.path.join(images_folder_path, f"{entity_type}_examples_distribution.png")
        plt.savefig(fig_path)

    if show_fig:
        plt.show()


def main(args):

    if args.save_fig:
        os.makedirs(args.images_folder, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    all_ann_files_paths = [os.path.join(args.annotations_folder, file) for file in os.listdir(args.annotations_folder)
                           if ".ann" in file]
    val_ann_files_paths = [path for path in all_ann_files_paths if 240 <= int(re.findall(r'\d+', path)[0]) < 300]
    test_ann_files_paths = [path for path in all_ann_files_paths if 400 <= int(re.findall(r'\d+', path)[0]) < 500]
    train_ann_files_paths = [path for path in all_ann_files_paths if path not in val_ann_files_paths
                             and path not in test_ann_files_paths]

    ann_files_paths = {
        "all": all_ann_files_paths,
        "train": train_ann_files_paths,
        "val": val_ann_files_paths,
        "test": test_ann_files_paths,
    }

    for set_type in ann_files_paths.keys():
        quantities, num_of_files = calculate_quantities(ann_files_paths[set_type], count_entities_types, nlp, None)
        fig_title = f"Entities quantities within {num_of_files} files - {set_type} recipes"
        plot_entities_quantities(quantities, args.num_of_categories, args.images_folder, fig_title, args.save_fig,
                                 args.show_fig)

    entity_types = ["food"] + [el for el in list(ENTITIES_MAP.keys()) if not el.startswith("food")]
    for entity_type in entity_types:

        quantities, num_of_files = calculate_quantities(ann_files_paths['all'], count_entities_examples,
                                                        nlp, entity_type)

        if entity_type == "quantity":
            quantities_with_numeric = {"numeric": 0}
            for k, v in quantities.items():
                if bool(re.search(r'([0-9]+.[1-9][0-9]|[0-9]+)', k)):
                    quantities_with_numeric["numeric"] += v
                else:
                    quantities_with_numeric[k] = v

            quantities = quantities_with_numeric

        plot_entities_examples_quantities(quantities, num_of_files, args.num_of_categories, entity_type,
                                          args.images_folder, args.save_fig, args.show_fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-af', '--annotations-folder', type=str, default="../data/annotations",
                        help='Path to folder with annotations')
    parser.add_argument('-if', '--images-folder', type=str, default="./images", help='Path where images are saved')
    parser.add_argument('-noc', '--num-of-categories', type=int, required=True,
                        help='Number of categories to be included')
    parser.add_argument('--show-fig', type=bool, default=True, help='Whether to show figs')
    parser.add_argument('--save-fig', type=bool, default=True, help='Whether to save figs')

    args = parser.parse_args()

    main(args)
