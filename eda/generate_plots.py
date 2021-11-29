import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from src.prepare_data_utils import process_ann_file


def count_entities_types(quantities_dict, entities, nlp):

    for entity in entities.values():
        if entity.type in quantities_dict.keys():
            quantities_dict[entity.type] += 1
        else:
            quantities_dict[entity.type] = 1

    return quantities_dict


def count_food_entities(quantities_dict, entities, nlp):#TODO kwargs for nlp argument or default none

    for entity in entities.values():
        if "food" in entity.type:
            #print(entity.text) #TODO print text: ...

            doc = nlp(entity.text)
            entity_lemmas = " ".join([token.lemma_ for token in doc])
            #print(entity_lemmas)

            if entity_lemmas in quantities_dict.keys():
                quantities_dict[entity_lemmas] += 1
            else:
                quantities_dict[entity_lemmas] = 1

    return quantities_dict


def calculate_quantities(path_to_annotations_folder, count_func, nlp):

    quantities = {}

    files = os.listdir(path_to_annotations_folder)
    ann_files = [file for file in files if ".ann" in file]

    num_of_files = 0

    for ann_file in ann_files:

        num_of_files += 1
        path_to_ann_file = os.path.join(path_to_annotations_folder, ann_file)
        entities, relations = process_ann_file(path_to_ann_file)

        quantities = count_func(quantities, entities, nlp)

    return quantities, num_of_files


def set_plotting_style():
    sns.set_style("whitegrid")


def plot_entities_quantities(quantities, num_of_files, num_of_categories, images_folder_path, save_fig, show_fig):

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

    plt.title(f"Entities quantities within {num_of_files} recipes", size=20)
    plt.tight_layout()

    if save_fig:
        fig_path = os.path.join(images_folder_path, "entities_types_distribution.png")
        plt.savefig(fig_path)

    if show_fig:
        plt.show()


def plot_food_entities_quantities(quantities, num_of_files, num_of_categories, images_folder_path, save_fig, show_fig):

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
        ax.text(i, v + 10, str(v), color='blue', fontweight='bold', rotation=0, ha='center', size=10)

    plt.title(f"Food entities quantities within {num_of_files} recipes", size=20)
    plt.tight_layout()

    if save_fig:
        fig_path = os.path.join(images_folder_path, "entities_types_distribution.png")
        plt.savefig(fig_path)

    if show_fig:
        plt.show()


def main(args):

    if args.save_fig:
        os.makedirs(args.images_folder, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")

    quantities, num_of_files = calculate_quantities(args.annotations_folder, count_entities_types, nlp)

    plot_entities_quantities(quantities, num_of_files, args.num_of_categories, args.images_folder, args.save_fig,
                             args.show_fig)

    quantities, num_of_files = calculate_quantities(args.annotations_folder, count_food_entities, nlp)

    plot_food_entities_quantities(quantities, num_of_files, args.num_of_categories, args.images_folder, args.save_fig,
                                  args.show_fig)


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
