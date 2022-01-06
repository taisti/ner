ENTITIES_MAP = {
    "food_product_with_unit": "FOOD",
    "food_product_without_unit_uncountable": "FOOD",
    "food_product_whole": "FOOD",
    "food_product_without_unit_countable": "FOOD",
    "quantity": "QUANT",
    "unit": "UNIT",
    "physical_quality": "O",
    "color": "O",
    "trade_name": "O",
    "purpose": "O",
    "taste": "O",
    "process": "O",
    "example": "O",
    "part": "O",
    "diet": "O",
    "possible_substitute": "O",
    "excluded": "O",
    "exclusive": "O"
}


ENTITY_TO_LABEL = {
    'O': 0,
    'B-FOOD': 1,
    'I-FOOD': 2,
    'B-QUANT': 3,
    'I-QUANT': 4,
    'B-UNIT': 5,
    'I-UNIT': 6
}


LABEL_TO_ENTITY = {v: k for k, v in ENTITY_TO_LABEL.items()}
