Download the model and its config (optionally tokenizer configs) from Google Drive and put them into the ner_model folder. All of these models were trained on 440 recipes.

Currently available models are:
1. one using bert-base-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1-gv1H8cjcP-0uvi75wwf6WsfNxo_3RRH?usp=sharing).
  * Val set (60 recipes):

  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | FOOD | 0.886 | 0.894 | 0.89 | 370 |
  | QUANT | 0.977 | 0.985 | 0.981 | 346 |
  | UNIT | 0.993 | 0.961 | 0.976 | 270 |
  | all | 0.947 | 0.944 | 0.946 | 986 |

  * Test set (98 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | FOOD | 0.885 | 0.906 | 0.896 | 557 |
  | QUANT | 0.965 | 0.99 | 0.978 | 518 |
  | UNIT | 0.968 | 0.984 | 0.976 | 439 |
  | all | 0.937 | 0.957 | 0.947 | 1514 |

2.  one using bert-base-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1-XQ07_5WOa5pKLspkYMDUaKdl-nlIlbs?usp=sharing). It classifies all entities.
  * Val set (60 recipes):

  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: |
  | COLOR | 0.8 | 0.923 | 0.857 | 15 |
  | DIET | 0.222 | 0.182 | 0.2 | 9 |
  | EXAMPLE | 0.5 | 0.375 | 0.429 | 6 |
  | EXCLUDED | 0.0 | 0.0 | 0.0 | 0 |
  | EXCLUSIVE | 0.0 | 0.0 | 0.0 | 0 |
  | FOOD | 0.893 | 0.913 | 0.903 | 375 |
  | PART | 0.167 | 0.25 | 0.2 | 6 |
  | PHYSICAL QUALITY | 0.733 | 0.764 | 0.748 | 75 |
  | POSSIBLE SUBSTITUTE | 0.0 | 0.0 | 0.0 | 0 |
  | PROCESS | 0.91 | 0.925 | 0.917 | 122 |
  | PURPOSE | 0.786 | 0.846 | 0.815 | 14 |
  | QUANTITY | 0.985 | 0.985 | 0.985 | 343 |
  | TASTE | 0.556 | 0.5 | 0.526 | 9 |
  | TRADE NAME | 0.2 | 0.2 | 0.2 | 5 |
  | UNIT | 0.978 | 0.961 | 0.969 | 274 |
  | all | 0.911 | 0.917 | 0.914 | 1253 |
  
  * Test set (98 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: |
  | COLOR | 0.914 | 0.889 | 0.901 | 35 |
  | DIET | 0.0 | 0.0 | 0.0 | 11 |
  | EXAMPLE | 0.364 | 0.333 | 0.348 | 11 |
  | EXCLUDED | 0.0 | 0.0 | 0.0 | 0 | 
  | EXCLUSIVE | 0.0 | 0.0 | 0.0 | 0 | 
  | FOOD | 0.902 | 0.93 | 0.916 | 561 |
  | PART | 1.0 | 0.571 | 0.727 | 4 |
  | PHYSICAL QUALITY | 0.728 | 0.806 | 0.765 | 114 |
  | POSSIBLE SUBSTITUTE | 0.0 | 0.0 | 0.0 | 0 |
  | PROCESS | 0.866 | 0.92 | 0.892 | 119 |
  | PURPOSE | 0.737 | 0.824 | 0.778 | 19 | 
  | QUANTITY | 0.961 | 0.988 | 0.975 | 519 | 
  | TASTE | 0.429 | 0.4 | 0.414 | 14 |
  | TRADE NAME | 0.412 | 0.389 | 0.4 | 17 | 
  | UNIT | 0.962 | 0.991 | 0.976 | 445 |
  | all | 0.902 | 0.929 | 0.916 | 1869 |
3. one using bert-large-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1-heWiri49i5fztjZRJT41Zl-Ao0zu6Q0?usp=sharing).
  * Val set (60 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | FOOD | 0.906 | 0.916 | 0.911 | 371 | 
  | QUANTITY | 0.98 | 0.985 | 0.983 | 345 |
  | UNIT | 0.982 | 0.968 | 0.975 | 275 |
  | all | 0.953 | 0.954 | 0.954 | 991 |

  * Test set (98 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | FOOD | 0.906 | 0.934 | 0.919 | 561 |
  | QUANTITY | 0.967 | 0.996 | 0.981 | 520 |
  | UNIT | 0.964 | 0.988 | 0.976 | 443 |
  | all | 0.944 | 0.971 | 0.957 | 1524 |
