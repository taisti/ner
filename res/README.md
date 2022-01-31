Download the model and its config from Google Drive and put it into the ner_model folder.

Currently available models are:
1. one using bert-base-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1HREtqeC8UrmCn5Iw-dRxibszsmZSYGgx?usp=sharing).
  * Val set (60 recipes):

  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | UNIT | 0.985 | 0.964 | 0.975 | 272 | 
  | QUANT | 0.968 | 0.983 | 0.975 | 348 |
  | FOOD | 0.878 | 0.905 | 0.891 | 378 |
  | all | 0.939 | 0.948 | 0.944 | 998 |

  * Test set (98 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | UNIT | 0.966 | 0.988 | 0.977 | 442 | 
  | QUANT | 0.961 | 0.986 | 0.974 | 518 |
  | FOOD | 0.876 | 0.923 | 0.899 | 573 |
  | all | 0.931 | 0.964 | 0.947 | 1533 |

2. one using bert-large-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1Q24_e6rJWjlPEHrYYDPe47jsfYSQU4k5?usp=sharing]).
  * Val set (60 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | UNIT | 0.968 | 0.986 | 0.977 | 283 | 
  | QUANT | 0.966 | 0.985 | 0.975 | 350 |
  | FOOD | 0.903 | 0.91 | 0.906 | 370 |
  | all | 0.943 | 0.957 | 0.95 | 1003 |

  * Test set (98 recipes):
  
  | Entity | Precision | Recall | F1-score | Support |
  | :----------: | :----------: | :----------: | :----------: | :----------: | 
  | UNIT | 0.953 | 0.988 | 0.97 | 448 | 
  | QUANT | 0.961 | 0.988 | 0.975 | 519 |
  | FOOD | 0.903 | 0.919 | 0.911 | 554 |
  | all | 0.938 | 0.963 | 0.95 | 1521 |
