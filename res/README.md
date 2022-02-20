Download the model and its config (optionally tokenizer configs) from Google Drive and put them into the ner_model folder. This model was trained on all 599 recipes. Previously, hyperparameters were chosen on a 5-fold Cross-Validation.

Currently available models are:
1. one using bert-base-cased checkpoint. It is available [here](https://drive.google.com/drive/folders/1-9sQ9AG76WQR0GRDEvMzDW8IAlDH545s?usp=sharing).

  | Entity | Precision | Recall | F1-score |
  | :----------: | :----------: | :----------: | :----------: |
  | COLOR | 0.889 ± 0.063 | 0.916 ± 0.048 | 0.902 ± 0.051 |
  | FOOD | 0.897 ± 0.009 | 0.913 ± 0.013 | 0.905 ± 0.011 |
  | PHYSICAL QUALITY | 0.766 ± 0.043 | 0.795 ± 0.064 | 0.779 ± 0.049 |
  | PROCESS | 0.905 ± 0.018 | 0.918 ± 0.014 | 0.911 ± 0.013 |
  | QUANTITY | 0.977 ± 0.008 | 0.985 ± 0.006 | 0.981 ± 0.006 |
  | UNIT | 0.966 ± 0.016 | 0.977 ± 0.009 | 0.971 ± 0.013 |
  | all | 0.929 ± 0.009 | 0.942 ± 0.008 | 0.936 ± 0.008 |
