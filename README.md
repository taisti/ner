Repo for NER.

So far Python 3.6.9 has been used. If you have other version, perhaps it is
fine if you just remove versions from `requirements.txt`.

TODO: dockerization or equivalent

Note: Preferably unzip data from Google Drive (0.zip) to data/annotations.

Run
```bash
python -m venv ner_env
. ./ner_env/bin/activate
pip install --upgrade pip  # upgrade pip to newest version
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

In order to download models (currently, only FoodNer) use git-lfs:
```bash
git lfs pull origin data-preparation-and-foodner
```
Perhaps you might need to install Git LFS, for instance
```bash
sudo apt-get install git-lfs
```

In src/prepare_data_utils.py you can find a script for data preparation. It 
is partly, and will be fully functional to provide flexibility. However, right
now we need to design a full architecture to code the data preparation process
accordingly.

In src/FoodNer you can FoodNer, a BERT fine-tuned for food entities extraction.
If you run it, you'll get predictions for our data and associated metrics.
You can run it with, for instance, with the command below:
```bash
python src/FoodNer.py \
--foodner-path ./res/foodner \
--annotations-folder \
./data/annotations \
--recipes-folder ./data/annotations
```

FoodNer's paper: [A Fine-Tuned Bidirectional Encoder Representations From Transformers
Model for Food Named-Entity Recognition: Algorithm Development and
Validation](https://www.researchgate.net/publication/353789336_A_Fine-Tuned_Bidirectional_Encoder_Representations_From_Transformers_Model_for_Food_Named-Entity_Recognition_Algorithm_Development_and_Validation)


Ideas for NER models:
* classicaly, HuggingFace and `BertForTokenClassification`
* Nerda library, API for NER (looks very friendly!)
  * https://github.com/ebanalyse/NERDA
  * tutorial: https://towardsdatascience.com/easy-fine-tuning-of-transformers-for-named-entity-recognition-d72f2b5340e3