Repo for NER.

So far Python 3.6.9 has been used. If you have other version, perhaps it is
fine if you just remove versions from `requirements.txt`.

TODO: dockerization or equivalent
TODO: runner script for TaistiNer
TODO: data_processing refactoring (more functional, less conditional)

Note: Preferably unpack data from Google Drive (0-1-2-3-4.zip) to data/annotations.

Run
```bash
python -m venv ner_env
. ./ner_env/bin/activate
pip install --upgrade pip  # upgrade pip to newest version
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To freely import all scripts, please add the followings to the PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:<path-to-ner-repo-folder>/src
```

In order to download models please check (this folder)[./res/].

TODO: use git-lfs (may be a problem due to constraints for the free version)

In src/prepare_data_utils.py you can find a script for data preparation. It 
is partly, and will be fully functional to provide flexibility. However, right
now we need to design a full architecture to code the data preparation process
accordingly.

In `src/model_ner_runner.ipynb` you can see the training, evaluation and prediction.
These relate to `res/ner_model` a first version of NERTaisti model. 

You can also generate some plots, for instance, with the following command:
```bash
python eda/generate_plots.py \
-af ./data/annotations \
-if ./eda/images \
-noc 20 \
--show-fig True \
--save-fig True
```
See ./eda/generate_plots.py for more details.
