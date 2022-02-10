import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train_en.npz'
test_dir = data_dir + 'test_en.npz'
files = ['train', 'test']
bert_model = 'bert-base-cased'
roberta_model = 'bert-base-cased'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = ''

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['FOOD', 'UNIT', 'QUANT']

label2id = {
    'O': 0,
    'B-FOOD': 1,
    'I-FOOD': 2,
    'B-UNIT': 3,
    'I-UNIT': 4,
    'B-QUANT': 5,
    'I-QUANT': 6
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
