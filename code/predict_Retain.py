import torch
import numpy as np
import dill
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd

from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, sequence_output_process

torch.manual_seed(1203)
np.random.seed(1203)

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records

def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open('voc_final.pkl','wb'))
    return diag_voc, med_voc, pro_voc


def eval(model, data_eval, voc_size):
    # evaluate
    print('')
    model.eval()
    for i, input in enumerate(data_eval):
        if len(input) < 2: # visit > 2
            print(f"{i+1} Skipped")
            continue
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for i in range(1, len(input)):

            y_pred_label_tmp = []
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = model(input[:i])

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.3] = 1
            y_pred_tmp[y_pred_tmp < 0.3] = 0
            y_pred.append(y_pred_tmp)
            for idx, value in enumerate(y_pred_tmp):
                if value == 1:
                    y_pred_label_tmp.append(idx)
            y_pred_label.append(y_pred_label_tmp)
        print([med_voc.idx2word[label] for label in y_pred_label_tmp])

def main():
    if not os.path.exists(os.path.join("baseline/saved", model_name)):
        os.makedirs(os.path.join("baseline/saved", model_name))

    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point+eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    model = Retain(voc_size, device=device)
    model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)
    eval(model, data_eval, voc_size)


model_name = 'Retain'
resume_name = "baseline/saved/Retain/final.model"

path='../data/data_final.pkl'
df = pd.read_pickle(path)
diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
main()
