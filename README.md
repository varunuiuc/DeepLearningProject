# DeepLearningProject

DeepLearning Final Project for Healt  CS 598 Deep Learning for Healthcarecare:CS 598 Deep Learning for Healthcare
GAMENet : Graph Augmented MEmory Networks for Recommending Medication Combination


# Citation to the original paper
GAMENet : Graph Augmented MEmory Networks for Recommending Medication Combination
https://arxiv.org/abs/1809.01852

# Link to the original paper’s repo (if applicable)
https://github.com/sjy1203/GAMENet

# Dependencies
`
pip install -U pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install dill dnc scikit-learn pandas
`
* Pytorch >=0.4
* Python >=3.5

# Data download instruction:
GAMENet is tested on real-world clinical dataset MIMIC-III.
Demo data set can be accessed to public.
https://physionet.org/content/mimiciii-demo/1.4/
Data can be downloaded as 

Real world dataset (MIMIC-III v1.4) on same physionet site can be accessed based on credential access.
With valid/approved credentials data can be downloaded as csv or can be accessed on AWS or GCP.

* download MIMIC data and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
* download DDI data and put it in ./data/


# Preprocessing code + command (if applicable):
run code ./preproccesing/EDA.ipynb

# Training code + command


# Evaluation code + command


# Pretrained model (if applicable):
NA

# Table of results :


