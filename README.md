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

DDI data link : https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0

# Preprocessing code + command (if applicable):

run code ./preproccesing/EDA.ipynb. This can be executed on jupyter notebook for clear understanding.

# Training code + command

Run code for baseline


* ./code/baseline/train_Retain.py
* ./code/baseline/train_Leap.py

Run code for GAMENet
* ./code/train_GAMENet.py

# Evaluation code + command
GAMENet

* Run ./code/predict.py
python predict.py

Baseline:

* Run ./code/predict_Leap.py
* Run ./code/predict_Retain.py



# Pretrained model (if applicable):
NA

# Table of results :


