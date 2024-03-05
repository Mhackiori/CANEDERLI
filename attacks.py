import numpy as np
import os
import pandas as pd
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchattacks

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils.helpers import *
from utils.models import *
from utils.params import *

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

setSeed()




for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')
    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

    features = df.drop(['Class'], axis=1).values
    labels = df['Class'].values

    features, labels = RandomUnderSampler(random_state=seed).fit_resample(features, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

    input_size = len(df.columns) - 1
    hidden_size = 64
    output_size = 4

    fcn = FCNMultiClass(input_size, hidden_size, output_size)
    cnn = CNNMultiClass(input_size, output_size)
    lstm = LSTMMultiClass(input_size, hidden_size, output_size)

    models = [fcn, cnn, lstm]

    fcnPath = f'./models/{vehicle}/FCN_multi.pth'
    cnnPath = f'./models/{vehicle}/CNN_multi.pth'
    lstmPath = f'./models/{vehicle}/LSTM_multi.pth'

    paths = [fcnPath, cnnPath, lstmPath]

    for model, model_name, path in zip(models, model_names, paths):
        if model_name == 'LSTM':
            device = torch.device('cpu')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(path))
        model.to(device)

        for epsilon in epsilons:
            attacks = [
                torchattacks.BIM(model, eps=epsilon),
                torchattacks.FGSM(model, eps=epsilon),
                torchattacks.PGD(model, eps=epsilon),
                torchattacks.RFGSM(model, eps=epsilon),
            ]

            for attack, attack_name in zip(attacks, attack_names):

                advFold = f'./attacks/{vehicle}/{model_name}/{attack_name}/'

                if not os.path.exists(advFold):
                    os.makedirs(advFold)
                    
                advPath = f'{advFold}{vehicle}_{model_name}_{attack_name}_{epsilon}.csv'

                if not os.path.exists(advPath):
                    adversarial_samples = []
                    adversarial_labels = []

                    for i, sample in enumerate(X_test):
                        print(f'\t[⚔️ {model_name} {attack_name} @ {epsilon}] {i+1}\{len(X_test)}', end='\r')

                        sample_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
                        label_tensor = torch.tensor(y_test[i], dtype=torch.long).to(device)

                        adversarial_sample = attack(sample_tensor.unsqueeze(0), label_tensor.unsqueeze(0))

                        adversarial_samples.append(adversarial_sample.squeeze(0).cpu().numpy())
                        adversarial_labels.append(y_test[i])

                    adversarial_dataset = pd.DataFrame(adversarial_samples, columns=df.drop(['Class'], axis=1).columns)
                    adversarial_dataset['Class'] = adversarial_labels

                    adversarial_dataset.to_csv(advPath, index=False)

                    print()
                else:
                    print(f'\t[✅ DONE] {model_name} {attack_name} @ {epsilon}')