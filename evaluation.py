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



# Target vehicle = vehicle on which the attack is evaluated
# Source vehicle = vehicle from which the attacks are generated

for target_vehicle in vehicles:
    datasetPath = f'./dataset/{target_vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

    results_df = pd.DataFrame(columns=['Target_Vehicle', 'Source_Vehicle', 'Target_Model', 'Source_Model', 'Attack', 'Epsilon', 'Accuracy', 'F1'])

    input_size = len(df.columns) - 1
    hidden_size = 64
    output_size = 4

    fcn = FCNMultiClass(input_size, hidden_size, output_size)
    cnn = CNNMultiClass(input_size, output_size)
    lstm = LSTMMultiClass(input_size, hidden_size, output_size)

    target_models = [fcn, cnn, lstm]

    fcnPath = f'./models/{target_vehicle}/FCN_multi.pth'
    cnnPath = f'./models/{target_vehicle}/CNN_multi.pth'
    lstmPath = f'./models/{target_vehicle}/LSTM_multi.pth'

    paths = [fcnPath, cnnPath, lstmPath]

    for source_vehicle in vehicles:
        for target_model, target_model_name, path in zip(target_models, model_names, paths):
            target_model.load_state_dict(torch.load(path))
            target_model.to(device)

            for source_model_name in model_names:
                for attack_name in attack_names:
                    for epsilon in epsilons:
                        attackPath = f'./attacks/{source_vehicle}/{source_model_name}/{attack_name}/{source_vehicle}_{source_model_name}_{attack_name}_{epsilon}.csv'

                        attackdf = pd.read_csv(attackPath)
                        
                        features = attackdf.drop(['Class'], axis=1).values
                        labels = attackdf['Class'].values

                        _, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

                        X_test_tensor = torch.FloatTensor(X_test)
                        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

                        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

                        accuracy, f1 = evaluate_multi_class_model(target_model, test_dataloader, device)
                        
                        print(f'[👑 {target_vehicle} {target_model_name}] {source_vehicle} @ {source_model_name} @ {attack_name} @ {epsilon}')
                        print(f'\t[📏 ACCURACY] {accuracy:.3f}')
                        print(f'\t[⚖️ F1 SCORE] {f1:.3f}\n')
                        
                        results_df = results_df.append({
                            'Target_Vehicle': target_vehicle,
                            'Source_Vehicle': source_vehicle,
                            'Target_Model': target_model_name,
                            'Source_Model': source_model_name,
                            'Attack': attack_name,
                            'Epsilon': epsilon,
                            'Accuracy': accuracy,
                            'F1': f1
                        }, ignore_index=True)

    results_df.to_csv(f'./results/transferability/{target_vehicle}.csv', index=False)