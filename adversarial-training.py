import numpy as np
import os
import pandas as pd
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchattacks
import glob

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



# Adversarial fine-tuning
print('[🔧 FINE-TUNING]\n')

# Same vehicle, same model, all attacks
# Evaluation on everything
print('[⚔️ ALL-ATTACKS]')

# Fine-tuning models
for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')

    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

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

    # Fine-tuning models
    for model, model_name, path in zip(models, model_names, paths):
        advFold = f'./models/{vehicle}/adversarial-training/all-attacks/'
        
        if not os.path.exists(advFold):
            os.makedirs(advFold)

        advPath = os.path.join(advFold, f'{model_name}.pth')

        # Loading and processing attack datasets
        dfs_train = []
        dfs_test = []

        attacksDir = f'./attacks/{vehicle}/{model_name}/'
        for attack_type in os.listdir(attacksDir):
            attack_path = os.path.join(attacksDir, attack_type)

            if os.path.isdir(attack_path):
                for csv_file in os.listdir(attack_path):
                    if csv_file.endswith('.csv'):
                        csv_path = os.path.join(attack_path, csv_file)

                        df = pd.read_csv(csv_path)
                        
                        features = df.drop(['Class'], axis=1).values
                        labels = df['Class'].values

                        # Perform train-test split
                        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

                        # Append the split datasets to lists
                        dfs_train.append((X_train, y_train))
                        dfs_test.append((X_test, y_test))

        # Combine all splits into training and testing sets
        X_train_all = np.concatenate([item[0] for item in dfs_train], axis=0)
        y_train_all = np.concatenate([item[1] for item in dfs_train], axis=0)

        X_test_all = np.concatenate([item[0] for item in dfs_test], axis=0)
        y_test_all = np.concatenate([item[1] for item in dfs_test], axis=0)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_all)
        y_train_tensor = torch.FloatTensor(y_train_all).unsqueeze(1)

        X_test_tensor = torch.FloatTensor(X_test_all)
        y_test_tensor = torch.FloatTensor(y_test_all).unsqueeze(1)

        # Create DataLoader for training set
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Loading pre-trained model
        model.load_state_dict(torch.load(path))
        model.to(device)

        # Adversarial training
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(advPath):
            epochs = 30
            for epoch in range(epochs):
                print(f'\t[💪 {model_name}] {epoch+1}/{epochs}', end='\r')
                train_multi_class_model(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            print()
            torch.save(model.state_dict(), advPath)
        else:
            model.load_state_dict(torch.load(advPath))

        accuracy, f1 = evaluate_multi_class_model(model, test_dataloader, device)
        print(f'\t[👑 {model_name}] Accuracy: {accuracy:.3f}, F1: {f1:.3f}')

# Evaluating models
        
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

    fcnPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/FCN.pth'
    cnnPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/CNN.pth'
    lstmPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/LSTM.pth'

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

    results_df.to_csv(f'./results/adversarial-training/all-attacks/{target_vehicle}.csv', index=False)



############################################################################################################



# Same vehicle, all models, all attacks
# Evaluation on everything
print('[🤖 ALL-MODELS]')

# Fine-tuning models
for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')

    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

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

    # Fine-tuning models
    for model, model_name, path in zip(models, model_names, paths):
        advFold = f'./models/{vehicle}/adversarial-training/all-models/'
        
        if not os.path.exists(advFold):
            os.makedirs(advFold)

        advPath = os.path.join(advFold, f'{model_name}.pth')

        # Loading and processing attack datasets
        dfs_train = []
        dfs_test = []

        attacksDir = f'./attacks/{vehicle}/'
        csv_files = glob.glob(f'{attacksDir}/**/*.csv', recursive=True)

        for csv_file in csv_files:
            df = pd.read_csv(csv_path)
                        
            features = df.drop(['Class'], axis=1).values
            labels = df['Class'].values

            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

            # Append the split datasets to lists
            dfs_train.append((X_train, y_train))
            dfs_test.append((X_test, y_test))

        # Combine all splits into training and testing sets
        X_train_all = np.concatenate([item[0] for item in dfs_train], axis=0)
        y_train_all = np.concatenate([item[1] for item in dfs_train], axis=0)

        X_test_all = np.concatenate([item[0] for item in dfs_test], axis=0)
        y_test_all = np.concatenate([item[1] for item in dfs_test], axis=0)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_all)
        y_train_tensor = torch.FloatTensor(y_train_all).unsqueeze(1)

        X_test_tensor = torch.FloatTensor(X_test_all)
        y_test_tensor = torch.FloatTensor(y_test_all).unsqueeze(1)

        # Create DataLoader for training set
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Loading pre-trained model
        model.load_state_dict(torch.load(path))
        model.to(device)

        # Adversarial training
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(advPath):
            epochs = 30
            for epoch in range(epochs):
                print(f'\t[💪 {model_name}] {epoch+1}/{epochs}', end='\r')
                train_multi_class_model(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            print()
            torch.save(model.state_dict(), advPath)
        else:
            model.load_state_dict(torch.load(advPath))

        accuracy, f1 = evaluate_multi_class_model(model, test_dataloader, device)
        print(f'\t[👑 {model_name}] Accuracy: {accuracy:.3f}, F1: {f1:.3f}')

# Evaluating models
        
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

    fcnPath = f'./models/{target_vehicle}/adversarial-training/all-models/FCN.pth'
    cnnPath = f'./models/{target_vehicle}/adversarial-training/all-models/CNN.pth'
    lstmPath = f'./models/{target_vehicle}/adversarial-training/all-models/LSTM.pth'

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

    results_df.to_csv(f'./results/adversarial-training/all-models/{target_vehicle}.csv', index=False)



############################################################################################################



# All vehicles, all models, all attacks
# Evaluation on everything
print('[🚜 ALL-VEHICLES]')

# Fine-tuning models
for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')

    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

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

    # Fine-tuning models
    for model, model_name, path in zip(models, model_names, paths):
        advFold = f'./models/{vehicle}/adversarial-training/all-vehicles/'
        
        if not os.path.exists(advFold):
            os.makedirs(advFold)

        advPath = os.path.join(advFold, f'{model_name}.pth')

        # Loading and processing attack datasets
        dfs_train = []
        dfs_test = []

        attacksDir = f'./attacks/'
        csv_files = glob.glob(f'{attacksDir}/**/*.csv', recursive=True)

        for csv_file in csv_files:
            df = pd.read_csv(csv_path)
                        
            features = df.drop(['Class'], axis=1).values
            labels = df['Class'].values

            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

            # Append the split datasets to lists
            dfs_train.append((X_train, y_train))
            dfs_test.append((X_test, y_test))

        # Combine all splits into training and testing sets
        X_train_all = np.concatenate([item[0] for item in dfs_train], axis=0)
        y_train_all = np.concatenate([item[1] for item in dfs_train], axis=0)

        X_test_all = np.concatenate([item[0] for item in dfs_test], axis=0)
        y_test_all = np.concatenate([item[1] for item in dfs_test], axis=0)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_all)
        y_train_tensor = torch.FloatTensor(y_train_all).unsqueeze(1)

        X_test_tensor = torch.FloatTensor(X_test_all)
        y_test_tensor = torch.FloatTensor(y_test_all).unsqueeze(1)

        # Create DataLoader for training set
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Loading pre-trained model
        model.load_state_dict(torch.load(path))
        model.to(device)

        # Adversarial training
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(advPath):
            epochs = 30
            for epoch in range(epochs):
                print(f'\t[💪 {model_name}] {epoch+1}/{epochs}', end='\r')
                train_multi_class_model(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            print()
            torch.save(model.state_dict(), advPath)
        else:
            model.load_state_dict(torch.load(advPath))

        accuracy, f1 = evaluate_multi_class_model(model, test_dataloader, device)
        print(f'\t[👑 {model_name}] Accuracy: {accuracy:.3f}, F1: {f1:.3f}')

# Evaluating models
        
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

    fcnPath = f'./models/{target_vehicle}/adversarial-training/all-vehicles/FCN.pth'
    cnnPath = f'./models/{target_vehicle}/adversarial-training/all-vehicles/CNN.pth'
    lstmPath = f'./models/{target_vehicle}/adversarial-training/all-vehicles/LSTM.pth'

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

    results_df.to_csv(f'./results/adversarial-training/all-vehicles/{target_vehicle}.csv', index=False)



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



# Online adversarial learning
print('[🔧 ONLINE]\n')

# Same vehicle, same model, all attacks
# Evaluation on everything
print('[⚔️ ALL-ATTACKS]')

# Training models
for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')

    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

    features = df.drop(['Class'], axis=1).values
    labels = df['Class'].values

    features, labels = RandomUnderSampler(random_state=seed).fit_resample(features, labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = len(df.columns) - 1
    hidden_size = 64
    output_size = 4

    fcn = FCNMultiClass(input_size, hidden_size, output_size)
    cnn = CNNMultiClass(input_size, output_size)
    lstm = LSTMMultiClass(input_size, hidden_size, output_size)

    models = [fcn, cnn, lstm]

    # Fine-tuning models
    for model, model_name in zip(models, model_names):
        if model_name == 'LSTM':
            device = torch.device('cpu')
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        advFold = f'./models/{vehicle}/adversarial-training/all-attacks/'
        
        if not os.path.exists(advFold):
            os.makedirs(advFold)

        advPath = os.path.join(advFold, f'{model_name}_online.pth')

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(advPath):
            epochs = 30
            for epoch in range(epochs):
                print(f'\t[💪 {model_name}] {epoch+1}/{epochs}', end='\r')
                online_adversarial_training(model, train_dataloader, nn.CrossEntropyLoss(), optimizer, device)
            print()
            torch.save(model.state_dict(), advPath)
        else:
            model.load_state_dict(torch.load(advPath))

        accuracy, f1 = evaluate_multi_class_model(model, test_dataloader, device)
        print(f'\t[👑 {model_name}] Accuracy: {accuracy:.3f}, F1: {f1:.3f}')

# Evaluating models
        
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

    fcnPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/FCN_online.pth'
    cnnPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/CNN_online.pth'
    lstmPath = f'./models/{target_vehicle}/adversarial-training/all-attacks/LSTM_online.pth'

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

    results_df.to_csv(f'./results/adversarial-training/all-attacks/{target_vehicle}_online.csv', index=False)



############################################################################################################



# Same vehicle, all model, all attacks
# Evaluation on everything
print('[🤖 ALL-MODELS]')

# Training models
for vehicle in vehicles:
    print(f'[🚗 VEHICLE] {vehicle}')

    datasetPath = f'./dataset/{vehicle}_multi.csv'

    df = pd.read_csv(datasetPath)
    df = df.rename(columns={'Flag': 'Class'})

    features = df.drop(['Class'], axis=1).values
    labels = df['Class'].values

    features, labels = RandomUnderSampler(random_state=seed).fit_resample(features, labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = len(df.columns) - 1
    hidden_size = 64
    output_size = 4

    fcn = FCNMultiClass(input_size, hidden_size, output_size)
    cnn = CNNMultiClass(input_size, output_size)
    lstm = LSTMMultiClass(input_size, hidden_size, output_size)

    models = [fcn, cnn, lstm]

    # Training models
    device = torch.device('cpu')
    
    advFold = f'./models/{vehicle}/adversarial-training/all-models/'
    
    if not os.path.exists(advFold):
        os.makedirs(advFold)

    advPathFCN = os.path.join(advFold, f'FCN_online.pth')
    advPathCNN = os.path.join(advFold, f'CNN_online.pth')
    advPathLSTM = os.path.join(advFold, f'LSTM_online.pth')

    advPaths = [advPathFCN, advPathCNN, advPathLSTM]

    optimizers = []
    for model in models:
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizers.append(optimizer)

    if not os.path.exists(advPathFCN) or not os.path.exists(advPathCNN) or not os.path.exists(advPathLSTM):
        epochs = 30
        for epoch in range(epochs):
            print(f'\t[💪 ALL MODELS] {epoch+1}/{epochs}', end='\r')
            online_adversarial_training_all_models(models, train_dataloader, nn.CrossEntropyLoss(), optimizers, device)
        print()
        for model, advPath in zip(models, advPaths):
            torch.save(model.state_dict(), advPath)
    else:
        for i in range(3):
            models[i].load_state_dict(torch.load(advPaths[i]))

    for model, model_name in zip(models, model_names):
        accuracy, f1 = evaluate_multi_class_model(model, test_dataloader, device)
        print(f'\t[👑 {model_name}] Accuracy: {accuracy:.3f}, F1: {f1:.3f}')

# Evaluating models
        
# Target vehicle = vehicle on which the attack is evaluated
# Source vehicle = vehicle from which the attacks are generated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    fcnPath = f'./models/{target_vehicle}/adversarial-training/all-models/FCN_online.pth'
    cnnPath = f'./models/{target_vehicle}/adversarial-training/all-models/CNN_online.pth'
    lstmPath = f'./models/{target_vehicle}/adversarial-training/all-models/LSTM_online.pth'

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

    results_df.to_csv(f'./results/adversarial-training/all-models/{target_vehicle}_online.csv', index=False)