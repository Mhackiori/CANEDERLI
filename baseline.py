import numpy as np
import os
import pandas as pd
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    datasetPath = f'./dataset/{vehicle}.csv'

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

    ### FCN ###

    input_size_fcn = len(df.columns) - 1
    hidden_size_fcn = 64
    output_size_fcn = 1
    model_fcn = FCN(input_size_fcn, hidden_size_fcn, output_size_fcn).to(device)

    optimizer_fcn = optim.Adam(model_fcn.parameters(), lr=0.001)

    fcnPath = f'./models/{vehicle}/FCN.pth'

    if not os.path.exists(fcnPath):
        epochs_fcn = 30
        for epoch in range(epochs_fcn):
            print(f'\t[💪 FCN] {epoch+1}/{epochs_fcn}', end='\r')
            train_model(model_fcn, train_dataloader, nn.BCELoss(), optimizer_fcn, device)
        print()
        torch.save(model_fcn.state_dict(), fcnPath)
    else:
        model_fcn.load_state_dict(torch.load(fcnPath))

    accuracy_fcn, f1_fcn = evaluate_model(model_fcn, test_dataloader, device)
    print(f'\t[👑 FCN] Accuracy: {accuracy_fcn:.3f}, F1: {f1_fcn:.3f}')

    ### CNN ###

    input_size_cnn = len(df.columns) - 1
    output_size_cnn = 1
    model_cnn = CNN(input_size_cnn, output_size_cnn).to(device)

    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

    cnnPath = f'./models/{vehicle}/CNN.pth'

    if not os.path.exists(cnnPath):
        epochs_cnn = 30
        for epoch in range(epochs_cnn):
            print(f'\t[💪 CNN] {epoch+1}/{epochs_cnn}', end='\r')
            train_model(model_cnn, train_dataloader, nn.BCELoss(), optimizer_cnn, device)
        print()
        torch.save(model_cnn.state_dict(), cnnPath)
    else:
        model_cnn.load_state_dict(torch.load(cnnPath))

    accuracy_cnn, f1_cnn = evaluate_model(model_cnn, test_dataloader, device)
    print(f'\t[👑 CNN] Accuracy: {accuracy_cnn:.3f}, F1: {f1_cnn:.3f}')

    ### LSTM ###
    input_size_lstm = len(df.columns) - 1
    hidden_size_lstm = 64
    output_size_lstm = 1
    model_lstm = LSTM(input_size_lstm, hidden_size_lstm, output_size_lstm).to(device)
    
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

    lstmPath = f'./models/{vehicle}/LSTM.pth'

    if not os.path.exists(lstmPath):
        epochs_lstm = 30
        for epoch in range(epochs_lstm):
            print(f'\t[💪 LSTM] {epoch+1}/{epochs_lstm}', end='\r')
            train_model(model_lstm, train_dataloader, nn.BCELoss(), optimizer_lstm, device)
        print()
        torch.save(model_lstm.state_dict(), lstmPath)
    else:
        model_lstm.load_state_dict(torch.load(lstmPath))

    accuracy_lstm, f1_lstm = evaluate_model(model_lstm, test_dataloader, device)
    print(f'\t[👑 LSTM] Accuracy: {accuracy_lstm:.3f}, F1: {f1_lstm:.3f}')

print()

# Multiclass
for vehicle in vehicles:
    print(f'[🚗 MULTICLASS VEHICLE] {vehicle}')
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

    ### FCN ###

    input_size_fcn = len(df.columns) - 1
    hidden_size_fcn = 64
    output_size_fcn = 4
    model_fcn = FCNMultiClass(input_size_fcn, hidden_size_fcn, output_size_fcn).to(device)

    optimizer_fcn = optim.Adam(model_fcn.parameters(), lr=0.001)

    fcnPath = f'./models/{vehicle}/FCN_multi.pth'

    if not os.path.exists(fcnPath):
        epochs_fcn = 30
        for epoch in range(epochs_fcn):
            print(f'\t[💪 MULTI FCN] {epoch+1}/{epochs_fcn}', end='\r')
            train_multi_class_model(model_fcn, train_dataloader, nn.CrossEntropyLoss(), optimizer_fcn, device)
        print()
        torch.save(model_fcn.state_dict(), fcnPath)
    else:
        model_fcn.load_state_dict(torch.load(fcnPath))

    accuracy_fcn, f1_fcn = evaluate_multi_class_model(model_fcn, test_dataloader, device)
    print(f'\t[👑 MULTI FCN] Accuracy: {accuracy_fcn:.3f}, F1: {f1_fcn:.3f}')

    ### CNN ###

    input_size_cnn = len(df.columns) - 1
    output_size_cnn = 4
    model_cnn = CNNMultiClass(input_size_cnn, output_size_cnn).to(device)

    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

    cnnPath = f'./models/{vehicle}/CNN_multi.pth'

    if not os.path.exists(cnnPath):
        epochs_cnn = 30
        for epoch in range(epochs_cnn):
            print(f'\t[💪 MULTI CNN] {epoch+1}/{epochs_cnn}', end='\r')
            train_multi_class_model(model_cnn, train_dataloader, nn.CrossEntropyLoss(), optimizer_cnn, device)
        print()
        torch.save(model_cnn.state_dict(), cnnPath)
    else:
        model_cnn.load_state_dict(torch.load(cnnPath))

    accuracy_cnn, f1_cnn = evaluate_multi_class_model(model_cnn, test_dataloader, device)
    print(f'\t[👑 MULTI CNN] Accuracy: {accuracy_cnn:.3f}, F1: {f1_cnn:.3f}')

    ### LSTM ###
    input_size_lstm = len(df.columns) - 1
    hidden_size_lstm = 64
    output_size_lstm = 4
    model_lstm = LSTMMultiClass(input_size_lstm, hidden_size_lstm, output_size_lstm).to(device)
    
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

    lstmPath = f'./models/{vehicle}/LSTM_multi.pth'

    if not os.path.exists(lstmPath):
        epochs_lstm = 30
        for epoch in range(epochs_lstm):
            print(f'\t[💪 MULTI LSTM] {epoch+1}/{epochs_lstm}', end='\r')
            train_multi_class_model(model_lstm, train_dataloader, nn.CrossEntropyLoss(), optimizer_lstm, device)
        print()
        torch.save(model_lstm.state_dict(), lstmPath)
    else:
        model_lstm.load_state_dict(torch.load(lstmPath))

    accuracy_lstm, f1_lstm = evaluate_multi_class_model(model_lstm, test_dataloader, device)
    print(f'\t[👑 MULTI LSTM] Accuracy: {accuracy_lstm:.3f}, F1: {f1_lstm:.3f}')