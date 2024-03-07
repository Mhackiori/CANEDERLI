import numpy as np
import os
import random

import torch
import torchattacks
from sklearn.metrics import accuracy_score, f1_score

from .params import *



def setSeed(seed=seed):
    """
    Setting the seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setSeed()


def train_model(model, train_dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def train_multi_class_model(model, train_dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def online_adversarial_training(model, train_dataloader, criterion, optimizer, device):
    model.train()
    attacks = [
        torchattacks.BIM(model, eps=0.2),
        torchattacks.FGSM(model, eps=0.2),
        torchattacks.PGD(model, eps=0.2),
        torchattacks.RFGSM(model, eps=0.2),
    ]
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        for attack in attacks:
            adversarial_samples = attack(inputs, labels)
            adversarial_samples_outputs = model(adversarial_samples)

            attack_loss = criterion(adversarial_samples_outputs, labels)
            attack_loss.backward()

        optimizer.step()


def online_adversarial_training_all_models(models, train_dataloader, criterion, optimizers, device):
    attacks_all = []
    for model in models:
        model.train()
    
        attacks_model = [
            torchattacks.BIM(model, eps=0.2),
            torchattacks.FGSM(model, eps=0.2),
            torchattacks.PGD(model, eps=0.2),
            torchattacks.RFGSM(model, eps=0.2),
        ]
        attacks_all.append(attacks_model)

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        for attacks in attacks_all:
            for attack in attacks:
                adversarial_samples = attack(inputs, labels)
                for model in models:
                    adversarial_samples_outputs = model(adversarial_samples)

                    attack_loss = criterion(adversarial_samples_outputs, labels)
                    attack_loss.backward()

            optimizer.step()


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)

    return accuracy, f1


def evaluate_multi_class_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='macro')

    return accuracy, f1


def attacks_evaluation(df, bw='white'):
    """
    Takes in input a list of DataFrames and white/back-box type evaluation
    """

    models_acc = []
    models_f1 = []

    attacks_acc = []
    attacks_f1 = []

    accuracy = []
    f1 = []

    if bw == 'white':
        sub = df[df['Target_Vehicle'] == df['Source_Vehicle']]
        sub = sub[sub['Target_Model'] == sub['Source_Model']]
    elif bw == 'black':
        sub = df[df['Target_Vehicle'] != df['Source_Vehicle']]
        sub = sub[sub['Target_Model'] != sub['Source_Model']]    
    else:
        raise ValueError('bw must be either white or black')

    for model_name in model_names:
        model_df = sub[sub['Target_Model'] == model_name]

        models_acc.append(model_df['Accuracy'].mean()) 
        models_f1.append(model_df['F1'].mean())

    for attack_name in attack_names:
        attack_df = sub[sub['Attack'] == attack_name]

        attacks_acc.append(attack_df['Accuracy'].mean())
        attacks_f1.append(attack_df['F1'].mean())
        
    accuracy.append(sub['Accuracy'].mean())
    f1.append(sub['F1'].mean())
    

    return models_acc, models_f1, attacks_acc, attacks_f1, np.mean(accuracy), np.mean(f1)