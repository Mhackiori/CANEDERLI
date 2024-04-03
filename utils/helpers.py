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


def online_adversarial_training(model, train_dataloader, criterion, optimizer, device, epoch, epochs, model_name):
    model.train()
    max_eps = 0.2
    num_batches = len(train_dataloader)
    for i, (inputs, labels) in enumerate(train_dataloader):
        print(f'\t[💪 {model_name}] {epoch+1}/{epochs} | {i+1}/{num_batches}', end='\r')
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_eps = max_eps * (i/num_batches)

        attacks = [
            torchattacks.BIM(model, eps=epoch_eps),
            torchattacks.FGSM(model, eps=epoch_eps),
            torchattacks.PGD(model, eps=epoch_eps),
            torchattacks.RFGSM(model, eps=epoch_eps),
        ]

        for attack in attacks:
            adversarial_samples = attack(inputs, labels)
            adversarial_samples_outputs = model(adversarial_samples)

            attack_loss = criterion(adversarial_samples_outputs, labels)
            attack_loss.backward()
            optimizer.step()


def online_adversarial_training_all_models(models, train_dataloader, criterion, optimizers, device):
    num_batches = len(train_dataloader)
    max_eps = 0.2
    for model in models:
        model.train()

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.squeeze(1).long().to(device)
        
        all_adversarial_samples = []
        all_adversarial_labels = []

        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_eps = max_eps * (i/num_batches)

            attacks = [
                torchattacks.BIM(model, eps=epoch_eps),
                torchattacks.FGSM(model, eps=epoch_eps),
                torchattacks.PGD(model, eps=epoch_eps),
                torchattacks.RFGSM(model, eps=epoch_eps),
            ]

            model_adversarial_samples = []
            model_adversarial_labels = []
            for attack in attacks:
                adversarial_samples = attack(inputs, labels)
                model_adversarial_samples.append(adversarial_samples)
                model_adversarial_labels.append(labels)

            all_adversarial_samples.append(model_adversarial_samples)
            all_adversarial_labels.append(model_adversarial_labels)

        for model_adversarial_sample, model_adversarial_label in zip(all_adversarial_samples, all_adversarial_labels):
            for adversarial_sample, adversarial_label in zip(model_adversarial_sample, model_adversarial_label):
                for model, optimizer in zip(models, optimizers):
                    adversarial_samples_outputs = model(adversarial_sample)

                    attack_loss = criterion(adversarial_samples_outputs, adversarial_label)
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