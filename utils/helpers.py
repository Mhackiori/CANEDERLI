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