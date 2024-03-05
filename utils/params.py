import torch



# Seed for reproducibility
seed = 151836

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path of the dataset
datasetFolder = './dataset/'

# Columns of the dataset
columns = ['Timestamp', 'CAN ID', 'DLC', 'DATA [0]', 'DATA [1]', 'DATA [2]', 'DATA [3]', 'DATA [4]', 'DATA [5]', 'DATA [6]', 'DATA [7]', 'Flag']

# Vehicles included in the dataset
vehicles = ['sonata', 'soul', 'spark']

# Names of the datasets
datasetNames = ['flooding', 'fuzzy', 'malfunction', 'no-attack']

model_names = [
    'FCN',
    'CNN',
    'LSTM'
]

epsilons = [
    0.1,
    0.2,
    0.3
]

attack_names = [
    'BIM',
    'FGSM',
    'PGD',
    'RFGSM'
]