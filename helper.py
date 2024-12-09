ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
import pandas as pd
import sys
import os
sys.path.append(f'{ROOT_DIR}/code/run_models')
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TABULAR = ['Synthetic', 'Credit', 'Weather']
CLASS_ADJUST = ['EMNIST', 'CIFAR']
SQUEEZE = ['Synthetic', 'Credit']
LONG = ['EMNIST', 'CIFAR', 'ISIC']
CLASS_ADJUST = ['EMNIST', 'CIFAR']
TENSOR = ['IXITiny']
CONTINUOUS_OUTCOME = ['Weather']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parameters_for_dataset(DATASET):
    default_params = {
        'Synthetic': {'rounds': 20, 'batch_size': 2000, 'runs': 500, 'runs_lr': 5, 'metric': 'F1'},
        'Credit': {'rounds': 20, 'batch_size': 2000, 'runs': 500, 'runs_lr': 5, 'metric': 'F1'},
        'Weather': {'rounds': 20, 'batch_size': 4000, 'runs': 500, 'runs_lr': 5, 'metric': 'R2'},
        'EMNIST': {'rounds': 20, 'batch_size': 5000, 'runs': 50, 'runs_lr': 3, 'metric': 'Accuracy'},
        'CIFAR': {'rounds': 20, 'batch_size': 256, 'runs': 20, 'runs_lr': 3, 'metric': 'Accuracy'},
        'IXITiny': {'rounds': 10, 'batch_size': 12, 'runs': 3, 'runs_lr': 1, 'metric': 'DICE'},
        'ISIC': {'rounds': 10, 'batch_size': 128, 'runs': 3, 'runs_lr': 1, 'metric': 'Balanced_accuracy'}
    }
    params = default_params[DATASET]
    DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
    return DATA_DIR, params

def get_default_lr(DATASET):
    default_params = {
        'Synthetic': 5e-3,
        'Credit': 5e-3,
        'Weather': 5e-3,
        'EMNIST': 5e-3,
        'CIFAR': 1e-3,
        'IXITiny': 1e-3,
        'ISIC': 1e-3
    }
    return default_params[DATASET]

def get_default_reg(DATASET):
    default_params = {
        'Synthetic': 1e-1,
        'Credit': 1e-1,
        'Weather': 1e-1,
        'EMNIST': 5e-2,
        'CIFAR': 5e-2,
        'IXITiny': 5e-2,
        'ISIC': 5e-2
    }
    return default_params[DATASET]

def sample_per_class(labels, class_size=500):
    df = pd.DataFrame({'labels': labels})
    df_stratified = df.groupby('labels').apply(lambda x: x.sample(class_size, replace=False))
    return df_stratified.index.get_level_values(1)


def get_common_name(full_path):
    return os.path.basename(full_path).split('_')[0]


def align_image_label_files(image_files, label_files):
    labels_dict = {get_common_name(path): path for path in label_files}
    images_dict = {get_common_name(path): path for path in image_files}
    
    common_keys = sorted(set(labels_dict.keys()) & set(images_dict.keys()))
    return [images_dict[key] for key in common_keys], [labels_dict[key] for key in common_keys]



def loadData(DATASET, DATA_DIR, data_num, cost):
    try:
        if DATASET in TABULAR:
            # Load tabular data
            column_counts = {'Synthetic': 13, 'Credit': 29, 'Weather': 124}
            sample_sizes = {'Synthetic': 800, 'Credit': 800, 'Weather': 1600}

            file_path = f'{DATA_DIR}/data_{data_num}_{cost:.2f}.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            X = pd.read_csv(file_path, sep=' ', names=[i for i in range(column_counts[DATASET])])
            X = X.sample(sample_sizes[DATASET], replace=(DATASET == 'Credit'))

            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
            return X.values, y.values

        elif DATASET in CLASS_ADJUST:
            # Load class-adjusted data
            file_path = f'{DATA_DIR}/data_{data_num}_{cost:.2f}.npz'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            data = np.load(file_path)
            X, y = data['data'], data['labels']
            class_size = 250

            ind = sample_per_class(y, class_size)
            X_sample, y_sample = X[ind], y[ind]

            unique_labels = np.unique(y_sample)
            y_sample_mapped = np.vectorize({label: idx for idx, label in enumerate(unique_labels)}.get)(y_sample)
            return X_sample, y_sample_mapped

        elif DATASET == 'IXITiny':
            # Load IXITiny data
            sites = {
                0.08: [['Guys'], ['HH']], 
                0.28: [['IOP'], ['Guys']], 
                0.30: [['IOP'], ['HH']],
                'all': [['IOP'], ['HH'], ['Guys']]}
            site_names = sites[cost][data_num - 1]
            image_files, label_files = [], []

            image_dir = os.path.join(DATA_DIR, 'flamby/image')
            label_dir = os.path.join(DATA_DIR, 'flamby/label')
            for name in site_names:
                image_files.extend([f'{image_dir}/{file}' for file in os.listdir(image_dir) if name in file])
                label_files.extend([f'{label_dir}/{file}' for file in os.listdir(label_dir) if name in file])

            image_files, label_files = align_image_label_files(image_files, label_files)
            return np.array(image_files), np.array(label_files)

        elif DATASET == 'ISIC':
            # Load ISIC data
            dataset_pairings = {
                0.06: (2, 2), 
                0.15: (2, 0), 
                0.19: (2, 3), 
                0.25: (2, 1), 
                0.3: (1, 3),
                'all': (0,1,2,3)}
            site = dataset_pairings[cost][data_num - 1]

            file_path = f'{DATA_DIR}/site_{site}_files_used.csv'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            files = pd.read_csv(file_path, nrows=2000)
            image_files = [f'{DATA_DIR}/ISIC_2019_Training_Input_preprocessed/{file}.jpg' for file in files['image']]
            return np.array(image_files), files['label'].values
    except Exception as e:
        raise ValueError(f"Error loading data for dataset '{DATASET}': {e}")


def get_dice_loss(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return torch.mean(1 - dice_score)
