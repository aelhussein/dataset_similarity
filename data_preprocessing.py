from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
import albumentations
import random
from monai.transforms import EnsureChannelFirst, AsDiscrete,Compose,NormalizeIntensity,Resize,ToTensor

global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
DATASET_TYPES_TABULAR = {'Synthetic', 'Credit', 'Weather'}
DATASET_TYPES_IMAGE = {'CIFAR', 'EMNIST', 'IXITiny', 'ISIC'}
CONTINUOUS_OUTCOME = {'Weather'}
LARGE_TEST_SET = {'Synthetic', 'Credit', 'Weather', 'CIFAR', 'EMNIST'}
torch.manual_seed(1)
np.random.seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDataset(Dataset):
    """Base class for all datasets"""
    def __init__(self, X, y, is_train, **kwargs):
        self.X = X
        self.y = y
        self.is_train = is_train
        self.transform = self.get_transform()
 
    def __len__(self):
        return len(self.X)
    
    def get_transform(self):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def get_scalers(self):
        """Return any parameters that need to be shared with val/test datasets"""
        return {}

class BaseTabularDataset(BaseDataset):
    """Base class for tabular datasets"""
    def __init__(self, X, y, is_train, **kwargs):
        self.scalers = kwargs.get('scalers', {
            'feature_scaler': StandardScaler(),
            'label_scaler': StandardScaler()
        })
        super().__init__(X, y, is_train, **kwargs)

    def get_transform(self):
        if self.is_train:
            return lambda X, y: (
                self.scalers['feature_scaler'].fit_transform(X),
                y
            )
        return lambda X, y: (
            self.scalers['feature_scaler'].transform(X),
            y
        )
    
    def __getitem__(self, idx):
        X_transformed, y = self.transform(
            self.X[idx:idx+1], 
            self.y[idx:idx+1]
        )
        return (
            torch.tensor(X_transformed, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
    
    def get_scalers(self):
        return self.scalers

class SyntheticDataset(BaseTabularDataset):
    """Dataset handler for synthetic data with categorical outcomes"""
    pass  # Inherits all functionality from BaseTabularDataset

class CreditDataset(BaseTabularDataset):
    """Dataset handler for credit data with categorical outcomes"""
    pass  # Inherits all functionality from BaseTabularDataset

class WeatherDataset(BaseTabularDataset):
    """Dataset handler for weather data with continuous outcomes"""    
    def get_transform(self):
        if self.is_train:
            return lambda X, y: (
                self.scalers['feature_scaler'].fit_transform(X),
                self.scalers['label_scaler'].fit_transform(y.reshape(-1, 1))
            )
        return lambda X, y: (
            self.scalers['feature_scaler'].transform(X),
            self.scalers['label_scaler'].transform(y.reshape(-1, 1))
        )

class BaseImageDataset(BaseDataset):
    """Base class for image datasets"""
    def __init__(self, X, y, is_train, **kwargs):

        super().__init__(X, y, is_train,  **kwargs)
        self.transform = self.get_transform()

class EMNISTDataset(BaseImageDataset):
    """EMNIST dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ]
        
        if self.is_train:
            augmentation = [
                transforms.RandomRotation((-15, 15)),
                transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomErasing(p=0.1)
            ]
            return transforms.Compose(augmentation + base_transform)
        
        return transforms.Compose(base_transform)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

class CIFARDataset(BaseImageDataset):
    """CIFAR-100 dataset handler"""
    def get_transform(self):
        base_transform = [
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ]
        
        if self.is_train:
            augmentation = [
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.RandomErasing(p=0.1),
                transforms.RandomRotation(15)
            ]
            return transforms.Compose(augmentation + base_transform)
        
        return transforms.Compose(base_transform)

class ISICDataset(BaseImageDataset):
    """ISIC dataset handler for skin lesion images"""
    def __init__(self, image_paths, labels, is_train):
        self.sz = 200  # Image size
        super().__init__(image_paths, labels, is_train)
    
    def get_transform(self):
        if self.is_train:
            return albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.Flip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.RandomCrop(self.sz, self.sz),
                albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])
        else:
            return albumentations.Compose([
                albumentations.CenterCrop(self.sz, self.sz),
                albumentations.Normalize(
                    mean=(0.585, 0.500, 0.486),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True
                ),
            ])

    def __getitem__(self, idx):
        image_path = self.X[idx] 
        label = self.y[idx]

        # Read image as numpy array for albumentations
        image = np.array(Image.open(image_path))
        
        # Apply albumentations transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.tensor(label, dtype=torch.int64)
        
        return image, label

class IXITinyDataset(BaseImageDataset):
    """IXITiny dataset handler for 3D medical images"""
    def __init__(self, image_paths, label_paths, is_train):
        self.common_shape = (48, 60, 48)
        super().__init__(image_paths, label_paths, is_train)
        self.label_transform = self._get_label_transform()
    
    def get_transform(self):
        """Use the same transform for both training and validation"""
        default_transform = Compose([
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape)
        ])
        
        intensity_transform = Compose([
            NormalizeIntensity()
        ])
        
        return lambda x: intensity_transform(default_transform(x))
    
    def _get_label_transform(self):
        """Transform for labels in medical imaging"""
        default_transform = Compose([
            ToTensor(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(self.common_shape)
        ])
        
        one_hot_transform = Compose([
            AsDiscrete(to_onehot=2)
        ])
    
        return lambda x: one_hot_transform(default_transform(x))

    def __getitem__(self, idx):
        image_path = self.X[idx]    
        label_path = self.y[idx]    
    
        # Load 3D medical images
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Apply transforms
        image = self.transform(image)
        label = self.label_transform(label)
        
        return image.to(torch.float32), label


class DataPreprocessor:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_class = self._get_dataset_class()
        
    def _get_dataset_class(self):
        dataset_classes = {
            'Synthetic': SyntheticDataset,
            'Credit': CreditDataset,
            'Weather': WeatherDataset,
            'EMNIST': EMNISTDataset,
            'CIFAR': CIFARDataset,
            'IXITiny': IXITinyDataset,
            'ISIC': ISICDataset
        }
        return dataset_classes[self.dataset_name]

    def process_clients(self, client_data):
        """Process data for multiple clients and create a joint dataset."""
        processed_data = {}
        
        # Process individual client data
        for client_id, data in client_data.items():
            processed_data[client_id] = self._process_single_client(data)
            
        # Process joint data
        joint_data = self._combine_client_data(client_data)
        processed_data['client_joint'] = self._process_single_client(joint_data)
            
        return processed_data

    def _process_single_client(self, data):
        """Process data for a single client."""
        # Split data into train/val/test
        train_data, val_data, test_data = self._split_data(data['X'], data['y'])
        
        # Create train dataset first
        train_dataset = self.dataset_class(train_data[0], train_data[1], is_train=True)
        
        # Get scalers if the dataset uses them
        scalers = train_dataset.get_scalers()
        
        # Create val and test datasets
        val_dataset = self.dataset_class(
            val_data[0], 
            val_data[1], 
            is_train=False, 
            **scalers
        )
        test_dataset = self.dataset_class(
            test_data[0], 
            test_data[1], 
            is_train=False, 
            **scalers
        )
        
        # Create and return dataloaders
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        )
    
    def _combine_client_data(self, client_data):
        """Combine data from multiple clients into a single dataset."""
        combined_X = np.concatenate([data['X'] for data in client_data.values()])
        combined_y = np.concatenate([data['y'] for data in client_data.values()])
        return {'X': combined_X, 'y': combined_y}

    def _split_data(self, X, y):
        test_size = 0.6 if self.dataset_name in LARGE_TEST_SET else 0.2
        val_size = 0.2

        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=np.random.RandomState(42)
        )

        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=np.random.RandomState(42)
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)