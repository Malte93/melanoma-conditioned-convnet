import torch
import os
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

class MelanomaDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, split: str = 'Train',
                 transform: transforms.Compose = None, train: bool = None, seed: int = 42) -> None:
        """
        Initalizes the dataset (either train, val or test).
        
        Parameters
        -------------
            csv_file : str
                Represents the path to the meta .csv file
                (contains information about the images and the corresponding patients).
            root_dir : str
                Represents the directory path where all images are stored.
            split : str
                Split affects only the training dataset. If it is set to 'Train'
                we are getting the training data from our original training dataset.
                If we set it to 'Val' however, we are getting the validation set
                from our original training dataset.
            transform : transforms.Compose
                Transformation (i.e. data augmentation or totensor operation)
                which is applied on a batch of images if it is defined.
            train : bool
                Determines whether the current run is in training or test mode.
                And therefore returns the labels if and only if the current run
                is in training mode.
            seed : int
                Random seed for the randomizer Module from Numpy
                (for the purpose of reproducibility).
        """
        self.meta = pd.read_csv(csv_file)
        self.split = split
        if self.split == 'Train':
            self.meta = self.meta.loc[self.meta['val'] == 0].reset_index(drop=True) # 0 indicates training data
            np.random.seed(seed)
        elif self.split == 'Val':
            self.meta = self.meta.loc[self.meta['val'] == 1].reset_index(drop=True) # 1 indicates validation data
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.seed = seed

    def get_class_sizes(self) -> tuple[int, int]:
        """
        Determines the number of benign and malignant samples.
        
        Returns
        -------------
            benign : int
                The number of lesions which are benign.
            malignant : int
                The number of lesions which are malignant.
        """
        benign, malignant = self.meta.groupby('target').count()['image_name'].tolist()

        return benign, malignant

    def get_labels(self) -> list:
        """
        Determines the labels for the dataset and returns them.

        Returns
        -------------
            labels : list
                The ground truth labels for the dataset.
        """
        labels = self.meta['target'].tolist()

        return labels

    def __len__(self) -> int:
        """
        The length of the dataset (number of samples within the dataset).
        
        Returns
        -------------
            len(self.meta) : int
                The number of samples within the dataset.
        """
        return len(self.meta)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a sample from the dataset.

        Parameters
        -------------
            idx : int
                Index of the sample which shall be returned.

        Returns
        -------------
            image_rgb : torch.tensor
                The lesion image which shall be classified.
            label : torch.tensor
                The ground truth labels (but only if train == True since they are only defined in the training dataset) of image_rgb.
            auxiliary_image_rgb : torch.tensor
                The conditional information (image) of the same patient as the lesion image image_rgb.
            image_name : str
                Name of the image which shall be classified i.e., ISIC_xxxxxxx_IP.
            auxiliary_image_name : str 
                Name of the image which serves as the contextual information i.e., ISIC_xxxxxxx_IP.
        """
        if self.split != 'Train':
            np.random.seed(self.seed)
        if torch.is_tensor(idx):
            idx = idx.tolistist()
        # Get label if available for the current phase
        if self.train == True:
            label = self.meta.loc[idx, 'target']
        img_path = os.path.join(self.root_dir, str(self.meta.iloc[idx, 0] + '.jpg'))
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Determine auxiliary input
        # Get the image_name and the_patient id of the above image
        image_name, patient_id = self.meta.iloc[idx, 0:2]
        # Get images of the same patient which shall be classified (but not the same image as above)
        image_names = self.meta.loc[(self.meta['patient_id'] == patient_id) &
                                    (self.meta['image_name'] != image_name)]['image_name']
        # Pick one of those images as the auxiliiary image
        auxiliary_image_name = np.random.choice(image_names)
        auxiliary_image_path = os.path.join(self.root_dir, str(auxiliary_image_name + '.jpg'))
        auxiliary_image = cv2.imread(auxiliary_image_path)
        auxiliary_image_rgb = cv2.cvtColor(auxiliary_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image_rgb = self.transform(image_rgb)
            auxiliary_image_rgb = self.transform(auxiliary_image_rgb)

        if self.train == True:
            return image_rgb, label, auxiliary_image_rgb, image_name, auxiliary_image_name
        else:
            return image_rgb, auxiliary_image_rgb, image_name, auxiliary_image_name