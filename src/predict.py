import os
import argparse
import torch
from film_ed import FiLMedNetwork
from melanoma_dataset import MelanomaDataset
from torch.utils.data import DataLoader
from utils import utils

class Prediction():
    """
    This class tests a trained model on the testing dataset and predicts the
    class for each sample. The predictions will be stored in a .csv. To determine the
    AUROC value - upload the .csv file on https://challenge.isic-archive.com/landing/live/
    (already closed for uploads). Alternatively, upload the .csv file on the
    corresponding Kaggle site: https://www.kaggle.com/c/siim-isic-melanoma-classification/overview.
    """
    def __init__(self, path_dataset: str, path_model: str) -> None:
        """
        Initializes the prediction class.

        Parameters
        -------------
            path_dataset : str
                The path to the dataset directory.
            path_model : str
                The path to the trained model.
        """
        # Load test dataset
        test_dataset = MelanomaDataset(os.path.join(path_dataset, os.path.normpath('test.csv')),
                                  os.path.join(path_dataset, os.path.normpath('test')), train=False)

        # Load test dataset with the DataLoader class
        self.testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Setup FiLM-ed network
        self.net = FiLMedNetwork()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path_model, map_location=device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(device)

    def test(self) -> None:
        """
        Tests the trained model on the test dataset.
        """
        # Track probability of being malignant and the according image name
        prob_list = []
        img_names_list = []

        # Run through the whole test dataset and compute the probabilities of being malignant
        self.net.eval()
        with torch.no_grad():
            for data in self.testloader:
                img, auxiliary_img, img_names = data
                outputs = self.net((img, auxiliary_img))
                probs = torch.sigmoid(outputs)
                prob_list.append(probs)
                img_names_list.append(img_names)

        # Prepare prob_list and img_names_list and call create_submission_file to create a submission file
        prob_list = [item.item() for tensor in prob_list for item in tensor]
        rows = list(map(list, zip(img_names_list, prob_list)))
        utils.create_submission_file(self, self.model_path, rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', help='The path where your model weights are saved. This path should be defined as an absolut path and as string, e.g.: "--model_path /Your/Path/Here/Experiments/.../model.pt".', type=str, required=True)
    parser.add_argument('--root_directory', help='This path shall be specified as an absolut path and as a string. The root directory contains the test dataset, e.g.: "--root_directory /Your/Path/Here/Dataset/".', type=str, required=True)
    args = parser.parse_args()
    
    path_model = os.path.normpath(args.path_model)
    path_directory = os.path.normpath(args.root_directory)
    if os.path.exists(path_model) and os.path.exists(path_directory):
        prediction = Prediction(path_directory, path_model)
        prediction.test()
    else:
        raise Exception(f"The path to your trained model {path_model} and / or the path to your root directory {path_directory} does not exist.")