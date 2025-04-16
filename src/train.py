import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from validation import val_step
from film_ed import FiLMedNetwork
from melanoma_dataset import MelanomaDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

class train():
    def __init__(self, hyperparameter: dict, path_dataset: str,
                 path_experiment: str, seed: int, args: argparse.Namespace) -> None:
        """
        Initializes all components for training a FiLM network with contextual information.

        Parameters
        -------------
            hyperparameter : dict
                Includes all required hyperparameters for the current training run,
                which are specified via the console.
            path_dataset : str
                Absolute path to the isic-2020 dataset. Within the root
                directory there should be two subdirectories (train and test)
                with lesion images in .jpg format and two meta files
                (train.csv and test.csv).
            path_experiment : str
                Absolute path where the experiment results and model dict shall
                be stored.
            seed : int
                Seed for numpy and torch results such that they can be
                reproduced.
            args : argparse.Namespace
                Defined arguments.
        """
        # args contains all positional keyword arguments defined in the console
        self.args = args 
        # Declare and initialize hyperparameters which were defined in the console
        self.hparams = hyperparameter

        # Global seed for reproducibility and deterministic CUDNN algorithms
        self.set_reproudcibility(seed)

        # Get transformations for training and validation dataset
        self.compound_transformation_train, self.compound_transformation_val = self.get_compound_transformation()

        # Load training and validation dataset
        training = MelanomaDataset(os.path.join(path_dataset, 'train.csv'),
                                   os.path.join(path_dataset, os.path.normpath('train/')),
                                   train=True,
                                   transform=self.compound_transformation_train,
                                   seed=self.args.seed)
        validation = MelanomaDataset(os.path.join(path_dataset, 'train.csv'),
                                     os.path.join(path_dataset, os.path.normpath('train/')),
                                     split='Val',
                                     train=True, # Set train to True because we are still in a training loop and need the ground truth labels for each image
                                     transform=self.compound_transformation_val,
                                     seed=self.args.seed)
        # Setup dataset for WeightedRandomSampler due the imbalanced dataset and load train and val dataset with the DataLoader class
        if self.args.weighted_random_sampler:
            benign, malignant = training.get_class_sizes()
            targets = training.get_labels()
            sampler = utils.get_sampler(targets, benign, malignant)
            self.trainloader = DataLoader(training, batch_size=self.args.batch_size, sampler=sampler, num_workers=2)
        else:
            self.trainloader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=2)
        self.valloder = DataLoader(validation, batch_size=self.args.batch_size, shuffle=False, num_workers=2)

        # Declare all available optimizers and loss functions
        optimizers = {'SGD' : optim.SGD,
                      'Adam' : optim.Adam}
        loss_functions = {'BCEWithLogitLoss' : torch.nn.BCEWithLogitsLoss}

        # Setup the FiLM-ed network and pass a list with integer values. 
        # Those values define in which stage of the FiLM-ed network 
        # film-layers shall be present.
        self.net = FiLMedNetwork(self.args)
    
        # Get the device on which the model should be trained on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Allocate an accelearator if available, else use cpu
        self.net.to(self.device)

        # Get optimizer from predefined list
        optimizer = optimizers[self.args.optimizer]
        optimizers_parameter = dict()
        if self.args.learning_rate:
            optimizers_parameter['lr'] = self.args.learning_rate
        if self.args.weight_decay:
            optimizers_parameter['weight_decay'] = self.args.weight_decay
        # Initialize the optimizer with the defined optimizer_parameter dictionary
        self.optimizer = optimizer(self.net.parameters(), **optimizers_parameter)

        # Get loss function from predefined list
        criterion = loss_functions[self.args.loss_function]
        # Initialize loss function from predefined list
        self.criterion = criterion()

        self.path_experiment = path_experiment
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.path_experiment, 'results/'))
        # Log hyperparameters for epoch 0 to the TensorBoard
        utils.hparams_to_tensorboard(self.writer, epoch=-1, hparams=self.hparams, metricparams=utils.get_initial_state_metric()) # Why -1? Because we want to track epoch 0 and within the called function we apply epoch += 1
        
    def train(self) -> None:
        """
        Training the classifier for N epochs.
        """
        # Keep track of the best area under the receiver operating characteristic curve value (AUROC).
        best_roc_auc = 0.0

        for epoch in range(self.hparams['epochs']):
            # Train the network for the current epoch
            training_results = self.train_step(epoch)
            # Validate the network for the current epoch
            val_roc_auc, validation_results = val_step(self.net, self.valloder, epoch, self.device, self.criterion)

            utils.to_tensorboard(self.net, self.writer, training_results, validation_results, self.hparams, epoch)
            utils.to_csv(training_results, self.path_experiment, 'training.csv', epoch)
            utils.to_csv(validation_results, self.path_experiment, 'validation.csv', epoch)

            # Save the model with the best val roc auc value and the model from the last epoch
            if best_roc_auc < val_roc_auc:
                best_roc_auc = val_roc_auc
                utils.save_model(self.path_experiment, f'Model/best_model.pt', self.net, self.optimizer, training_results, validation_results, epoch)
            if self.args.epochs == epoch+1:
                utils.save_model(self.path_experiment, f'Model/last_model.pt', self.net, self.optimizer, training_results, validation_results, epoch)

        # Close SummaryWriter
        self.writer.close()

        print("Training process done")

    def train_step(self, epoch: int) -> dict:
        """
        Training a model for one epoch.

        Parameters
        -------------
            epoch : int
                Current epoch of the training cycle.

        Returns
        -------------
            training_results : dict
                Contains:
                loss, accuracy, confusion matrix, normalized confusion matrix, area under the receiver operating characteristic curve value,
                f1 metric score, ground truth values, predicted values, image names and auxiliary image names
        """
        # Set model into training mode
        self.net.train()

        # Initialize variables for tracking stats
        running_loss = 0.0
        correct_classified = 0
        # track amount of data since the DataLoader does not use drop_last
        amount_data = 0
        # Track targets, pred_probs, img_names and auxiliar_image_names to store it in a .csv file
        targets = []
        pred_probs = []
        img_names = []
        auxiliary_image_names = []
        # Empty confusion matrix
        cfm_training = np.zeros(shape=(2, 2))
        # Train one epoch
        for i, data in enumerate(self.trainloader, 0):
            # Unwrap data and assign them to the determined device
            img, labels, auxiliary_img, image_name, auxiliary_image_name = data
            img, labels, auxiliary_img = img.to(self.device), labels.to(self.device), auxiliary_img.to(self.device)
            labels = torch.unsqueeze(labels, 1).to(torch.float32)

            # Set the gradients to zero, compute the logit and the loss
            self.optimizer.zero_grad()
            outputs = self.net(img, auxiliary_img)
            loss = self.criterion(outputs, labels)

            # Calculate the gradients and move to the steepest descent (local minima)
            loss.backward()
            self.optimizer.step()

            # Compute the model prediction with logit (model output) - threshold .5
            probs = torch.sigmoid(outputs)
            prediction = torch.round(probs)

            # Track ground truth labels, predictions for roc_auc and image names
            targets.append(labels)
            pred_probs.append(probs)
            img_names.append(image_name)
            auxiliary_image_names.append(auxiliary_image_name)

            # Add the computational graph of the model to the tensorboard
            utils.graph_to_tensorboard(self.net, self.writer, img, auxiliary_img, epoch, i)

            # Compute statistical metrics
            amount_data += labels.size()[0]
            running_loss += loss.item() * img.size(0)
            correct_classified += torch.sum(labels.cpu() == prediction.cpu()).item()
            # The purpose for the argument labels=[1, 0] is that the order of occurence will be switched: now it is [[TP],[FN],
                                                                                                                    #[FP], [TN]] source: https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
            cfm_training += confusion_matrix([k.item() for k in labels], [k.item() for k in prediction], labels=[1, 0])

        # Flatten list of targets and predictions tensors into a single list for the computation of the roc_auc_score
        targets = [item.item() for tensor in targets for item in tensor]
        pred_probs = [item.item() for tensor in pred_probs for item in tensor]

        # Compute area under the receiver operating characteristic curve
        train_roc_auc = roc_auc_score(targets, pred_probs)

        # Compute the f1_score
        score_f1 = f1_score(targets, list(map(lambda x : round(x), pred_probs)))

        # Compute confusion matrix
        training_norm_cfm = cfm_training / np.sum(cfm_training, axis=1).reshape(-1, 1)

        # Compute loss and accuracy
        loss_training = running_loss / amount_data
        accuracy_training = correct_classified / amount_data

        # Put the computed metrics in a dictionary for futher processing
        training_results = {'loss': loss_training,
                            'accuracy': accuracy_training,
                            'confusion_matrix': cfm_training,
                            'confusion_matrix_norm': training_norm_cfm,
                            'roc_auc': train_roc_auc,
                            'score_f1' : score_f1,
                            'ground_truth': targets,
                            'predicted': pred_probs,
                            'image_names': img_names,
                            'auxiliary_image_names' : auxiliary_image_names}

        # Prints the current epoch and the calculated metrics
        print(f'Epoch {epoch + 1} - Loss {loss_training} - Accuracy {accuracy_training} - AUROC {train_roc_auc}')

        return training_results

    def set_reproudcibility(self, seed: int = 42) -> None:
        """
        Sets the same seed for torch.manual_seed() and np.random.seed(). torch.manual_seed() considers
        also torch.cuda.manual_seed_all(). Further, this method sets torch.use_deterministic_algorithms to True,
        to ensure that CUDNN uses only deterministic algorithms if available. This may influence 
        the training time negatively. It also sets torch.backends.cudnn.benchmark to False
        such that CUDNN will indeed pick deterministic algorithms and the CUBLAS_WORKSPACE_CONFG to 40968:8.
        
        Parameters
        -------------
            seed : int
                Seed for numpy and torch.
        """
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True) 
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # For CUBLAS reproducibility (deterministic algorithms): See also https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        torch.backends.cudnn.benchmark = False # See answer from Thomas V https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/2
        torch.manual_seed(seed)

    def get_compound_transformation(self) -> tuple[transforms.Compose,
                                                   transforms.Compose]:
        """
        Initializes transformations for the training and the validation dataset.
        The following transformations are defined within this method for the
        training dataset:
        1) ToTensor(),
        2) RandomHorizontalFlip(p=0.5),
        3) RandomVerticalFlip(p=0.5)
        4) and RandomApply([RamdomRotation[90,90], p=0.5]).
        The transformations will be executed in the mentioned order.

        And the following transformations for the validation dataset:
        1) ToTensor()

        Returns
        -------------
            (compound_transformation_train, compound_transformation_val : tuple
                The tuple contains two transforms.Compose objects.
        """
        # Transformation for training samples
        compound_transformation_train = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation([90,90])], p=0.5)])

        # Transformation for validation samples
        compound_transformation_val = transforms.Compose(
            [transforms.ToTensor()])

        return compound_transformation_train, compound_transformation_val


if __name__ == '__main__':
    """
    ATTENTION BEFORE RUNNING A NEW (SUB)EXPERIMENT:  
        Increase at least the variable run_experiment_number for each new experiment run.
        Adapt the variable current_experiment, if the run is a complete new experiment and has nothing to do with the former one.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, required=True, default='/Your/Default/Path/Here/',
                        help='Specify the absolute path to the root directory where your dataset is located. Default root directory is "/Your/Default/Path/Here/". You should type the absolute path as a string, e.g. --path_dataset "/Your/Default/Path/Here". In the root directory you should have two subdirectories (train and test) and two csv files (train.csv and test.csv). You can get the dataset from the https://www.isic-archive.com. Be aware that you have to rename the .csv files if you want to use the original dataset from the isic website.')
    parser.add_argument('--path_experiment', type=str, required=True, default='/YourPathHere/Experiments/',
                        help='Specify the absolute path where your root direcory for your experiments is located. Within this directory you can save your experiment results. Default directory is "/YourPathHere/Experiments/" The path should be defined as a string value, e.g. --path_experiments "/YourPathHere/Experimente/". All experiments will be stored within this root directory.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Specify the name of your experiment. This name determines the subdirectory in your experiment directory for a particular experiment. This value has be specified as a string, e.g. --experiment_name "vanilla". In this case the directory would look like /YourPathHere/Experiments/Vanilla/ .')
    parser.add_argument('--experiment_number', type=str, required=True,
                        help='Specify the current experiment number as a combination of the string "number_xxx" and a sequence number xxx (also a string). This means we are starting at number_001 followed by number_002 and so on. This value shall be defined as a string, e.g. --experiment_number "number_001". This will create a subdirectory within your experiment_name directory in which the actual experiment will be saved. For an example your directory may look like /YourPathHere/Experiments/Vanilla/number_001/ .')
    parser.add_argument('--seed', type=int, required=True, default=42,
                        help='Specify a random seed for the numpy and torch library and shall ensure reproducibility. This value shall be defined as an integer and therefore without quotation marks, e.g. --seed 42. The default seed is 42.')
    parser.add_argument('--epochs', type=int, required=True, default=20,
                        help='Specify how many epochs your model shall be trained. This value shall be defined as an integer and therefore without quotation marks, e.g. --epochs 20. The default value for epochs is 20.')
    parser.add_argument('--batch_size', type=int, required=True, default=32,
                        help='Specify the number of batches. This value shall be defined as an integer and therefore without quotation marks, e.g. --batch_size 32. The default value for batch_size is 32. Common sizes are 2^N, whereby N is a natural number.')
    parser.add_argument('--learning_rate', type=float, required=True, default=0.0001,
                        help='Specify the learning rate for your training cycle. This value shall be defined as a float. Common values are in the following intervall [0.0003, 0.001]. The default learning rate value is 0.0001.')
    parser.add_argument('--weight_decay', type=float,
                        help='Specify the weight decay value for the Adam optimizer. The value shall be defined as float value. This mean no quotation marks, e.g. --weight_decay 0.00001. Default value is None.')
    parser.add_argument('--film_layer', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], nargs='*', required=True,
                        help='Specify in which stage of the efficientnet the intermediate feature map shall be modulated. Choose from a pre-defined list, i.e. [1, 2, 3, 4, 5, 6, 7, 8, 9]. You can choose multiple values from that list, e.g. --film_layer 1 9.')
    parser.add_argument('--weighted_random_sampler', action='store_true',
                        help='If this argument will be set, then we are using a weighted random sampler to tackle the issue with the imbalanced dataset.')
    parser.add_argument('--loss_function', choices=['BCEWithLogitLoss'], type=str, required=True, default='BCEWithLogitLoss',
                        help='Pick one the following loss functions. To get information about the loss functions you can visit the website https://pytorch.org/docs/stable/nn.html#loss-functions.')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam',], type=str, required=True,
                        help='Select one of the following optimizer as string value:\n 1) SGD (stochastic gradient descent)/\
                        \n 2) Adam')
    parser.add_argument('--pretrained', action='store_true', help='If this argument will be set, then the efficientnet_b4 (FiLMed) and the efficientnet_b0 (FiLM-Generator) will use pretrained weights. Else we do not use pretrained cnns.')
    parser.add_argument('--dropout_active', action='store_true, help=If this argument will be set, then the efficientnet_b4 (FiLMed) will use dropout within the classification layer.')
    parser.add_argument('--batchnorm_active', action='store_true', help='If this argument will be set, then the efficientnet_b4 (FiLMed) will use batchnormalization layers within the classification layer.')
    args = parser.parse_args()

    path_dataset = os.path.normpath(args.path_dataset)
    path_experiments = os.path.normpath(args.path_experiment)
    experiment_name = os.path.normpath(args.experiment_name)
    experiment_number = os.path.normpath(args.experiment_number)

    if os.path.exists(path_dataset):
        path_experiment = os.path.join(path_experiments, experiment_name, experiment_number)
        if os.path.exists(path_experiment):
            raise Exception(f'The directory for the experiment {path_experiment} already exists. You don\'t want to override existing experiment results! Check your experiment directory name and the sequential number.')
        else:
            hyperparameter = {
                'epochs' : args.epochs,
                'batch_size' : args.batch_size,
                'lr' : args.learning_rate,
                'weight_decay' : args.weight_decay,
                'optimizer' : args.optimizer,
                'loss_function' : args.loss_function,
                'weighted_random_sampler' : args.weighted_random_sampler,
                'pretrained' : args.pretrained,
                'dropout_active' : args.dropout_active,
                'batchnorm_active' : args.batchnorm_active
                }
            os.makedirs(path_experiment)
            os.mkdir(os.path.join(path_experiment, 'Model/'))
            train = train(hyperparameter=hyperparameter, path_experiment=path_experiment, seed=args.seed, path_dataset=path_dataset, args=args)
            train.train()
    else:
        raise Exception(f"The path {path_dataset} does not exist. Make sure that you typed the correct path into the console.")