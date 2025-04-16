import torch
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from io import BytesIO
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler

class utils():
    """
    This class provides several static methods.  For an example:
        * create_submission_file -> creates a file that can be uploaded on
          ISIC / Kaggle to receive the AUROC value.
        * to_csv -> Stores training / Validation results.
        * graph_to_tensorboard -> Visualizes the model architecture.
    """

    @staticmethod
    def create_submission_file(submission_file_path: str, rows: list) -> None:
        """
        Creates a submission file for the results from the test dataset.
        This submission file contains the attributes image_name and target.
        Whereby, the target attribute is the model prediction (ranges from 0 to 1).
        
        Parameters
        -------------
            submission_file_path : str
                Specifies the path where the .csv file shall be saved. Tip: save it in the experiment directory.
            rows : list
                Contains image_name and target pairs as a list e.g., [[img_1, 0.43245], [img_2, 0.84563], ...].
                whereby, the target refers to the likelihood that the image is malignant. 
        """
        # Insert a header (features: image_name and target)
        rows.insert(0, ['image_name', 'target'])
        # Create a submission file with .csv as file extension
        with open(os.path.join(submission_file_path, os.path.normpath('/submission.csv')), 'w', newline='') as submission_file:
            writer = csv.writer(submission_file, delimiter=',')
            writer.writerows(rows)

    @staticmethod
    def get_initial_state_metric() -> dict:
        """
        Initializes a dictionary which contains all tracked metrics which serves as a representation
        for training epoch 0. This dictionary will be called only once per experiment.
        
        Returns
        -------------
            metricparams : dict
                A dictionary which consists of all metrics (initialized with .0).
        """
        metricparams = {
            'accuracy/train': 0.0,
            'loss/train': 0.0,
            'roc_auc/train': 0.0,
            'f1_score/train': 0.0,
            'accuracy/val': 0.0,
            'loss/val': 0.0,
            'roc_auc/val': 0.0,
            'f1_score/val': 0.0,
            'true_positive/train': 0.0,
            'false_positive/train': 0.0,
            'true_negative/train': 0.0,
            'false_negative/train': 0.0,
            'true_positive/val': 0.0,
            'false_positive/val': 0.0,
            'true_negative/val': 0.0,
            'false_negative/val': 0.0,
            'tpr/train': 0.0,
            'fpr/train': 0.0,
            'fnr/train': 0.0,
            'tnr/train': 0.0,
            'tpr/val': 0.0,
            'fpr/val': 0.0,
            'fnr/val': 0.0,
            'tnr/val': 0.0
        }

        return metricparams

    @staticmethod
    def to_csv(results: dict, path_experiment: str, filename: str, epoch: int) -> None:
        """
        Saves the training / validation results in a csv file. Thereby, we will keep track of all metric results
        for each epoch.

        Parameters
        -------------
            results : dict
                The following attributes are stored in results:
                the loss value, accuracy, confusion matrix, normalized confusion matrix, receiver operating
                characteristic curve, f1 score, ground truth values, predicted values, image names which shall be classified
                and auxiliary image names. 
            path_experiment : str
                The absolute path to the experiment directory.
            filename : str
                The name of the .csv file where all the features from the results dict will be stored as well as the current epoch number.
            epoch : int
                The current epoch for which we will save the metric results in to .csv.
        """
        image_names = results['image_names']
        auxiliary_image_names = results['auxiliary_image_names']
        image_names_list = [os.path.splitext(os.path.basename(image_name))[0]
                            for mini_batch in image_names for image_name in mini_batch]
        auxiliary_image_names_list = [os.path.splitext(os.path.basename(auxiliary_image_name))[0]
                                     for mini_batch in auxiliary_image_names for auxiliary_image_name in mini_batch]

        # Wrap everything into a dictionary
        data = {
            'image_names': image_names_list,
            'auxiliary_image_names': auxiliary_image_names_list,
            'prob_predicted': results['predicted'],
            'predicted': list(map(lambda x: round(x), results['predicted'])),
            'ground_truth': results['ground_truth'],
            'loss': [results['loss']]*len(results['predicted']),
            'accuracy': [results['accuracy']]*len(results['predicted']),
            'f1_score': [results['score_f1']]*len(results['predicted']),
            'roc_auc': [results['roc_auc']]*len(results['predicted']),
            'epoch': [epoch+1]*len(results['predicted'])
        }

        # If the .csv file already exists at the declared path then save data into the existing .csv file
        # else create a .csv file.
        df = pd.DataFrame.from_dict(data)
        if os.path.exists(os.path.join(path_experiment, filename)):
            df.to_csv(os.path.join(path_experiment, filename),
                      mode='a', index=False, header=False)
        else:
            df.to_csv(os.path.join(path_experiment, filename), index=False)

    @staticmethod
    def graph_to_tensorboard(net: torch.nn.Module, writer: SummaryWriter, img: torch.tensor,
                             auxiliary_img: torch.tensor, epoch: int, iterator: int) -> None:
        """
        Adds a model visualization to the tensorboard for epoch 0.

        Parameters
        -------------
            net : torch.nn.Module
                The model which shall be visualized.
            writer : SummaryWriter)
                Writer serves as an instance for communication between the tensorboard.
            img : torch.tensor
                These are the images that serve as input for the model to classify.
            auxiliary_img : torch.tensor
                These are the images that serve as conditional information.
            epoch : int
                The current training epoch.
            iterator : int
                The current batch in that epoch.
        Source: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
        """
        if epoch == 0 and iterator == 0:
            writer.add_graph(net, input_to_model=(img, auxiliary_img))
            writer.flush()

    @staticmethod
    def hparams_to_tensorboard(writer: SummaryWriter, epoch: int, hparams: dict,
                               metricparams: dict) -> None:
        """
        Save the hyperparameters and calculated metrics for a specific epoch (> 0) to tensorboard.
        
        Parameters
        -------------
            writer : SummaryWriter
                Writer serves as an instance for communication between the tensorboard.
            epoch : int
                The current training epoch.
            hparams : dict
                All hyperparameters which shall be tracked are listed within this dictionary.
            metrickparams : dict
                All metrics are listed within this dictionary which shall be tracked.
        """
        hparams['epoch'] = epoch+1
        writer.add_hparams(hparams, metricparams)
        writer.flush()

    @staticmethod
    def to_tensorboard(net: torch.nn.Module, writer: SummaryWriter, training_results: dict,
                       validation_results: dict, hparams: dict, epoch: int) -> None:
        """
        Writes the following metrics to TensorBoard:
            Confusion Matrix incl. normalized version
            Accuracy
            Loss
            AUROC
            PR-Curve
            TP/FN/FP/TN
            TPR/FNR/FPR/TNR
            Parameter histogram of the model
        
        Parameters
        -------------
            net : torch.nn.Module
                The convolutional neural network.
            writer : SummaryWriter
                Writer serves as an instance for communication between the TensorBoard.
            training_results : dict
                The training dictionary which holds metric values such as accuracy, loss and AUROC.
            validation_results : dict
                The validation dictionary which holds metric values such as accuracy, loss and AUROC.
            hparams : dict:
                All hyperparameters which shall be tracked are listed within this dictionary.
            epoch : int
                Number of epochs for a particular training cycle.
        """
        # Confusion matrix for the current training cycle
        training_cfm = training_results['confusion_matrix']
        training_tp, training_fn, training_fp, training_tn = training_cfm.ravel()

        # Calculate tpr, fnr, fpr, tnr for the current training cycle
        training_tpr = (training_tp/(training_tp+training_fn))
        training_fpr = (training_fp/(training_fp+training_tn))
        training_fnr = (training_fn/(training_fn+training_tp))
        training_tnr = (training_tn/(training_tn+training_fp))

        # Confusion matrix validation cycle
        validation_cfm = validation_results['confusion_matrix']
        validation_tp, validation_fn, validation_fp, validation_tn = validation_cfm.ravel()

        # Calculate tpr, fnr, fpr, tnr for the validation cycle
        validation_tpr = (validation_tp/(validation_tp+validation_fn))
        validation_fpr = (validation_fp/(validation_fp+validation_tn))
        validation_fnr = (validation_fn/(validation_fn+validation_tp))
        validation_tnr = (validation_tn/(validation_tn+validation_fp))

        # Putting all together in the TensorBoard as scalars
        # Training:
        writer.add_scalar('accuracy/train', training_results['accuracy'], epoch+1)
        writer.add_scalars('accuracy/overall', {'train': training_results['accuracy'], 'val': validation_results['accuracy']}, epoch+1)
        writer.add_scalar('loss/train', training_results['loss'], epoch+1)
        writer.add_scalars('loss/overall', {'train': training_results['loss'], 'val': validation_results['loss']}, epoch+1)
        writer.add_scalar('auroc/train', training_results['roc_auc'], epoch+1)
        writer.add_scalars('auroc/overall', {'train': training_results['roc_auc'], 'val': validation_results['roc_auc']}, epoch+1)
        writer.add_scalar('f1_score/train', training_results['score_f1'], epoch+1)
        writer.add_scalars('f1_score/overall', {'train': training_results['score_f1'], 'val': validation_results['score_f1']}, epoch+1)
        writer.add_scalar('true_positive/train', training_tp, epoch+1)
        writer.add_scalar('false_positive/train', training_fp, epoch+1)
        writer.add_scalar('false_negative/train', training_fn, epoch+1)
        writer.add_scalar('tpr/train', training_tpr, epoch+1)
        writer.add_scalar('fpr/train', training_fpr, epoch+1)
        writer.add_scalar('tnr/train', training_tnr, epoch+1)
        writer.add_scalar('fnr/train', training_fnr, epoch+1)
        
        # Validation:
        writer.add_scalar('true_negative/train', training_tn, epoch+1)
        writer.add_scalar('accuracy/val', validation_results['accuracy'], epoch+1)
        writer.add_scalar('loss/val', validation_results['loss'], epoch+1)
        writer.add_scalar('auroc/val', validation_results['roc_auc'], epoch+1)
        writer.add_scalar('f1_score/val', validation_results['score_f1'], epoch+1)
        writer.add_scalar('true_positive/val', validation_tp, epoch+1)
        writer.add_scalar('false_positive/val', validation_fp, epoch+1)
        writer.add_scalar('true_negative/val', validation_tn, epoch+1)
        writer.add_scalar('false_negative/val', validation_fn, epoch+1)
        writer.add_scalar('tpr/val', validation_tpr, epoch+1)
        writer.add_scalar('fpr/val', validation_fpr, epoch+1)
        writer.add_scalar('tnr/val', validation_tnr, epoch+1)
        writer.add_scalar('fnr/val', validation_fnr, epoch+1)
        
        # Save to TensorBoard
        writer.flush()

        # Add precision recall curve
        writer.add_pr_curve('precision_recall/train', np.array(
            training_results['ground_truth']), np.array(training_results['predicted']), epoch+1)
        writer.add_pr_curve('precision_recall/val', np.array(
            validation_results['ground_truth']), np.array(validation_results['predicted']), epoch+1)
        
        # Save to TensorBoard
        writer.flush()

        # Add confusion matrix image to TensorBoard
        cfm_train = utils.get_heatmap_from_cfm(
            training_results['confusion_matrix'])
        writer.add_image('cfm/train', cfm_train, epoch+1)
        cfm_train_norm = utils.get_heatmap_from_cfm(
            training_results['confusion_matrix_norm'])
        writer.add_image('cfm_norm/train', cfm_train_norm, epoch+1)
        cfm_val = utils.get_heatmap_from_cfm(
            validation_results['confusion_matrix'])
        writer.add_image('cfm/val', cfm_val, epoch+1)
        cfm_val_norm = utils.get_heatmap_from_cfm(
            validation_results['confusion_matrix_norm'])
        writer.add_image('cfm_norm/val', cfm_val_norm, epoch+1)
        
        # Save to TensorBoard
        writer.flush()

        # Add roc curve to TensorBoard
        fig = utils.get_roc_graph(
            training_results['ground_truth'], training_results['predicted'], f'Training - Area under the curve in epoch {epoch+1}')
        writer.add_image('roc/train', fig, epoch+1)
        fig = utils.get_roc_graph(
            validation_results['ground_truth'], validation_results['predicted'], f'Validation - Area under the curve in epoch {epoch+1}')
        writer.add_image('roc/val', fig, epoch+1)
        
        # Save to TensorBoard
        writer.flush()

        # Add histogram for model weights and biases and save them to TensorBoard
        for name, param in net.named_parameters():
            writer.add_histogram(name, param, epoch+1)
            writer.flush()

        # Add current hyperparameters and computed metric values to TensorBoard
        utils.hparams_to_tensorboard(writer,
                                     epoch,
                                     hparams,
                                     metricparams={
                                         'accuracy/train': training_results['accuracy'],
                                         'loss/train': training_results['loss'],
                                         'roc_auc/train': training_results['roc_auc'],
                                         'f1_score/train': training_results['score_f1'],
                                         'accuracy/val': validation_results['accuracy'],
                                         'loss/val': validation_results['loss'],
                                         'roc_auc/val': validation_results['roc_auc'],
                                         'f1_score/val': validation_results['score_f1'],
                                         'true_positive/train': training_tp,
                                         'false_positive/train': training_fp,
                                         'true_negative/train': training_tn,
                                         'false_negative/train': training_fn,
                                         'true_positive/val': validation_tp,
                                         'false_positive/val': validation_fp,
                                         'true_negative/val': validation_tn,
                                         'false_negative/val': validation_fn,
                                         'tpr/train': training_tpr,
                                         'fpr/train': training_fpr,
                                         'fnr/train': training_fnr,
                                         'tnr/train': training_tnr,
                                         'tpr/val': validation_tpr,
                                         'fpr/val': validation_fpr,
                                         'fnr/val': validation_fnr,
                                         'tnr/val': validation_tnr})
        
        # Save to TensorBoard
        writer.flush()

    @staticmethod
    def get_heatmap_from_cfm(cfm) -> torch.tensor:
        """
        Creates a heatmap for the confusion matrix.

        Parameters
        -------------
            cfm : numpy.array
                The confusion matrix represented as a numpy array.
        
        Returns
        -------------
            heatmap : torch.tennsor
                The heatmap.
        """
        sns_heatmap = sns.heatmap(cfm, annot=True, fmt='.2f',
                                  xticklabels=['Predicted malignant', 'Predicted benigne'],
                                  yticklabels=['Actual malignant', 'Actual benigne']).xaxis.tick_top()
        
        # Saves the heatmap to a buffer and converts it to a PyTorch tensor
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        buffer.seek(0)
        heatmap_image = Image.open(buffer)
        transform = transforms.ToTensor()
        heatmap = transform(heatmap_image)
        plt.close()
        
        return heatmap

    @staticmethod
    def get_roc_graph(ground_truth : float, predicted_scores : float, estimator_name : str) -> torch.tensor:
        """
        Creates a receiver operating characteristic (roc) plot.

        Parameters
        -------------
            ground_truth : float
                The true target labels.
            predicted_scores : float
                The predicted scores.
            estimator_name : str
                Specifies whether the mode is training or validation, and includes
                the current epoch.

        Returns
        -------------
            roc_auc_image : torch.tensor
                roc graph.
        """
        fpr, tpr, _ = roc_curve(ground_truth, predicted_scores)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                                  roc_auc=roc_auc, estimator_name=estimator_name)
        display.plot()

        # Saves the roc_auc graph to a buffer and converts it to a PyTorch tensor
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        buffer.seek(0)
        roc_auch_image = Image.open(buffer)
        transform = transforms.ToTensor()
        roc_auch_image = transform(roc_auch_image)
        plt.close()

        return roc_auch_image

    @staticmethod
    def save_model(path_experiment: str, filename: str, net: torch.nn.Module, optimizer: torch.optim,
                   training_results: dict, validation_results, epoch: int) -> None:
        '''
        Stores the trained convolutional neural network model along with its 
        computed metrics, optimizer state, and epoch.

        Parameters
        -------------
            path_experiment :str
                The absolut path to the current experiment. Determines where
                the model will be stored.
            filename : str
                The filename for the model.
            net : nn.Module
                The convolutional neural network (trained model).
            optimizer : torch.otim
                the optimizer function used for the training cycle.
            training_results : dict
                The dictionary with the computed metrics for the current epoch.
            validation_results : dict
                The dictionary with the computed metrics for the current epoch.
            epoch : int
                The current epoch.
        '''
        path = os.path.join(path_experiment, filename)
        torch.save({'loss_training': training_results['loss'],
                    'accuracy_training': training_results['accuracy'],
                    'roc_auc_training': training_results['roc_auc'],
                    'f1_score_training': training_results['score_f1'],
                    'loss_validation': validation_results['loss'],
                    'accuracy_validation': validation_results['accuracy'],
                    'roc_auc_validation': validation_results['roc_auc'],
                    'f1_score_validation': validation_results['score_f1'],
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch+1},
                   path)

    @staticmethod
    def get_sampler(targets: list, benign: int, malignant: int) -> torch.utils.data.WeightedRandomSampler:
        """
        Generates a PyTorch WeightedRandomSampler to handle class imbalance in training data.
        This method calculates the weights for each class label in the `targets` list based on 
        the number of benign and malignant samples.
        
        Parameters
        -------------
            targets : list
                All class labels within the train subset with kept order.
            benign : int
                The number of class labels which are considered as benign within
                the train subset.
            malignant : int
                The number of class labels which are considered as malignant within
                the train subset.
        
        Returns
        -------------
            sampler : torch.utils.data.WeightedRandomSampler 
                A sampler object created with the PyTorch WeightedRandomSampler class.
                TODO: I am curious whether Oversampling is more decent compared to weighted random sampling.
        """
        if (benign and malignant) != 0:
            class_weights = [1 / benign, 1 / malignant]
        else:  # TODO: There have to be a better way to handle empty classes
            benign += 0.00001
            malignant += 0.00001
            class_weights = [1 / benign, 1 / malignant]
        weights = [0] * len(targets)
        for i in range(len(targets)):
            class_weight = class_weights[targets[i]]
            weights[i] = class_weight
        generator_random_split = torch.Generator()
        generator_random_split.set_state(torch.get_rng_state())
        sampler = WeightedRandomSampler(weights, num_samples=len(
            weights), replacement=True, generator=generator_random_split)
        
        return sampler