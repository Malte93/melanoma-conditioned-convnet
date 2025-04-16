import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

def val_step(net: torch.nn.Module, valloader: DataLoader, epoch: int,
             device: torch.device, criterion: torch.nn) -> tuple[roc_auc_score, dict]:
    """
    Evaluates the model for the current epoch on the validation dataset.

    Parameters
    -------------
        net : torch.nn.Module
            The model.
        valloader : DataLoader
            The validation dataset which will be used for the evaluation.
        epoch : int
            The current epoch.
        device : torch.device
            Either cpu or gpu. Is used for transferring the data to an accelerator if available.
        criterion : torch.nn
            The loss function for the current experiment.

    Returns
    -------------
        roc_auc_score : sklearn.metrics.roc_auc_score
            The area under receiver operating characteristic curve score.
        validation_results : dict
            loss, accuracy, confusion matrix, normalized confusion matrix,
            AUROC, f1 score, targets, predicted values, image names, auxiliary
            image names
    """
    # Set model in to validation mode
    net.eval()

    # Initialize variables for tracking stats
    running_loss = 0
    # track number of data since the DataLoader does not use drop_last?
    number_data = 0
    # Track targets, pred_probs, img_names and auxiliar_image_names for a .csv file
    correct_classified = 0
    targets = []
    pred_probs = []
    img_names = []
    auxiliary_image_names = []
    # Empty confusion matrix
    cfm_validation = np.zeros(shape=(2, 2))

    # Evaluate model
    for i, data in enumerate(valloader, 0):
        net.eval()
        with torch.no_grad():
            # Unwrap data and assign them to the device
            img, labels, auxiliary_img, image_name, auxiliary_image_name = data
            img, labels, auxiliary_img = img.to(device), labels.to(device), auxiliary_img.to(device)
            labels = torch.unsqueeze(labels, 1).to(torch.float32)

            # Compute the logit and the loss
            outputs = net(img, auxiliary_img)
            loss = criterion(outputs, labels)

            # Compute the model prediction with logit (model output) - threshold .5
            probs = torch.sigmoid(outputs)
            prediction = torch.round(probs)

            # Track ground truth labels, predictions for roc_auc and image names
            targets.append(labels) # .extend would be more appropriate, then I don't have to unravel it later on
            pred_probs.append(probs)
            img_names.append(image_name)
            auxiliary_image_names.append(auxiliary_image_name)

            # Compute statistical metrics
            number_data += labels.size()[0]
            running_loss += loss.item() * img.size(0)
            correct_classified += torch.sum(labels.cpu(), prediction.cpu()).item()
            cfm_validation += confusion_matrix([k.item() for k in labels], [
                                               k.item() for k in prediction], labels=[1, 0])

    # Flatten list of targets and predictions tensors in to a single list for roc_auc_score
    targets = [item.item() for tensor in targets for item in tensor]
    pred_probs = [item.item() for tensor in pred_probs for item in tensor]

    # Compute area under the receiver operating characteristic curve
    val_roc_auc = roc_auc_score(targets, pred_probs)

    # Compute f1 score
    score_f1 = f1_score(targets, list(map(lambda x: round(x), pred_probs)))

    # Compute confusion matrix
    validation_norm_cfm = cfm_validation / \
        np.sum(cfm_validation, axis=1).reshape(-1, 1)

    # Compute loss and accuracy
    loss_validation = running_loss / number_data
    accuray_validation = correct_classified / number_data

    # Put the computed metrics in a dictionary for futher processing
    validation_results = {'loss': loss_validation,
                          'accuracy': accuray_validation,
                          'confusion_matrix': cfm_validation,
                          'confusion_matrix_norm': validation_norm_cfm,
                          'roc_auc': val_roc_auc,
                          'score_f1': score_f1,
                          'ground_truth': targets,
                          'predicted': pred_probs,
                          'image_names': img_names,
                          'auxiliary_image_names': auxiliary_image_names}

    # Prints the current epoch and metrics for that epoch
    print(f'Epoch {epoch + 1} - Loss {loss_validation} - Accuracy {accuray_validation} - AUROC {val_roc_auc}')

    return val_roc_auc, validation_results