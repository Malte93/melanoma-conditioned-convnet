# Classification of malignant melanoma with a conditioned ConvNet

This repository provides a brief overview to my bachelor's thesis and the corresponding source code.

## Objective
The overall goal was to implement a conditioned convolutional neural network that uses lesion images from a patient to classify whether a given lesion is a malignant melanoma or a benign lesion, and to assess whether the conditioned convolutional neural network achieves a higher evaluation score than a non-conditioned convolutional neural network. 

## Dataset
The [dataset](https://challenge2020.isic-archive.com/) [1] used for this thesis was provided by the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/) and the [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/). Originally compiled for the [SIIM-ISIC Melanoma Classification Challenge 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview), the dataset comprises $$33,126$$ training images of skin lesions from $$2,056$$ individual patients. Of these, $$98$$\% are considered benign and $$2$$\% are considered malignant. Each patient has at least two different lesion images across the training and testing datasets. Further, each patient has a unique identifier and each lesion image has a unique identifier as well. Thus, it is possible to link each patient to all of their lesion images, which was crucial for the thesis.

## What are conditioned ConvNet's?
Instead of using just a single image (or any other appropriate format for a ConvNet, such as an audio file) for classification or regression, the artificial neural network will incorporate additional information related to the image. This additional information is often called conditioning information or contextual information and is used as an auxiliary input to the artificial neural network. Conditioning information may be metadata, images, or any other format that seems reasonable for the task.

Using this conditioning information, the input to the hidden layers will be transformed in a specific way, attempting to learn useful patterns from it. Not only the input to one hidden layer may be transformed. The choice of which hidden layer’s input, and how many of them, should be transformed (i.e., conditioned) by the conditioning information is determined by a hyperparameter.

One method to transform the non-conditioning input is to simply concatenate the conditioning information with the input to the hidden layer. For my Bachelor's thesis, however, I chose to apply a Feature-Wise Linear Modulation (FiLM) approach, which takes as input the lesion image to be classified, and a second lesion image from the same patient, that serves as the conditioning information.

## Feature-wise Linear Modulation (FiLM)
FiLM applies feature-wise affine transformations to the feature maps of a convolutional neural network, thereby influencing the output. The feature wise-affine transformation ensures that, for example, intermediate features are either scaled up or down, allowing certain features to be emphasized or suppressed. [2]

A convolutional neural network, or more generally, an artificial neural network that incorporates FiLM, consists of the following components:
* FiLM-Generator is an artificial neural network (in the case of my thesis, a convolutional neural network) that takes conditioning information as input and outputs two learned parameters, $$\gamma_{i,c}$$ and $$\beta_{i,c}$$. These parameters modulate the non-conditioning information. $$\gamma_{i,c}$$ is a scaling parameter, whereas $$\beta_{i,c}$$ is a shifting parameter. Here, $$i$$ indicates the sample for which the parameter was calculated, and $$c$$ refers to the feature map (i.e., channel) for which the parameter was calculated. Thus, the FiLM-Generator can be seen as a function $$f$$ that, given an image $$l_{i}$$ (in this thesis), computes $$\gamma_{i,c}$$ and $$\beta_{i,c}$$. $$f$$ can then be written as $$f(l_{i})=(\gamma_{i,c}, \beta_{i,c}) \in \mathbb{R}^{2}$$. [2]
* The Feature-wise Linearly Modulated Network (FiLM-ed network) in this thesis is, like the FiLM-Generator, also a convolutional neural network that takes the non-conditioning information (a lesion image, denoted as $$x_i$$) as input. At certain hidden layers, called FiLM layers, it incorporates the parameters $$(\gamma, \beta)$$ computed by the FiLM-Generator [2]. Within the FiLM layers, the intermediate features of $$x_{i}$$, denoted as $$F_{i}$$, are modulated by these parameters as follows [2]:

$$FiLM(F_{i,c}|\gamma_{i,c},\beta_{i,c}) = \gamma_{i,c} \cdot F_{i,c} + \beta_{i,c}.$$

Although the FiLM-Generator and the Feature-wise Linearly Modulated Network are individual artificial neural networks, the entire network was trained end-to-end. For simplicity, let's refer to the network that incorporates both the FiLM-Generator and the FiLM-ed network as the FiLM network. The following diagram depicts the architecture:

<figure>
  <p align=center>
    <img src="https://github.com/user-attachments/assets/4593f4b9-ed2c-4f67-a16b-c637f171e593" alt="FiLM architecture" width="750">
    <figcaption>
      <p align=center>
        Figure 1) The FiLM network architecture. On the left side the FiLM-Generator with the conditioned information $$(l_{i})$$ as input to predict $$(\gamma_{i}, \beta_{i})$$. On the right side the FiLM-ed network which gets as an input the non-conditioned information $$(x_{i})$$ to make a prediction. The FiLM-ed network incorporates the predicted parameters $$(\gamma_{i}, \beta_{i})$$ within its FiLM layer to condition the intermediate features $$F_{i}$$ from $$x_{i}$$. Original image source [3].
      </p>
    </figcaption>
  </p>
</figure>

## Backbone
EfficientNet-b0 and EfficientNet-b4 were used for the FiLM-Generator and the FiLM-ed network, respectively. The performance of the FiLM network was compared to that of a standard EfficientNet-b4 network without any FiLM layers. For more information about EfficientNet, refer to the [paper](https://arxiv.org/abs/1905.11946) by M. Tan and Q. V. Le.

## Evaluation
The model's performance was measured using the Area Under the Receiver Operating Characteristic Curve (AUROC). For more information about AUROC, visit the corresponding [google developer article](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).

## Results
Multiple experiments were conducted, with the best result achieved using a pretrained FiLM network. The FiLM network reached an AUROC score of $$0.8850$$, while the non-conditioned network (a standard EfficientNet-b4) achieved an AUROC of $$0.8510$$. This corresponds to an improvement of $$|0.8850 - 0.8510| = 0.034$$ in AUROC.

_If you have any questions regarding my bachelor thesis feel free to write me a PM._

## References
[1] [Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos, E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D., Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K., Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos, A., Tschandl, P., Weber, J. & Soyer, P. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. Sci Data 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z](https://challenge2020.isic-archive.com/)

[2] [Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018, April). Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1)](https://arxiv.org/abs/1709.07871)

[3] [Dumoulin, V., Perez, E., Schucher, N., Strub, F., Vries, H. D., Courville, A., & Bengio, Y. (2018). Feature-wise transformations. Distill, 3(7), e11.](https://distill.pub/2018/feature-wise-transformations/)
