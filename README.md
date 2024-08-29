
# DGMMC : Deep Gaussian Mixture Model Classifier

This GitHub repository houses the implementation of a Deep Gaussian Mixture Model classifier (DGMMC) for image classification, with an emphasis on capturing complex data distributions. The model is implemented using a deep neural network architecture (using Pytorch impementation) to efficiently learn intricate patterns within data. It is mainly test on features provded by pretrained models such as CLIP or ImageBind in order to use the information already captured without retraining the whole model. The classifier proposed by Hayashi et al. (https://github.com/HideakiHayashi/SDGM_ICLR2021) is also considered in our experiments.

Please cite our paper related to this work:
```
@techreport{
}
```

## Key features

- Deep Gaussian Mixture Model Implementation: The repository includes well-documented code for the Deep Gaussian Mixture Model, leveraging a neural network to model the latent structure of the data through a mixture of Gaussian distributions.

- Image classification using pretrained models as feature extractor : Experiment using CLIP and ImageBind to extract features from the several datasets (MNIST, CIFAR10, CIFAR100, ESC-50 and ImageNet) and directly use classifer on top of it for image classification.

- Feature dimension reduction : We propose to reduce the feature dimension using a learnt linear layer and a PCA decomposition.

## Usage:

1. Clone the repository to your local machine.
1. Explore the ```src``` directory to find the DGMMC implementation
1. Run the experiments on the MNIST dataset using the  Jupyter Notebook ```Example.ipynb``` that is available to illustrate the creation of the dataset, the training of our classifier and the visualization of  the trained parameters of the network.


## Experiment on different dataset

### Datasets

The datasets considered are images (MNIST, CIFAR10, CIFAR100 and ImageNet) and audio sound (ESC-50). In order to use thos datasets in our experiments, there is a need to extract the features from the images using either CLIP or ImageBind. The codes that was used to do it is available in the folder "Code_to_extract_features_and_prepare_datasets" but it is required to download the source files from CLIP (https://github.com/openai/CLIP) and ImageBind (https://github.com/facebookresearch/ImageBind) to use it.

To use the available code, ensure that the feature are stored according to template defined in the extracting code.

For example, to run the tutorial, the MNIST features provided by ImageBind should be stored this way : 
 
```
Project
│   README.md
│   code_to_execute_MNIST.py  
│
└───Features
     |
     |
     └───MNIST
           |
           |
           └───ImageBind
                    │   
                    │   
                    │
                    └───Train
                    |     │   file1.pt
                    |     │   file2.pt
                    |     │   ...
                    ───Test
                          │   file1.pt
                          │   file2.pt
                          │   ...
  
```


## Other code
A lot of files are avialble to run the experiments on the several dataset. In thos file, you just need to replace the PATH to the folder containing the feature extracted by CLIP and/or ImageBind and also specify where the dataset is stored on the disk. Custom datasets have been implemented and are available in the "src" folder to load from the disk the feature on the go.

The code to gather the results and plot some of them is also available.

## Requirements
This code wa developped in python 3.10. The specific version of the main packages used in this repository are detailled in the ```requirements.txt``` file.

## Contributions and Issues:
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License:
This code is available under the [MIT](https://choosealicense.com/licenses/mit/) License.
