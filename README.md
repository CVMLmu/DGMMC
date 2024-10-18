
# DGMMC : Deep Gaussian Mixture Model Classifier

This GitHub repository houses the implementation of a **Deep Gaussian Mixture Model classifier (DGMMC)** for image classification, with an emphasis on capturing complex data distributions. The model is implemented using a deep neural network architecture using Pytorch implementation. It has been mainly tested on features provided by pretrained models such as CLIP or ImageBind in order to use the information already captured without retraining the whole model. 
The <a href="https://github.com/HideakiHayashi/SDGM_ICLR2021" target="_blank">SDGM classifier</a> proposed by Hayashi et al.  is also considered in our experiments (see <a href="https://arxiv.org/abs/2410.13421" target="_blank">paper</a>).

**Please cite our paper related to this work:**
```
@techreport{DGMMC2024,
title={Performance of Gaussian Mixture Model Classifiers on Embedded Feature Spaces},
authors={Jeremy Chopin and Rozenn Dahyot},
year={2024},
eprint={2410.13421},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2410.13421}, 
month={October},
doi={10.48550/arXiv.2410.13421},
}
```

## Usage:

1. Clone the repository to your local machine.
1. Explore the ```src``` directory to find the DGMMC implementation
1. A  Jupyter Notebook <a href="./Example.ipynb">Example.ipynb</a> on the CIFAR100 dataset  is provided to illustrate the creation of the dataset, the training of our classifier and the visualization of  the trained parameters of the network. Note that before running  code provided, features (CLIP or ImageBind) need to be computed and stored as explained below and some hardcoded PATH (i.e. look for */home/* in ```*.py``` files) needs to be updated. 

## Experiment on other datasets

The datasets tested were of images  (MNIST, CIFAR10, CIFAR100 and ImageNet) and audio sounds (ESC-50). In order to use those datasets, features need to be computed using either CLIP or ImageBind using the code available in the folder <a href="./Code_to_extract_features_and_prepare_datasets">Code_to_extract_features_and_prepare_datasets</a>. Note, source files from <a href="https://github.com/openai/CLIP">CLIP</a>  and 
<a href="https://github.com/facebookresearch/ImageBind">ImageBind</a>  need to be dowdloaded.
To use our  code, ensure that the feature are stored according to template defined in the extracting code.

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


Additional files are also provided to run the experiments on the other datasets. 
In those files, you just need to replace the PATH to the folder containing the feature extracted by CLIP and/or ImageBind and also specify where the dataset is stored on the disk. Custom datasets have been implemented and are available in the "src" folder to load from the disk the feature on the go.
Additional  codes to collect results and to provide plots are also available in this repo.

## Requirements

This code was developped in python 3.10. 


## License:
This code is available under the [MIT](https://choosealicense.com/licenses/mit/) License.
