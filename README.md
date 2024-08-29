
# DGMMC : Deep Gaussian Mixture Model Classifier

This GitHub repository houses the implementation of a Deep Gaussian Mixture Model classifier (DGMMC) for image classification, with an emphasis on capturing complex data distributions. The model is implemented using a deep neural network architecture (using Pytorch impementation) to efficiently learn intricate patterns within data.

## Key features

- Deep Gaussian Mixture Model Implementation: The repository includes well-documented code for the Deep Gaussian Mixture Model, leveraging a neural network to model the latent structure of the data through a mixture of Gaussian distributions.

- Preliminary Experiments on Synthetic Dataset: The code comes with an example set of preliminary experiments conducted on a synthetic dataset. This allows users to understand the model's behavior in controlled settings and assess its performance in capturing underlying data structures.

- Results and Visualizations: Detailed results from the experiments, including performance metrics, visualizations of the learned distributions, and comparisons with ground truth, are provided. This helps users evaluate the efficacy of the DGMM on synthetic data.

## Usage:

1. Clone the repository to your local machine.
1. Explore the src directory to find the DGMMC implementation
1. Run the experiments on the synthetic dataset using the provided Python scripts.

## Experiment on a Synthetic Dataset
This section will described the synthetic dataset we use in our experiments and also provide elements for the comprehension of the results that are stored as .csv file in this repository.

### Dataset
The synthetic dataset consist of a set of points in 2D where each point is associated to two features which are its spatial position in the 2d space. Those points are created as samples randomly provided by spherical Multivariate Gaussian Distribution with known parameters (means, covariance matrix, ...) and this for 6 different configuration. The points created for this preliminary experiments are stored as .csv file in the experiments folder and the script create_data.py was used to create all the points.

### Results
In the experiments folder, you can find the complete ```GMM_spherical_complete_results.csv``` file that contains our results for the different configuration in terms of accuracy for image classifiation, but also the Mean square Error (MSE) between our model and the parameters of the Gaussian that ware used to provide the points intialy. For each results, we trained 6 neural network and provided the average results over those 6 runs as long as the standard deviation and the standard error.

### Tutorial
A Jupyter Notebook is avaiable to illustrate the creation of the dataset, the training of our classifier and the visualization of the the trained parameters of the network.

## Requirements
This code wa developped in python 3.10. The specific version of the main packages used in this repository are detailled in the ```requirements.txt``` file.

## Contributions and Issues:
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License:
This code is available under the [MIT](https://choosealicense.com/licenses/mit/) License.
