import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def DGMMC(features_dim, nb_classes, nb_components, is_diagonal=False, is_prior_uniform = False, is_weights_uniform = False, init_means = None, init_bandwidths = None):

    if is_diagonal:
        pass
    else:
        return DGMMC_spherical(features_dim, nb_classes, nb_components, is_prior_uniform, is_weights_uniform, init_means, init_bandwidths)

class DGMMC_spherical(nn.Module):
    def __init__(self, features_dim, nb_classes, nb_components, is_prior_uniform = False, is_weights_uniform = False, init_means = None, init_bandwidths = None, **kwargs):
        super(DGMMC_spherical, self).__init__(**kwargs)

        assert features_dim > 0
        assert nb_classes > 0
        assert nb_components > 0

        # Hyperparameters
        self.features_dim = features_dim
        self.nb_classes = nb_classes
        self.nb_components = nb_components

        # Architecture parameters
        self.is_prior_uniform = is_prior_uniform
        self.is_weights_uniform = is_weights_uniform

        self.initialize_parameters(init_means, init_bandwidths)


    def initialize_parameters(self, init_means, init_bandwidths):

        # means
        if init_means is None:
            self.means = nn.Parameter(torch.randn(self.nb_classes * self.nb_components, self.features_dim), requires_grad=True)
        else:
            self.means = nn.Parameter(init_means, requires_grad=True)

        # bandwidths
        if init_bandwidths is None:
            self.bandwidths = nn.Parameter(0.15*torch.ones(self.nb_classes * self.nb_components), requires_grad=True)
        else:
            self.bandwidths = nn.Parameter(init_bandwidths, requires_grad=True)

        sigmas_eyes = torch.eye(self.features_dim).unsqueeze(0)
        self.sigma_eyes = nn.Parameter(sigmas_eyes.repeat((self.nb_classes * self.nb_components, 1, 1)), requires_grad=False)
        
        # prior probabilities
        priors = 1/self.nb_classes * torch.ones(self.nb_classes)
        if self.is_prior_uniform:
            self.priors = nn.Parameter(priors, requires_grad=False)
        else:
            self.priors = nn.Parameter(priors, requires_grad=True)

        # Gaussian component weigths
        if self.nb_components > 1 : 
            weights = 1/self.nb_components * torch.ones(self.nb_classes * self.nb_components)
            if self.is_weights_uniform:
                self.weights = nn.Parameter(weights, requires_grad=False)
            else:
                self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, x):

        batch_size = x.shape[0]

        self.apply_constraints()

        #lbd = lambda x: self.GMM_log_prob(x)

        #log = torch.func.vmap(lbd)(x)

        sigmas = torch.einsum("n,nij->nij", 1/self.bandwiths_constrained, self.sigma_eyes)

        distrib = torch.distributions.MultivariateNormal(loc=self.means, precision_matrix=sigmas)
        #distrib = torch.distributions.MultivariateNormal(loc=self.means, scale_tril=sigmas)

        temp_x = x.unsqueeze(1)

        log = distrib.log_prob(temp_x)

        if self.nb_components > 1:

            log = log + self.weigths_components.reshape(self.nb_classes * self.nb_components).log()

            log = log.reshape(batch_size, self.nb_classes, self.nb_components)

            log = log.logsumexp(dim=2, keepdim=False)

        log = log + self.priors_probabilities.reshape(self.nb_classes).log()

        log_mixture = log - log.logsumexp(dim=1, keepdim=True)

        return log_mixture
    
    def apply_constraints(self):

        self.bandwiths_constrained = torch.clamp(self.bandwidths, min=1e-3, max=1e2)

        if self.nb_components > 1:
            self.weigths_components = torch.softmax(self.weights.reshape(self.nb_classes, self.nb_components), dim=1)

        self.priors_probabilities = torch.softmax(self.priors, dim=0)
    
    def GMM_log_prob(self, value):
    
        diff_means = (value-self.means).reshape((self.nb_classes * self.nb_components, self.features_dim, 1))
        diff_means_T = torch.transpose(diff_means, 1,2)

        pi_element = self.features_dim * torch.log(torch.tensor(2*math.pi))
        
        determinant_element = self.features_dim * torch.log(self.bandwiths_constrained)

        MD_element = 1/self.bandwiths_constrained * torch.matmul(diff_means_T, diff_means).squeeze()

        log_probs = -0.5 * (pi_element + determinant_element + MD_element)
        
        return log_probs
    
class DGMMC_diagonal(nn.Module):
    def __init__(self, features_dim, nb_classes, nb_components, is_prior_uniform = False, is_weights_uniform = False, init_means = None, init_bandwidths = None, **kwargs):
        super(DGMMC_diagonal, self).__init__(**kwargs)

        assert features_dim > 0
        assert nb_classes > 0
        assert nb_components > 0

        # Hyperparameters
        self.features_dim = features_dim
        self.nb_classes = nb_classes
        self.nb_components = nb_components

        # Architecture parameters
        self.is_prior_uniform = is_prior_uniform
        self.is_weights_uniform = is_weights_uniform

        self.initialize_parameters(init_means, init_bandwidths)


    def initialize_parameters(self, init_means, init_bandwidths):

        # means
        if init_means is None:
            self.means = nn.Parameter(torch.randn(self.nb_classes * self.nb_components, self.features_dim), requires_grad=True)
        else:
            self.means = nn.Parameter(init_means, requires_grad=True)

        # bandwidths
        if init_bandwidths is None:
            self.bandwidths = nn.Parameter(0.15*torch.ones((self.nb_classes * self.nb_components, self.features_dim)), requires_grad=True)
        else:
            self.bandwidths = nn.Parameter(init_bandwidths, requires_grad=True)
        sigmas_eyes = torch.eye(self.features_dim).unsqueeze(0)
        self.sigma_eyes = nn.Parameter(sigmas_eyes.repeat((self.nb_classes * self.nb_components, 1, 1)), requires_grad=False)
        
        # prior probabilities
        priors = 1/self.nb_classes * torch.ones(self.nb_classes)
        if self.is_prior_uniform:
            self.priors = nn.Parameter(priors, requires_grad=False)
        else:
            self.priors = nn.Parameter(priors, requires_grad=True)

        # Gaussian component weigths
        if self.nb_components > 1 : 
            weights = 1/self.nb_components * torch.ones(self.nb_classes * self.nb_components)
            if self.is_weights_uniform:
                self.weights = nn.Parameter(weights, requires_grad=False)
            else:
                self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(self, x):

        batch_size = x.shape[0]

        self.apply_constraints()

        lbd = lambda xx: self.GMM_log_prob(xx)
        

        #log = torch.func.vmap(lbd)(x)

        sigmas = torch.einsum("ni,nij->nij", self.bandwiths_constrained, self.sigma_eyes)

        distrib = torch.distributions.MultivariateNormal(loc=self.means, covariance_matrix=sigmas)
        #distrib = torch.distributions.MultivariateNormal(loc=self.means, scale_tril=sigmas)

        temp_x = x.unsqueeze(1)

        log = distrib.log_prob(temp_x)

        '''for i in range(0, batch_size):
            test = distrib.log_prob(x[i])
            print(test)
            print(test_prob[i])'''

        

        if self.nb_components > 1:

            log = log + self.weigths_components.reshape(self.nb_classes * self.nb_components).log()

            log = log.reshape(batch_size, self.nb_classes, self.nb_components)

            log = log.logsumexp(dim=2, keepdim=False)

        log_prior = self.priors_probabilities.reshape(self.nb_classes).log()
        log = log + log_prior

        log_mixture = log - log.logsumexp(dim=1, keepdim=True)

        return log_mixture
    
    def apply_constraints(self):

        #self.bandwiths_constrained = torch.relu(self.bandwidths) + 1e-6
        self.bandwiths_constrained = torch.clamp(self.bandwidths, min=1e-6, max=1e3)

        if self.nb_components > 1:
            self.weigths_components = torch.softmax(self.weights.reshape(self.nb_classes, self.nb_components), dim=1)+1e-6

        self.priors_probabilities = torch.softmax(self.priors, dim=0)+1e-6
    
    def GMM_log_prob(self, value):
    
        diff_means = (value-self.means).reshape((self.nb_classes * self.nb_components, self.features_dim, 1))
        diff_means_T = torch.transpose(diff_means, 1,2)

        pi_element = self.features_dim * torch.log(torch.tensor(2*math.pi))
        
        #determinant_element = self.features_dim * torch.log(self.bandwiths_constrained)
        determinant_element = torch.log(self.bandwiths_constrained)
        determinant_element = torch.sum(determinant_element, dim=1, keepdim=False)
        #determinant_element = torch.logsumexp(determinant_element, dim=1, keepdim=False)

        #MD_element = 1/self.bandwiths_constrained * torch.matmul(diff_means_T, diff_means).squeeze()
        inverse_bandwidths = 1/self.bandwiths_constrained
        sigmas = torch.einsum("ni,nij->nij", inverse_bandwidths, self.sigma_eyes)
        MD_element = torch.matmul(diff_means_T, sigmas)
        MD_element = torch.matmul(MD_element, diff_means).squeeze()
        print(MD_element)

        log_probs = -0.5 * (pi_element + determinant_element + MD_element)
        
        return log_probs