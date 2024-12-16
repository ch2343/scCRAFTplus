import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scanpy as sc
import numpy as np
import umap
import torch.autograd as autograd
import scipy.sparse
import random
from sklearn.decomposition import PCA
import anndata
import pandas as pd
from typing import List
import time
from itertools import combinations
from torch.distributions import Normal, kl_divergence as kl
from torch.utils.data import DataLoader, TensorDataset,Dataset


torch.backends.cudnn.benchmark = True

from typing import Optional, Union
import collections
from typing import Iterable, List

from torch.distributions import Normal
from torch.nn import ModuleList
import jax.numpy as jnp






# Net + Loss function

def log_nb_positive(
    x: Union[torch.Tensor, jnp.ndarray],
    mu: Union[torch.Tensor, jnp.ndarray],
    theta: Union[torch.Tensor, jnp.ndarray],
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
        lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class Encoder(nn.Module):
    def __init__(self, p_dim, latent_dim=128):
        super(Encoder, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(p_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mean = nn.Linear(512, latent_dim)  # Output layer for mean
        self.fc_var = nn.Linear(512, latent_dim)   # Output layer for variance
        #self.fc_library = nn.Linear(512, 1)        # Output layer for library size
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        # Forward pass through the network
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.relu(self.bn2(x))

        # Separate paths for mean, variance, and library size
        q_m = self.fc_mean(x)
        q_v = torch.exp(self.fc_var(x)) + 1e-4
        #library = self.fc_library(x)  # Predicted log library size

        z = reparameterize_gaussian(q_m, q_v)
        
        return q_m, q_v, z



class Decoder(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim=256):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        
        # Main decoder pathway
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )
        
        # Additional pathway for the batch effect (ec)
        self.decoder_ec = nn.Sequential(
            nn.Linear(v_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, p_dim),
        )

        # Parameters for ZINB distribution
        self.px_scale_decoder = nn.Linear(p_dim, p_dim)  # mean (rate) of ZINB
        self.px_r_decoder = nn.Linear(p_dim, p_dim)  # dispersion

    def forward(self, z, ec):
        # Main decoding
        z_ec = torch.cat((z, ec), dim=-1)
        decoded = self.decoder(z_ec)
        decoded_ec = self.decoder_ec(ec)

        # Combining outputs
        combined = self.relu(decoded + decoded_ec)

        # NB parameters with safe exponential

        px_scale = torch.exp(self.px_scale_decoder(combined))
        px_r = torch.exp(self.px_r_decoder(combined))

        # Scale the mean (px_scale) with the predicted library size
        px_rate = px_scale
        
        return px_rate, px_r

   

class VAE(nn.Module):
    def __init__(self, p_dim, v_dim, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(p_dim, latent_dim)
        self.decoder = Decoder(p_dim, v_dim, latent_dim)


    def forward(self, x, ec):
        # Encoding
        q_m, q_v, z = self.encoder(x)

        # Decoding
        px_scale, px_r = self.decoder(z, ec)

        # Reconstruction Loss
        #reconst_loss = F.mse_loss(px_scale, x)
        reconst_loss = -log_nb_positive(x, px_scale, px_r)
        # KL Divergence
        mean = torch.zeros_like(q_m)
        scale = torch.ones_like(q_v)
        kl_divergence = kl(Normal(q_m, torch.sqrt(q_v)), Normal(mean, scale)).sum(dim=1)

        return reconst_loss, kl_divergence, z, px_scale


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log softmax to the output
        log_preds = F.log_softmax(output, dim=-1)
        
        # Compute the negative log likelihood loss
        loss = F.nll_loss(log_preds, target, reduction=self.reduction)
        
        return loss



class discriminator(nn.Module):
    def __init__(self, n_input, domain_number):
        super(discriminator, self).__init__()
        n_hidden = 128

        # Define layers
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, domain_number)
        self.loss = CrossEntropy()

    def forward(self, x, batch_ids, generator=False):
        # Forward pass through layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)
        softmax_probs = F.softmax(output, dim=1)
        
        D_loss = self.loss(output, batch_ids)
        if self.loss.reduction == 'mean':
             D_loss = D_loss.mean()
        elif self.loss.reduction == 'sum':
             D_loss = D_loss.sum()

        return D_loss

class Classifier(nn.Module):
    def __init__(self, n_input, label_number, xi=1e-6, eps=500, num_iters=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_input)
        self.fc2 = nn.Linear(n_input, n_input)
        self.fc3 = nn.Linear(n_input, label_number)
        self.xi = xi
        self.eps = eps
        self.num_iters = num_iters

    def forward(self, x, u=None, i=None, l=None, t=0.05, vat_coef=1, train=True, hard_label=False):
        logits = self.compute_logits(x)
        if train:
            if hard_label:
                cross_entropy_loss = 0 if l is None else self.compute_cross_entropy_loss(logits, l)
                vat_loss_value = self.compute_vat_loss(x, logits)  # Pass logits to VAT loss computation
                total_loss = cross_entropy_loss + 5 * vat_loss_value
            else:
                u = torch.softmax(u, dim=1)
                u = u**(1/t)
                u_labels = u / u.sum(dim=1, keepdim=True)
                valid_indices = (i == 0)

                if not valid_indices.any():
                    return torch.tensor(0.0, device=logits.device)

                #u_labels_filtered = u_labels[valid_indices]
                #logits_filtered = logits[valid_indices]
            
                cross_entropy_loss = F.cross_entropy(logits, u_labels)
                vat_loss_value = self.compute_vat_loss_filter(x, logits, i)  # Pass logits to VAT loss computation
                total_loss = cross_entropy_loss + vat_coef * vat_loss_value
        else:
            total_loss = 0
        return total_loss, logits

    def compute_logits(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)
        return logits


    def compute_vat_loss(self, x, logit):
        # Use the passed logits for VAT calculation
        vat_loss_value = vat_loss(self, x, logit, self.xi, self.eps, self.num_iters)
        return vat_loss_value

    def compute_vat_loss_filter(self, x, logit, i):
        # Use the passed logits for VAT calculation
        vat_loss_value = vat_loss_filter(self, x, logit, i, self.xi, self.eps, self.num_iters)
        return vat_loss_value

    def compute_cross_entropy_loss(self, logits, labels):
        # Filter out entries with placeholder label (-1) used for NaN
        valid_indices = labels != -1
        if not valid_indices.any():
            return torch.tensor(0.0).to(logits.device)
    
        labeled_logits = logits[valid_indices]
        labeled_labels = labels[valid_indices]

        # Ensure there are labeled entries before computing the loss
        if len(labeled_labels) > 0:
            loss = F.cross_entropy(labeled_logits, labeled_labels)
        else:
            loss = torch.tensor(0.0).to(logits.device)
    
        return loss



def vat_loss(model, x, original_logits, xi, eps, num_iters):
    d = torch.rand_like(x).sub(0.5)
    d.requires_grad_()  # Ensure d requires gradient
    for _ in range(num_iters):
        d.requires_grad_()
        perturbed_x = x.detach() + xi * d  # Ensure x is detached
        perturbed_logits = model.compute_logits(perturbed_x)
        kl_div = kl_div_with_logit(original_logits.detach(), perturbed_logits)
        kl_div.backward()

        if d.grad is not None:
            grad_d = d.grad.data  # Get the gradient data
            d = _l2_normalize(grad_d)  # Normalize the gradient to get the new direction
            d = d.clone().detach().requires_grad_(True)  # Detach and require grad for the next iteration
        else:
            raise RuntimeError("Gradient for d is None, VAT loss cannot be computed")
    r_adv = eps * d
    perturbed_x = x + r_adv
    perturbed_logits = model.compute_logits(perturbed_x)
    vat_loss_value = kl_div_with_logit(original_logits.detach(), perturbed_logits)

    return vat_loss_value

def vat_loss_filter(model, x, logits,i, xi=1e-6, eps=10.0, num_iters=1, temperature=0.5, tau_e=-1):
    # Compute the energy
    energies = -temperature * torch.logsumexp(logits / temperature, dim=1)

    # Check if the energies are below the threshold
    #valid_indices = energies < tau_e
    #valid_indices = (energies < tau_e) | (i == 0)
    valid_indices = (i == 0)
    
    if not valid_indices.any():
        return torch.tensor(0.0, device=logits.device)

    # Apply VAT only to data points where energy is below the threshold
    x_filtered = x[valid_indices]
    logits_filtered = logits[valid_indices]

    d = torch.rand_like(x_filtered).sub(0.5)
    d = _l2_normalize(d)

    for _ in range(num_iters):
        d.requires_grad_()
        perturbed_x = x_filtered.detach() + xi * d
        perturbed_logits = model.compute_logits(perturbed_x)
        kl_div = kl_div_with_logit(logits_filtered.detach(), perturbed_logits)
        kl_div.backward()
        grad_d = d.grad.data 
        d = _l2_normalize(d.grad.data)
        d = d.clone().detach().requires_grad_(True)

    r_adv = eps * d
    perturbed_x = x_filtered + r_adv
    perturbed_logits = model.compute_logits(perturbed_x)
    vat_loss_value = kl_div_with_logit(logits_filtered.detach(), perturbed_logits)

    return vat_loss_value
    

def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)
    qlogq = (q * logq).sum(dim=1).mean()
    qlogp = (q * logp).sum(dim=1).mean()
    return qlogq - qlogp

def _l2_normalize(d):
    d /= (torch.sqrt(torch.sum(d ** 2)) + 1e-16)
    return d


