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
import sys

# Dynamic import of tqdm based on the environment
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
from scCRAFTplus.networks import *
from scCRAFTplus.utils import *
    
    
class SCIntegrationModel(nn.Module):
    def __init__(self, adata, batch_key, z_dim, hard_label):
        super(SCIntegrationModel, self).__init__()
        self.p_dim = adata.shape[1]
        self.z_dim = z_dim
        self.v_dim = np.unique(adata.obs[batch_key]).shape[0]
        if hard_label:
            non_nan_labels = adata.obs["hard_label"].dropna()
            unique_labels = np.unique(non_nan_labels)

            # Set the dimensionality of the label space excluding NaN
            self.u_dim = len(unique_labels)
        else:
            self.u_dim = adata.obsm['U_scores'].shape[1]

        # Correctly initialize VAE with p_dim, v_dim, and latent_dim
        self.VAE = VAE(p_dim=self.p_dim, v_dim=self.v_dim, latent_dim=self.z_dim)
        self.D_Z = discriminator(self.z_dim, self.v_dim)
        self.C = Classifier(self.z_dim, label_number=self.u_dim)
        self.mse_loss = torch.nn.MSELoss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move models to CUDA if available
        self.VAE.to(self.device)
        self.D_Z.to(self.device)
        self.C.to(self.device)

        # Initialize weights
        self.VAE.apply(weights_init_normal)
        self.D_Z.apply(weights_init_normal)
        self.C.apply(weights_init_normal)

    def train_model(self, adata, batch_key, epochs, d_coef, kl_coef, warmup_epoch, hard_label):
        # Optimizer for VAE (Encoder + Decoder)
        optimizer_G_C = optim.Adam(
            list(self.VAE.parameters()) + list(self.C.parameters()), 
            lr=0.001, 
            weight_decay=0.0
        )
        # Optimizer for Discriminator
        optimizer_D_Z = optim.Adam(self.D_Z.parameters(), lr=0.001, weight_decay=0.)

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

        progress_bar = tqdm(total=epochs, desc="Overall Progress", leave=True)
        for epoch in range(epochs):
                set_seed(epoch)
                if hard_label:
                    data_loader = generate_balanced_dataloader_hardlabel(adata, batch_size = 512, batch_key= batch_key)
                else:
                    data_loader = generate_balanced_dataloader(adata, batch_size = 512, batch_key= batch_key)
                self.VAE.train()
                self.D_Z.train()
                self.C.train()
                all_losses = 0
                D_loss = 0
                T_loss = 0
                V_loss = 0
                
                for i, data in enumerate(data_loader):
                    if hard_label:
                        x, v, labels_low, labels_high, l = data
                    else:
                        x, v, i, labels_low, labels_high, u = data
                        i = i.to(self.device)
                    x = x.to(self.device)
                    v = v.to(self.device)
                    labels_low = labels_low.to(self.device)
                    labels_high = labels_high.to(self.device)
                    batch_size = x.size(0)
                    v_true = v
                    v_one_hot = torch.zeros(batch_size, self.v_dim).to(x.device)
                    # Use scatter_ to put 1s in the indices indicated by v
                    v = v.unsqueeze(1)  # Ensure v is of shape [batch_size, 1] if it's not already
                    
                    v_one_hot.scatter_(1, v, 1).to(v.device)

                    reconst_loss, kl_divergence, z, x_tilde = self.VAE(x, v_one_hot)
                    reconst_loss = torch.clamp(reconst_loss, max = 1e5)
                    
                    
                    loss_cos = (1 - torch.sum(F.normalize(x_tilde, p=2) * F.normalize(x, p=2), 1)).mean()
                    loss_VAE = torch.mean(reconst_loss.mean() + kl_coef * kl_divergence.mean())

                    
                    for disc_iter in range(10):
                        optimizer_D_Z.zero_grad()
                        loss_D_Z = self.D_Z(z, v_true)
                        loss_D_Z.backward(retain_graph=True)
                        optimizer_D_Z.step()
                    if hard_label:
                        classifier_loss,_ = self.C(z, l = l, hard_label = True)
                    else:
                        classifier_loss,_ = self.C(z, u = u, i = i, hard_label = False)
                    
                    optimizer_G_C.zero_grad()
                    loss_DA = self.D_Z(z, v_true, generator = False)
                    triplet_loss = create_triplets(z, labels_low, labels_high, v_true, margin = 5)
                    if epoch < warmup_epoch:
                        all_loss = - 0 * loss_DA + 1 * loss_VAE + 1 * triplet_loss + 20 * loss_cos
                    else:
                        all_loss = - d_coef * loss_DA + 1 * loss_VAE + 1 * triplet_loss + 20 * loss_cos + 1 * classifier_loss
                        
                    all_loss.backward()
                    optimizer_G_C.step()
                    all_losses += all_loss
                    D_loss += loss_DA
                    T_loss += triplet_loss
                    V_loss += loss_VAE
                    # Update the overall progress after each batch instead of after each epoch
                progress_bar.update(1)  # Increment the progress bar by one for each batch processed

                # Optionally, set the postfix to display the current epoch and loss
                progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", all_loss=all_losses.item(), disc_loss=D_loss.item())
        progress_bar.close()




def train_integration_model(adata, batch_key='batch', z_dim=256, epochs = 150, d_coef = 0.2, kl_coef = 0.005, warmup_epoch = 50, hard_label = False):
    number_of_cells = adata.n_obs
    number_of_batches = np.unique(adata.obs[batch_key]).shape[0]
    # Default number of epochs
    if epochs == 150:
        # Check if the number of cells goes above 100000
        if number_of_cells > 100000:
            calculated_epochs = int(1.5 * number_of_cells / (number_of_batches * 512))
            # If the calculated value is larger than the default, use it instead
            if calculated_epochs > epochs:
                epochs = calculated_epochs
    else:
        epochs = epochs
    model = SCIntegrationModel(adata, batch_key, z_dim = z_dim, hard_label = hard_label)
    print(epochs)
    start_time = time.time() 
    model.train_model(adata, batch_key=batch_key, epochs=epochs, d_coef = d_coef, kl_coef = kl_coef, warmup_epoch = warmup_epoch, hard_label = hard_label)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    model.VAE.eval()
    model.C.eval()
    return model.VAE, model.C

# Loader only for Visualization
def generate_adata_to_dataloader(adata, batch_size=2048):
    print("Generating data loader...")  # Debug print

    if isinstance(adata.X, scipy.sparse.spmatrix):
        print("Data is sparse, converting to dense...")  # Debug print
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    
    # Create a DataLoader for batch-wise processing
    dataset = torch.utils.data.TensorDataset(X_tensor, torch.arange(X_tensor.size(0)))  # include indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loader created with {len(data_loader)} batches.")  # Debug print

    return data_loader
    

def obtain_embeddings(adata, VAE, C, cell_types_markers, dim=50, temperature=1.0, tau_e=-2):
    VAE.eval()
    C.eval()
    data_loader = generate_adata_to_dataloader(adata)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store the original categorical labels and add 'Unknown' for uncertain predictions
    #label_mapping = adata.obs[label_key].astype('category').cat.categories.tolist()
    
    all_z = []
    all_l = []
    all_indices = []
    for i, (x, indices) in enumerate(data_loader):
        x = x.to(device)
        _, _, z = VAE.encoder(x)
        _, l = C(z, train = False)
        all_z.append(z.cpu().detach())
        all_l.append(l.cpu().detach())
        all_indices.extend(indices.tolist())
        
    all_z_combined = torch.cat(all_z, dim=0)
    all_l_combined = torch.cat(all_l, dim=0)
    all_indices_tensor = torch.tensor(all_indices)
    
    # Reorder according to original indices
    all_z_reordered = all_z_combined[all_indices_tensor.argsort()]
    all_l_reordered = all_l_combined[all_indices_tensor.argsort()]
    
    # Map integer labels back to original categorical labels including 'Unknown'
    #all_l_mapped = np.array(label_mapping)[all_l_reordered.numpy()]

    # Create anndata object with reordered embeddings and labels
    adata.obsm['X_scCRAFT'] = all_z_reordered.numpy()
    # Assuming 'C_logits' are already in adata.obsm and aligned in order
    logits_matrix = all_l_reordered.numpy()
    max_logits_indices = np.argmax(logits_matrix, axis=1)

    # Map indices to cell types
    cell_types = list(cell_types_markers.keys())   # Ensure this is correctly ordered as in logits
    predicted_labels = np.array(cell_types)[max_logits_indices]
    energies = -temperature * np.log(np.sum(np.exp(logits_matrix / temperature), axis=1))
    uncertain = energies > tau_e
    predicted_labels[uncertain] = 'Unknown'

    unique_labels = np.unique(predicted_labels)
    adata.obs['Predict_label'] = pd.Categorical(predicted_labels, categories=unique_labels)

    
    # PCA reduction
    pca = PCA(n_components=dim)
    X_scCRAFT_pca = pca.fit_transform(adata.obsm['X_scCRAFT'])
    adata.obsm['X_scCRAFT_pca'] = X_scCRAFT_pca
    
    sc.pp.neighbors(adata, use_rep="X_scCRAFT")
    sc.tl.louvain(adata, resolution=3)

    # Create a DataFrame with louvain and Predict_label
    df = adata.obs[['louvain', 'Predict_label']]

    # Calculate the proportion of each Predict_label within each louvain label
    proportions = df.groupby(['louvain', 'Predict_label']).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)

    # Assign Predict_label if proportion > 80%
    for louvain_label, row in proportions.iterrows():
        if (row > 0.8).any():
            predict_label = row.idxmax()
            adata.obs.loc[adata.obs['louvain'] == louvain_label, 'Predict_label'] = predict_label

    return adata

def obtain_embeddings_hard_label(adata, VAE, C, dim=50, temperature=1.0, tau_e=-2, hard_label_key="hard_label"):
    VAE.eval()
    C.eval()
    data_loader = generate_adata_to_dataloader(adata)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store the original categorical labels and add 'Unknown' for uncertain predictions
    #label_mapping = adata.obs[label_key].astype('category').cat.categories.tolist()
    
    all_z = []
    all_l = []
    all_indices = []
    # Collect original hard labels
    for i, (x, indices) in enumerate(data_loader):
        x = x.to(device)
        _, _, z = VAE.encoder(x)
        _, l = C(z, train = False)
        all_z.append(z.cpu().detach())
        all_l.append(l.cpu().detach())
        all_indices.extend(indices.tolist())
        
    all_z_combined = torch.cat(all_z, dim=0)
    all_l_combined = torch.cat(all_l, dim=0)
    all_indices_tensor = torch.tensor(all_indices)
    
    # Reorder according to original indices
    all_z_reordered = all_z_combined[all_indices_tensor.argsort()]
    all_l_reordered = all_l_combined[all_indices_tensor.argsort()]
    
    # Map integer labels back to original categorical labels including 'Unknown'
    adata.obsm['X_scCRAFT'] = all_z_reordered.numpy()
    # Assuming 'C_logits' are already in adata.obsm and aligned in order
    logits_matrix = all_l_reordered.numpy()
    max_logits_indices = np.argmax(logits_matrix, axis=1)

    cell_types = adata.obs[hard_label_key].cat.categories.tolist()
    predicted_labels = np.array(cell_types)[max_logits_indices]
    energies = -temperature * np.log(np.sum(np.exp(logits_matrix / temperature), axis=1))
    uncertain = energies > tau_e
    predicted_labels[uncertain] = 'Unknown'

    unique_labels = np.unique(predicted_labels)
    adata.obs['Predict_label'] = pd.Categorical(predicted_labels, categories=unique_labels)


    
    # PCA reduction
    pca = PCA(n_components=dim)
    X_scCRAFT_pca = pca.fit_transform(adata.obsm['X_scCRAFT'])
    adata.obsm['X_scCRAFT_pca'] = X_scCRAFT_pca
    
    sc.pp.neighbors(adata, use_rep="X_scCRAFT")
    sc.tl.louvain(adata, resolution=3)

    # Create a DataFrame with louvain and Predict_label
    df = adata.obs[['louvain', 'Predict_label']]

    # Calculate the proportion of each Predict_label within each louvain label
    proportions = df.groupby(['louvain', 'Predict_label']).size().unstack(fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)
    # Assign Predict_label if proportion > 80%
    for louvain_label, row in proportions.iterrows():
        if (row > 0.8).any():
            predict_label = row.idxmax()
            adata.obs.loc[adata.obs['louvain'] == louvain_label, 'Predict_label'] = predict_label

    return adata
