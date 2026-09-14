import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TabularVAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=8):
        super(TabularVAE, self).__init__()
        
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 16)
        self.enc_fc2 = nn.Linear(16, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 16)
        self.dec_fc2 = nn.Linear(16, 16)
        self.dec_out = nn.Linear(16, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        h = self.relu(self.enc_fc1(x))
        h = self.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        return self.dec_out(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    mse = nn.MSELoss(reduction='sum')(recon_x, x)
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld, mse, kld

class VAEGenerator:
    def __init__(self, input_dim=5, latent_dim=8):
        self.model = TabularVAE(input_dim, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.latent_dim = latent_dim
        
    def train(self, data, epochs=500):
        self.model.train()
        tensor_data = torch.FloatTensor(data)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(tensor_data)
            loss, mse, kld = vae_loss(recon_batch, tensor_data, mu, logvar)
            loss.backward()
            self.optimizer.step()
            
        return loss.item(), mse.item(), kld.item()
        
    def sample_ood(self, num_samples, sigma_threshold=2.5):
        self.model.eval()
        with torch.no_grad():
            # Generate z with high magnitude to force OOD
            # Normal distribution but push magnitude out
            z_base = torch.randn(num_samples, self.latent_dim)
            # Push vectors to have norm corresponding to > sigma_threshold per dimension roughly
            z_ood = z_base * sigma_threshold * 1.5 
            
            generated = self.model.decode(z_ood)
            return generated.numpy(), z_ood.numpy()
            
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
