import torch
import torch.nn as nn
import numpy as np

class AutoencoderNonlinear(nn.Module):
    def __init__(self, n_vars, n_latent, **kwargs):
        super().__init__()

        self.n_vars = n_vars
        self.n_latent = n_latent
        
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)

        self.mid_layers_size = kwargs.get('mid_layers_size', 600)
        self.after_mid_size = int(self.mid_layers_size/2)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_vars, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.after_mid_size),
            nn.BatchNorm1d(self.after_mid_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.after_mid_size, self.n_latent)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.after_mid_size),
            nn.BatchNorm1d(self.after_mid_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.after_mid_size, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.n_vars)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

def get_jacobian(net, x, noutputs):
    x.requires_grad_(True)
    y = net(x)
    for i in range(noutputs):
        grad_output = torch.zeros((x.shape[0], noutputs))
        grad_output[:, i] = 1.
        jac = torch.autograd.grad(y, x, grad_output, create_graph=True)
        yield jac[0]

def index_iter(n_obs, batch_size):
    indices = np.random.permutation(n_obs)
    for i in range(0, n_obs, batch_size):
        yield indices[i: min(i + batch_size, n_obs)]

def train_autoencoder(adata, autoencoder, lr, batch_size, num_epochs, lambda_=0.5,
                      test_data=None, optim = torch.optim.Adam, **kwargs):
    
    optimizer = optim(autoencoder.parameters(), lr=lr, **kwargs)

    I = torch.from_numpy(1-adata.varm['I']).float()

    if test_data is None:
        t_X = torch.from_numpy(adata.X)
        comment = '-- total train reconstruction loss: '
    else:
        t_X = test_data
        comment = '-- test reconsruction loss:'
    test_n_obs = t_X.shape[0]

    l2_loss = nn.MSELoss(reduction='sum')
    
    if adata.isbacked:
        select_X = lambda adata, selection: torch.from_numpy(adata[selection].X)
    else:
        select_X = lambda adata, selection: torch.from_numpy(adata.X[selection])

    for epoch in range(num_epochs):
        autoencoder.train()

        for step, selection in enumerate(index_iter(adata.n_obs, batch_size)):

            X = select_X(adata, selection)

            encoded, decoded = autoencoder(X)

            loss = l2_loss(decoded, X)
            
            for i, jac in enumerate(get_jacobian(autoencoder.decoder, encoded.detach(), adata.n_vars)):
                loss = loss + lambda_*torch.norm(jac*I[i], p=1)

            loss = loss / len(selection)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print('Epoch:', epoch, '| Step:', step, '| batch train loss: %.4f' % loss.data.numpy())

        autoencoder.eval()
        t_encoded, t_decoded = autoencoder(t_X)

        t_reconst = l2_loss(t_decoded, t_X).data.numpy()/test_n_obs

        print('Epoch:', epoch, comment, '%.4f' % t_reconst)