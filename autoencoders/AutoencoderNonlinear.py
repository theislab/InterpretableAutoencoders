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

def get_jacobian(out, inpt):
    noutputs = out.shape[1]
    o_grad = torch.ones(out.shape[0])
    for i in range(noutputs):
        o = out[:, i]
        jac = torch.autograd.grad(o, inpt, o_grad, create_graph=True)
        yield i, jac[0]

class ExpectedJacobian:

    def __init__(self, adata, autoencoder):
        self.adata = adata
        self.autoencoder = autoencoder

    def get_jacobian(self, inpt, full=False):
        batch_size = inpt.shape[0]
        ref_batch = torch.from_numpy(self.adata.chunk_X(batch_size))

        z_prime = self.autoencoder.encoder(ref_batch)
        z_input = self.autoencoder.encoder(inpt)

        z_diff = z_input - z_prime

        if not full:
            z_diff = z_diff.detach()

        unif_coef = torch.rand(batch_size, 1)
        z_interpl = z_prime + z_diff*unif_coef

        self.z_interpl_fun = z_interpl

        z_interpl_var = z_interpl.detach()
        z_interpl_var.requires_grad = True

        self.z_interpl_var = z_interpl_var

        out = self.autoencoder.decoder(z_interpl_var)
        noutputs = out.shape[1]
        o_grad = torch.ones(out.shape[0])

        for i in range(noutputs):
            o = out[:, i]
            jac = torch.autograd.grad(o, z_interpl_var, o_grad, create_graph=True)[0]*z_diff
            yield i, jac

def index_iter(n_obs, batch_size):
    indices = np.random.permutation(n_obs)
    for i in range(0, n_obs, batch_size):
        yield indices[i: min(i + batch_size, n_obs)]

def train_autoencoder(adata, autoencoder, lr, batch_size, num_epochs,
                      lambda_=0.5, lambda_1=None, full_grads=False, expected=False,
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

    if expected:
        exp_jac = ExpectedJacobian(adata, autoencoder)

    for epoch in range(num_epochs):
        autoencoder.train()

        for step, selection in enumerate(index_iter(adata.n_obs, batch_size)):

            X = select_X(adata, selection)

            if not expected:
                encoded_fun = autoencoder.encoder(X)
                encoded_var = encoded_fun.detach()
                encoded_var.requires_grad = True

                decoded = autoencoder.decoder(encoded_var)

                loss = l2_loss(decoded, X)
                if lambda_1 is not None:
                    loss = loss + lambda_1*encoded_var.pow(2).sum()
                loss = loss / len(selection)

                regz = sum(torch.norm(jac*I[i], p=1) for i, jac in get_jacobian(decoded, encoded_var))
                regz = lambda_ * regz / len(selection)

                optimizer.zero_grad()
                if full_grads:
                    t_loss = loss + regz
                    t_loss.backward()
                else:
                    regz.backward(retain_graph=True)
                    encoded_var.grad.zero_()
                    loss.backward()

                encoded_fun.backward(encoded_var.grad)

            else:
                encoded, decoded = autoencoder(X)
                loss = l2_loss(decoded, X)
                if lambda_1 is not None:
                     loss = loss + lambda_1*encoded.pow(2).sum()
                loss = loss / len(selection)

                regz = sum(torch.norm(jac*I[i], p=1) for i, jac in exp_jac.get_jacobian(X, full_grads))
                regz = lambda_ * regz / len(selection)

                t_loss = loss + regz

                optimizer.zero_grad()
                t_loss.backward(retain_graph=full_grads)
                if full_grads:
                    exp_jac.z_interpl_fun.backward(exp_jac.z_interpl_var.grad)

            optimizer.step()

            if step % 50 == 0:
                regz = regz.data.numpy()
                loss = loss.data.numpy()
                t_loss = loss + regz
                print('Epoch:', epoch, '| Step:', step, '| batch train loss: %.4f=%.4f+%.4f' % (t_loss, loss, regz))

        autoencoder.eval()
        t_encoded, t_decoded = autoencoder(t_X)

        t_reconst = l2_loss(t_decoded, t_X).data.numpy()/test_n_obs

        print('Epoch:', epoch, comment, '%.4f' % t_reconst)
