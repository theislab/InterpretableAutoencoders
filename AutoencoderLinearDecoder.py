import torch
import torch.nn as nn

# Proximal operators
class ProxOperGroupL2:
    def __init__(self, alpha, omega, inplace=True):
    # omega - vector of coefficients with size
    # equal to the number of groups
        self._group_vect = omega*alpha
        self._inplace = inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        norm_vect = W.norm(p=2, dim=0)
        norm_g_gr_vect = norm_vect>self._group_vect

        scaled_norm_vector = norm_vect/self._group_vect.view(-1)
        scaled_norm_vector+=(~(scaled_norm_vector>0)).float()

        W-=W/scaled_norm_vector
        W*=norm_g_gr_vect.float()

        return W

class ProxOperL1:
    def __init__(self, alpha, I, inplace=True):
        self._alpha=alpha
        self._I = ~I.bool()
        self._inplace=inplace

    def __call__(self, W):
        if not self._inplace:
            W = W.clone()

        W_geq_alpha = W>=self._alpha
        W_leq_neg_alpha = W<=-self._alpha

        W-=(W_geq_alpha&self._I).float()*self._alpha
        W+=(W_leq_neg_alpha&self._I).float()*self._alpha
        W-=(~W_geq_alpha&~W_leq_neg_alpha&self._I).float()*W

        return W

# Autoencoder with regularized linear decoder
class AutoencoderLinearDecoder(nn.Module):
    def __init__(self, n_vars, n_terms, **kwargs):
        super().__init__()

        self.n_vars = n_vars
        self.n_terms = n_terms
        self.dropout_rate = kwargs.get("dropout_rate", 0.2)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_vars, 400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(400, self.n_terms)
        )

        self.decoder = nn.Linear(self.n_terms, self.n_vars, bias=False)

        self.decoder.weight.data.normal_()
        self.decoder.weight.data/=self.n_terms**0.5

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

def get_prox_operator(I, alpha1, alpha2, omega):
    prox_op_l1 = ProxOperL1(alpha1, I)
    prox_op_group = ProxOperGroupL2(alpha2, omega)
    return lambda W: prox_op_group(prox_op_l1(W))

def train_autoencoder(adata, autoencoder, l2_reg_alpha, lr, batch_size, num_epochs, prox_operator):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(autoencoder.parameters(), lr=lr)

    t_X = torch.from_numpy(adata.X)
    n_obs = adata.n_obs

    zeros = torch.zeros((batch_size, autoencoder.n_terms))

    l2_loss = nn.MSELoss(reduction='sum')

    for epoch in range(num_epochs):
        autoencoder.train()

        for step in range(int(adata.n_obs/batch_size)):
            X = torch.from_numpy(adata.chunk_X(batch_size))

            encoded, decoded = autoencoder(X)

            loss = (l2_loss(decoded, X)+l2_reg_alpha*l2_loss(encoded, zeros))/batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prox_operator(autoencoder.decoder.weight.data)

            if step % 100 == 0:
                print('Epoch:', epoch, '| batch train loss: %.4f' % loss.data.numpy())

        autoencoder.eval()
        t_encoded, t_decoded = autoencoder(t_X)

        t_reconst = l2_loss(t_decoded, t_X).data.numpy()/n_obs
        t_regul = l2_reg_alpha*l2_loss(t_encoded, torch.zeros_like(t_encoded)).data.numpy()/n_obs
        t_loss = t_reconst + l2_reg_alpha*t_regul

        print('Epoch:', epoch, '-- total train loss: %.4f=%.4f+%.4f' % (t_loss, t_reconst, t_regul))

        n_deact_terms = (~(autoencoder.decoder.weight.data.norm(p=2, dim=0)>0)).float().sum().numpy()
        print('Number of deactivated terms:', int(n_deact_terms))

        n_deact_genes = (~(autoencoder.decoder.weight.data>0)).float().sum().numpy()
        print('Number of deactivated genes:', int(n_deact_genes))
