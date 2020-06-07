import torch
import torch.nn as nn

threshold =  nn.Threshold(0.5, 0)

class STMaskedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, logits, weight, bias=None):
        mask = torch.bernoulli(torch.sigmoid(logits))
        
        ctx.save_for_backward(input, mask, weight, bias)
        
        return nn.functional.linear(input, weight*mask, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, mask, weight, bias = ctx.saved_tensors
        
        grad_input = grad_logits = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight*mask)
        if ctx.needs_input_grad[1]:
            grad_logits = grad_output.t().mm(input)*weight
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(input)*mask
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return grad_input, grad_logits, grad_weight, grad_bias
    
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_sparse, out_dense=0, estimator='ST', f_eval='Mean', bias=True):
        if estimator not in ('ST', 'SF'):
            raise ValueError('Unknown estimator')
        if f_eval not in ('Mean', 'Mode'):
            raise ValueError('Unknown evaluation parameter')

        super().__init__(in_features, out_sparse+out_dense, bias)
        
        self.logits = nn.Parameter(torch.Tensor(out_sparse, in_features))
        self.logits.data.zero_()
        
        self._sparse_dense = (out_sparse, out_dense)
        
        self.estimator = estimator
        if self.estimator == 'SF':
            self.mask_distr = torch.distributions.bernoulli.Bernoulli(logits=self.logits)
            self.last_mask = None
            
        self.f_eval = f_eval
        
    def forward(self, input):
        d_sparse, d_dense = self._sparse_dense
        if d_dense > 0:
            sp_weight = self.weight[:d_sparse]
            ds_weight = self.weight[d_sparse:]
            sp_bias = None if self.bias is None else self.bias[:d_sparse]
            ds_bias = None if self.bias is None else self.bias[d_sparse:]
            
        if not self.training:
            mask = self.eval_mask()
            if d_dense == 0:
                return nn.functional.linear(input, self.weight*mask, self.bias)
            else:
                sp_part = nn.functional.linear(input, sp_weight*mask, sp_bias)
                ds_part = nn.functional.linear(input, ds_weight, ds_bias)
                return torch.cat((sp_part, ds_part), dim=1)
        if self.estimator == 'ST':
            if d_dense == 0:
                return STMaskedLinearFunction.apply(input, self.logits, self.weight, self.bias)
            else:
                sp_part = STMaskedLinearFunction.apply(input, self.logits, sp_weight, sp_bias)
                ds_part = nn.functional.linear(input, ds_weight, ds_bias)
                return torch.cat((sp_part, ds_part), dim=1)
        elif self.estimator == 'SF':
            mask = self.mask_distr.sample()
            self.last_mask = mask
            if d_dense == 0:
                return nn.functional.linear(input, self.weight*mask, self.bias)
            else:
                sp_part = nn.functional.linear(input, sp_weight*mask, sp_bias)
                ds_part = nn.functional.linear(input, ds_weight, ds_bias)
                return torch.cat((sp_part, ds_part), dim=1)                
        
    def log_prob(self):
        if self.estimator != 'SF':
            raise ValueError('Estimator should be SF')
        return self.mask_distr.log_prob(self.last_mask)
    
    def eval_mask(self):
        return torch.sigmoid(self.logits).detach() if self.f_eval=='Mean' else (self.logits>=0).float()

class ThresholdLinear(nn.Linear):
    def __init__(self, in_features, out_sparse, out_dense=0, bias=True):
        super().__init__(in_features, out_sparse+out_dense, bias)
        
        self.logits = nn.Parameter(torch.Tensor(out_sparse, in_features))
        self.logits.data.fill_(0.4)
        
        self._has_dense = out_dense > 0
        
    def forward(self, input):
        if self._has_dense:
            mask = torch.cat((self.get_mask(), self._dense_part))
        else:
            mask = self.get_mask()
        return nn.functional.linear(input, self.weight*mask, self.bias)
    
    def get_mask(self):
        return threshold(torch.sigmoid(self.logits))
    
class DetMaskLinear(nn.Linear):
    def __init__(self, I, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = I.t()
    def forward(self, input):
        return nn.functional.linear(input, self.weight*self.mask, self.bias)

# Autoencoder with two masked linear layers        
class MaskedAutoencoder(nn.Module):
    def __init__(self, n_vars, n_terms, n_dense, n_latent, estimator='ST', f_eval='Mode'):
        super().__init__()
        
        self.encoder = nn.Sequential(
            MaskedLinear(n_vars, n_terms, n_dense, estimator, f_eval, bias=False),
            nn.ELU(),
            MaskedLinear(n_terms+n_dense, n_latent, 0, estimator, f_eval)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_terms),
            nn.ELU(),
            nn.Linear(n_terms, n_vars),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# get loss function for MaskedAutoencoder, pass the return value to train_autoencoder_masked       
def get_loss_func_masked(I, alpha1, alpha2, alpha3, alpha4):
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    def regularized_loss(X, Y, logits1, logits2):
        sigm1 = torch.sigmoid(logits1)
        sigm2 = torch.sigmoid(logits2)
        return l2_loss(X, Y), alpha1*bce_loss(logits1, I.t())+alpha2*torch.mean(sigm1*(1-sigm1)), alpha3*torch.mean(sigm2)+alpha4*torch.mean(sigm2*(1-sigm2))
    
    return regularized_loss

# train MaskedAutoencoder     
def train_autoencoder_masked(adata, autoencoder, loss_func, lr, epochs, batch_size):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    t_X = torch.from_numpy(adata.X)
    
    logits1 = autoencoder.encoder[0].logits
    logits2 = autoencoder.encoder[2].logits

    for epoch in range(epochs):
        autoencoder.train()
        for step in range(int(adata.n_obs/batch_size)):
            batch = torch.from_numpy(adata.chunk_X(batch_size))
            optimizer.zero_grad()
            batch_loss = 0
            
            for sample in batch:
                sample = sample[None, :]
                encoded, decoded = autoencoder(sample)
                
                loss = sum(loss_func(decoded, sample, logits1, logits2))/batch_size
                loss.backward()
                
                batch_loss += loss.data
            
            optimizer.step()
            if step % 100 == 0: print('Epoch: ', epoch, '| batch train loss: %.4f' % batch_loss.numpy())
        autoencoder.eval()
        _, t_decoded = autoencoder(t_X)
        
        t_loss = loss_func(t_decoded, t_X, logits1, logits2)
        t_loss = [sum(t_loss)] + list(t_loss)
        t_loss = [l.data.numpy() for l in t_loss]
        
        print('Epoch: ', epoch, '-- total train loss: %.4f=%.4f+%.4f+%.4f' % tuple(t_loss))

# Autoencoder with one masked linear layer        
class MaskedLinAutoencoder(nn.Module):
    def __init__(self, n_vars, n_terms, n_dense, n_latent, estimator='ST', f_eval='Mode'):
        super().__init__()
        
        self.encoder = nn.Sequential(
            MaskedLinear(n_vars, n_terms, n_dense, estimator, f_eval, bias=False),
            nn.ELU(),
            nn.Linear(n_terms+n_dense, n_latent)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_terms+n_dense),
            nn.ELU(),
            nn.Linear(n_terms+n_dense, n_vars),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# get loss function for MaskedLinAutoencoder, pass the return value to train_autoencoder_masked_lin        
def get_loss_func_masked_lin(I, alpha1, alpha2):
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    def regularized_loss(X, Y, logits):
        sigm1 = torch.sigmoid(logits)
        return l2_loss(X, Y), alpha1*bce_loss(logits, I.t())+alpha2*torch.mean(sigm1*(1-sigm1))
    
    return regularized_loss

# train MaskedLinAutoencoder
# todo - this is almost the same as train_autoencoder_masked, need to remove
def train_autoencoder_masked_lin(adata, autoencoder, loss_func, lr, epochs, batch_size):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    
    t_X = torch.from_numpy(adata.X)
    
    logits = autoencoder.encoder[0].logits

    for epoch in range(epochs):
        autoencoder.train()
        for step in range(int(adata.n_obs/batch_size)):
            batch = torch.from_numpy(adata.chunk_X(batch_size))
            optimizer.zero_grad()
            batch_loss = 0
            
            for sample in batch:
                sample = sample[None, :]
                encoded, decoded = autoencoder(sample)
                
                loss = sum(loss_func(decoded, sample, logits))/batch_size
                loss.backward()
                
                batch_loss += loss.data
            
            optimizer.step()
            if step % 100 == 0: print('Epoch: ', epoch, '| batch train loss: %.4f' % batch_loss.numpy())
        autoencoder.eval()
        _, t_decoded = autoencoder(t_X)
        
        t_loss = loss_func(t_decoded, t_X, logits)
        t_loss = [sum(t_loss)] + list(t_loss)
        t_loss = [l.data.numpy() for l in t_loss]
        
        print('Epoch: ', epoch, '-- total train loss: %.4f=%.4f+%.4f' % tuple(t_loss))