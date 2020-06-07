import time
import slalom as sl
import numpy as np
from anndata import read_h5ad

print('reading')
ad = read_h5ad('mouse_retina_sbs.h5ad')

print('initializing')

t0 = time.time()

FA = sl.initFA(ad.X, ad.uns['terms'], ad.varm['I'], list(ad.var_names), noise='gauss', nHidden=0, nHiddenSparse=0, 
               minGenes=12, pruneGenes=True, do_preTrain=False)

print('training')

FA.train()

t1 = time.time()

print(t1 - t0)

sl.saveFA(FA, 'FA_macosko.hdf5', saveF=True)