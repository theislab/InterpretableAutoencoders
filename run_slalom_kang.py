import time
import slalom as sl
import numpy as np
from anndata import read_h5ad

print('reading')
ad = read_h5ad('kang_count.h5ad')

print('initializing')

t0 = time.time()

FA = sl.initFA(ad.X, ad.uns['terms'], ad.varm['I'], list(ad.var_names), noise='gauss', nHidden=3, nHiddenSparse=0, 
               minGenes=12, pruneGenes=True, do_preTrain=False)

print('training')

FA.train()

t1 = time.time()

print(t1 - t0)

sl.saveFA(FA, 'FA_kang_3_hidden.hdf5', saveF=True)
