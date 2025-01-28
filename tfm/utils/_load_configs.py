import pandas as pd
import numpy as np
import os
import yaml
import warnings
from tfm.utils._constants import *

config_path = "/home/james/projects/tsfc/code/code_11092024/tfm/config/"

def load_yaml_config(sn):
    ## define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])
    
    ## register the tag handler
    yaml.add_constructor('!join', join)
    with open(f'{config_path}/{sn}.yaml', 'r') as file:
      try:
          return yaml.load(file, Loader=yaml.FullLoader)
      except yaml.YAMLError as exc:
          print(exc)
    return None

start_date_oos = pd.to_datetime('2005-01-01')
config = os.environ.get('CONFIG')

if config in ('char_anom', 'scs', 'wrds'):
    mp = load_yaml_config(config)
    idx_ver, intercept_constr, normalize_ret, input_type, spec, max_lag, dir_input = mp.values()
    fit_rx = False if input_type == 'ret' else True
else:
    warnings.warn('The provided dataset should be one of: char_anom, scs, or wrds.')
    import sys
    sys.exit(0)


# def get_normalized_factors(dict_fit, reorder=True):
#     F,W,B,S=[dict_fit[key] for key in ['F','W','B','S']]
    
#     F_norm=np.linalg.norm(F,axis=0)
#     W_norm=np.linalg.norm(W,axis=0)
#     B_norm=np.linalg.norm(B,axis=0)
    
#     S_out=S*F_norm*W_norm*B_norm
#     F_out=F/F_norm[np.newaxis,:]
#     W_out=W/W_norm[np.newaxis,:]
#     B_out=B/B_norm[np.newaxis,:]
    
#     if reorder:
#         order=np.argsort(S_out)[::-1]
#         F_out,W_out,B_out=F_out[:,order],W_out[:,order],B_out[:,order]
#         S_out=S_out[order]
    
#     dict_norm={'F':F_out,'W':W_out, 'B':B_out, 'S':S_out}
#     return dict_norm


# def get_corr_mat(Q):
#     cov=np.cov(Q.T)
#     std=(np.diag(cov)**.5)[:,np.newaxis]
#     corr=cov/(std@std.T)
#     return corr

# def get_reconstruction_err_no_refit(dict_fit,X):
#     F,W,B,S=[dict_fit[key] for key in ['F','W','B','S']]
#     K_max=S.shape[0]
    
#     err_rc=[]
#     for K in range(1,K_max+1):
#         if K==1:
#             X_rc=np.tensordot(np.tensordot(F[:,0],W[:,0],axes=0),B[:,0],axes=0)*S[0]
#         else:
#             X_rc+=np.tensordot(np.tensordot(F[:,K-1],W[:,K-1],axes=0),B[:,K-1],axes=0)*S[K-1]
#         err=((X-X_rc)**2).sum()
#         err_rc.append(err)  
#     return err_rc

# def center_3d_tensor(Y, center_dim_1=True, center_dim_2=True, center_dim_3=True):
#     assert len(Y.shape)==3
#     dim_1,dim_2,dim_3=Y.shape
    
#     Y_center=Y
#     if center_dim_1:
#         Y_temp=Y_center.reshape(dim_1,-1)
#         Y_center_flatten=Y_temp-Y_temp.mean(axis=0)
#         Y_center=Y_center_flatten.reshape(dim_1,dim_2,dim_3)
#     if center_dim_2:
#         Y_temp=np.moveaxis(Y_center,[0,1,2],[1,0,2]).reshape(dim_2,-1)
#         Y_center_flatten=Y_temp-Y_temp.mean(axis=0)
#         Y_center=np.moveaxis(Y_center_flatten.reshape((dim_2, dim_1, dim_3)),[0,1,2],[1,0,2])
#     if center_dim_3:
#         Y_temp=np.moveaxis(Y_center,[0,1,2],[1,2,0]).reshape(dim_3,-1)
#         Y_center_flatten=Y_temp-Y_temp.mean(axis=0)
#         Y_center=np.moveaxis(Y_center_flatten.reshape((dim_3, dim_1, dim_2)),[0,1,2],[2,0,1])
#     Y_center_flatten=Y_center.reshape(dim_1,-1)
#     return Y_center, Y_center_flatten
        

# def reconstruct(F,W,B,S):
#     K=S.shape[0]
#     X_rc=np.tensordot(np.tensordot(F[:,0],W[:,0],axes=0),B[:,0],axes=0)*S[0]
#     for i in range(1,K):
#         X_rc+=np.tensordot(np.tensordot(F[:,i],W[:,i],axes=0),B[:,i],axes=0)*S[i]
#     return X_rc

# def get_F_reg(X, dict_fit, normalize=True):
#     W,B,S=[dict_fit[key] for key in ['W','B','S']]
    
#     # obtain F by regressing X on S, W, B 
#     K=len(S)
#     Z=np.full((K,B.shape[0]*W.shape[0]),np.nan)
#     for i in range(K):
#         Z[i]=np.kron(W[:,i], B[:,i])*S[i]
#     mat_weight_flatten=Z.T@np.linalg.inv(Z@Z.T) # dim: (num_char*max_lag, rank)
#     F_reg=X.reshape(T,-1)@mat_weight_flatten
    
#     # normalize each col of F_reg to have norm 1. move scaling to S
#     if normalize:
#         F_reg_norm=np.linalg.norm(F_reg,axis=0)
#         F_reg/=F_reg_norm[np.newaxis,:]
#         S*=F_reg_norm
        
#     dict_reg={'F':F_reg,'W':W, 'B':B, 'S':S}
#     return dict_reg



# def get_B_reg(X, dict_fit, normalize=True):
#     W,F,S=[dict_fit[key] for key in ['W','F','S']]
    
#     # get B from F and W
#     K=len(S)
#     Z=np.full((K,F.shape[0]*W.shape[0]),np.nan)
#     for i in range(K):
#         Z[i]=np.kron(F[:,i],W[:,i])*S[i]
#     mat_weight_flatten=Z.T@np.linalg.inv(Z@Z.T) # dim: (num_char*max_lag, rank)
#     B_reg=np.moveaxis(X,[0,1,2],[1,2,0]).reshape(X.shape[2],-1)@mat_weight_flatten
    
#     if normalize:
#         B_reg_norm=np.linalg.norm(B_reg,axis=0)
#         B_reg/=B_reg_norm[np.newaxis,:]
#         S*=B_reg_norm
        
#     dict_reg={'F':F, 'W':W, 'B':B_reg, 'S':S}
#     return dict_reg

# def get_W_reg(X, dict_fit, normalize=True):
#     B,F,S=[dict_fit[key] for key in ['B','F','S']]
    
#     # get W from F and B
#     K=len(S)
#     Z=np.full((K,F.shape[0]*B.shape[0]),np.nan)
#     for i in range(K):
#         Z[i]=np.kron(F[:,i],B[:,i])*S[i]
#     mat_weight_flatten=Z.T@np.linalg.inv(Z@Z.T) # dim: (num_char*T, rank)
#     W_reg=np.moveaxis(X,[0,1,2],[1,0,2]).reshape(X.shape[1],-1)@mat_weight_flatten
    
#     if normalize:
#         W_reg_norm=np.linalg.norm(W_reg,axis=0)
#         W_reg/=W_reg_norm[np.newaxis,:]
#         S*=W_reg_norm
        
#     dict_reg={'F':F, 'W':W_reg, 'B':B, 'S':S}
#     return dict_reg

