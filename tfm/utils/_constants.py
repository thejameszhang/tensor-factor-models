from typing import Union
import os
from functools import partial
import jax
import jax.numpy as jnp
import numpy.typing as npt

# constants 
main_compute_device = 'gpu'

def cpu(): 
    return main_compute_device == 'cpu'

char_const = 'const'  # name of the const characteristics
n_ind_default = 12
if os.environ.get('jax_enable_x64'):
    dtype = jnp.float64
    dtype_compact = jnp.float64
else:
    dtype = jnp.float32
    dtype_compact = jnp.float32

# parafac settings
random_seed = 100
n_iter_max = 100

# Data types
Tensor = Union[npt.NDArray[dtype_compact], jnp.ndarray]
STensor = Tensor #Annotated[Tensor, Literal['Scaling Vector (K)']]
FTensor = Tensor #Annotated[Tensor, Literal['Time Series Factor Matrix (TxK)']]
BTensor = Tensor #Annotated[Tensor, Literal['Cross Sectional Loadings Matrix (NxK)']]
WTensor = Tensor #Annotated[Tensor, Literal['Lags Loadings Matrix (LxK)']]