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

if config in ('char_anom', 'scs', 'wrds', 'ff', 'toy'):
    mp = load_yaml_config(config)
    idx_ver, intercept_constr, normalize_ret, input_type, spec, max_lag, dir_input = mp.values()
    fit_rx = False if input_type == 'ret' else True
else:
    warnings.warn('The provided dataset should be one of: char_anom, scs, wrds, or ff.')
    import sys
    sys.exit(0)