import numpy as np
import pandas as pd

from imblearn import over_sampling, under_sampling

class DoNothing:
    """
        This class does nothing
    """
    
    def __init__(self, **kwarg):
        return 
    def fit_sample(self, X, y):
        return X, y
    def get_params(self):
        return {}

class UnderAndOverSampler:
    def __init__(self, un_samp, ov_samp):                
        self.un_samp = un_samp.__class__(**un_samp.get_params())
        self.ov_samp = ov_samp.__class__(**ov_samp.get_params())
        
    def fit_sample(self, X, y):
        X_un, y_un = self.un_samp.fit_sample(X, y)
        return self.ov_samp.fit_sample(X_un, y_un)
    
    def get_params(self):
        return {'un_samp':self.un_samp.get_params(), 'ov_samp':self.ov_samp.get_params()}
    
class GenFromFile:
    """
        This class reads a file containing generated data by AES
    """
    
    def __init__(self, gen_method, aug_label, low_dim, data_root='data'):
        """
            gen_method:
                - 1: 
                - 2: 
            low_dim: {4,8,16,32, 64}
            label: 0 or 1
        """
        if aug_label not in [-1, 1]:
            raise Exception('incorrect label')        
        if gen_method not in [1, 2]:
            raise Exception('incorrect gen_method')
        if low_dim not in [4, 8, 16, 32, 64]:
            raise Exception('not supported low_dim')
            
        self.aug_label = aug_label
        self.gen_method = gen_method
        self.low_dim = low_dim
        self.data_root = data_root
        self.file_path = None
    
    def fit_sample(self, X, y):
        if self.aug_label == -1:
            label = 0
        else:
            label = 1
        
        self.file_path = '%s/gen_method%1d_class%1d_inp446_hid256_z%d.csv' % (self.data_root, self.gen_method, label, self.low_dim)
        df = pd.read_csv(self.file_path,).drop(['timestamp'], axis=1)
        add_X = df.drop(['class'], axis=1).as_matrix()
        add_y = np.zeros(add_X.shape[0]) + self.aug_label
        
        aug_X = np.vstack([X, add_X])
        aug_y = np.append(y, add_y)
        
        return aug_X, aug_y
    
    def get_params(self):
        return {'gen_method': self.gen_method, 'low_dim': self.low_dim, 'aug_label': self.aug_label, 'data_root': self.data_root, 'file_path': self.file_path}

    