import numpy as np
import pandas as pd


def read_data(machine_id, frac, set_id, data_type, data_dir):
    """
    machine_id: 1 ~ 4
    set_id: 0 ~ 4
    frac:  1, 5, 10, 20, 50, 80, 90, 100
    
    """
    
    data_file = data_dir + '/FD{:03d}/frac{}/{}_FD{:03d}_proc2_set{}.csv'.format(machine_id, frac, data_type, machine_id, set_id)
    df = pd.read_csv(data_file)

    X = df[[str(i) for i in range(17)]].values
    y = df['RUL'].values.astype('float')
    uid = df['RunID'].values
        
    return X, y, uid

def recast_matrix(X, y, lag):
    n_rows = X.shape[0]
    n_instances = n_rows - lag + 1
    
    X_out, y_out = X[:n_instances, :], y[lag-1:]
    for w in range(1, lag):
        Xw = X[w:w+n_instances, :]
        X_out = np.hstack((X_out, Xw))
    
    return X_out, y_out

def recast_data(X, y, lag, win_size, uid, data_type):    
    X_out = np.zeros((0, win_size, lag*X.shape[1]))
    y_out = np.zeros(0)
    for i in np.unique(uid):
        ind = uid == i
        Xi, yi = recast_matrix(X[ind, :], y[ind], lag)
        Xii, yii = recast_matrix(Xi, yi, win_size)
        
        Xii = Xii.reshape(-1, win_size, lag*X.shape[1])
        
        if data_type == 'train':
            X_out = np.concatenate([X_out, Xii])
            y_out = np.append(y_out, yii)
        elif data_type == 'test':
            X_out = np.concatenate([X_out, Xii[[-1], :, :]])
            y_out = np.append(y_out, yii[-1])
        else:
            raise Exception('In recast_data, data_type should be one of {train, test}')
        
    return X_out, y_out