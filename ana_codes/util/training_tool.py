import numpy as np
import pandas as pd

import keras

from scipy.stats import pearsonr
from scipy.stats import mode
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


def evaluate(y_true, y_pred, classification):
    if classification:
        acc = accuracy_score(y_true, y_pred)
        pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)

        cfs_mat = confusion_matrix(y_true, y_pred)
        fpr = 1.0 * cfs_mat[0,1] / np.sum(cfs_mat[0,:])

        return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'fpr': fpr}
    else:
        def asym_loss(y_true, y_pred):
            d_all = y_pred.reshape(-1) - y_true.reshape(-1)

            coef = np.zeros_like(d_all)
            coef[d_all < 0] = -1.0/10
            coef[d_all >= 0] = 1.0/13

            return np.sum(np.exp(d_all * coef) - 1)

        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        cos_sim = cosine_similarity(y_true.reshape(1,-1), y_pred.reshape(1,-1))[0,0]
        pcc = pearsonr(y_true, y_pred)[0]
        asym = asym_loss(y_true, y_pred)

        return {'mse': mse, 'rmse': rmse, 'cos_sim': cos_sim, 'pcc': pcc, 'asym': asym}

def eval_as_array(evals, metric_order):
    eval_array = []
    for eval_map in evals:
        eval_array.append([eval_map[metric] for metric in metric_order])
    return np.array(eval_array)

def eval_as_dataframe(evals, model_names, metric_order):
    if isinstance(evals, np.ndarray):
        eval_arr = evals
    else:
        eval_arr = eval_as_array(evals, metric_order)

    eval_df = pd.DataFrame(eval_arr, index=model_names, columns=metric_order)
    return eval_df


class EnsembleModel:
    """
    Model wrapper for homogenious ensembles.
    The ensemble is accomplished from randomness.
    In other words, model.fit should result in different training.
    """

    def __init__(self, model, model_copier, n_ensembles, voting):
        self.clones = [None] * n_ensembles
        for i in range(n_ensembles):
            self.clones[i] = model_copier(model)

        self.voting = voting

    def fit(self, X, y, **param):
        for cln in self.clones:
            cln.fit(X, y, **param)

    def predict(self, X):
        if self.voting:
            if isinstance(self.clones[0], keras.models.Sequential):
                y = np.array([cln.predict_classes(X).reshape(-1) for cln in self.clones]).T
            else:
                y = np.array([cln.predict(X).reshape(-1) for cln in self.clones]).T
            return mode(y, axis=1).mode
        else:
            y = np.array([cln.predict(X).reshape(-1) for cln in self.clones]).T
            return np.mean(y, axis=1, keepdims=True)

class Runner:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self, submission, n_ensembles, classification, return_train_pred=False, return_test_pred=False):
        train_pred, test_pred = [], []
        eval_trn, eval_tst = [], []
        for i, (model, model_copier, param) in enumerate(submission):
            esm_model = EnsembleModel(model, model_copier, n_ensembles, classification)
            esm_model.fit(self.X_train, self.y_train, **param)

            y_train_pred = esm_model.predict(self.X_train).reshape(-1)
            y_test_pred = esm_model.predict(self.X_test).reshape(-1)

            ev_trn = evaluate(self.y_train, y_train_pred, classification=classification)
            ev_tst = evaluate(self.y_test, y_test_pred, classification=classification)

            eval_trn.append(ev_trn)
            eval_tst.append(ev_tst)

            train_pred.append(y_train_pred)
            test_pred.append(y_test_pred)

        return_tuple = [np.array(eval_trn), np.array(eval_tst)]
        if return_train_pred:
            return_tuple.append(np.array(train_pred).T)
        if return_test_pred:
            return_tuple.append(np.array(test_pred).T)

        return tuple(return_tuple)
