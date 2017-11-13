import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, KFold

from scipy.stats.stats import pearsonr

from data_balancing import DoNothing

class RunnerWithBalancing:
    def __init__(self, models, bal_methods = None):
        """
        @param models a list of triples (name, class, param)
        @param bal_methods a list of triples (name, class, param)
        """
        self.models = models
        
        if bal_methods == None:
            self.bal_methods = [('none', DoNothing, {})]
        else:
            self.bal_methods = bal_methods
        
    def _evaluate_prediction(self, y, y_pred, classification=True):
        """
            return (accuracy, precision, recall, f1-score, false positive rate)
        """
        if classification:
            acc = accuracy_score(y, y_pred)
            pre, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', pos_label=1)

            cfs_mat = confusion_matrix(y, y_pred)
            fpr = 1.0 * cfs_mat[0,1] / np.sum(cfs_mat[0,:])
            
            return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'fpr': fpr}
        else:
            mse = np.mean((y - y_pred)**2)
            rmse = np.sqrt(mse)
            cos_sim = cosine_similarity(y.reshape(1,-1), y_pred.reshape(1,-1))[0,0]
            
            return {'mse': mse, 'rmse': rmse, 'cos_sim': cos_sim}

    def _learn_model(self, X_train, y_train, X_valid, y_valid, model, classification=True):
        """
            do fit_predict_evaluate
        """

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        train_eval = self._evaluate_prediction(y_train, y_train_pred, classification)
        valid_eval = self._evaluate_prediction(y_valid, y_valid_pred, classification)

        return {'trn': train_eval, 'val': valid_eval}, model
    
    def _run_fold(self, X_train, y_train, X_valid, y_valid, result, classification, verbose):
        if verbose:
            print 'FOLD %d' % fold_ix
            print 'Traning and validation datasets:'
            print '---- before argumentation'
            print '-------- # instances: trn(%d), val(%d)' % (X_train.shape[0], X_valid.shape[0])
            if classification:
                print '-------- # label 1: trn(%d), val(%d): \n' % ((y_train==1).sum(), (y_valid==1).sum())

        for bal_ix, (bal_name, bal_class, bal_param) in enumerate(self.bal_methods):
            bal_obj = bal_class(**bal_param)

            X_train_bal, y_train_bal = bal_obj.fit_sample(X_train, y_train)        

            if verbose:
                print '---- after argumentation by ', bal_name
                print '-------- # instances: trn(%d), val(%d)' % (X_train_bal.shape[0], X_valid.shape[0])
                if classification:
                    print '-------- # label 1: trn(%d), val(%d): ' % ((y_train_bal==1).sum(), (y_valid==1).sum())
                print '-------- Parameter setting: %s\n' % bal_obj.get_params()
                print '\n'

            for model_ix, (model_name, model_class, model_param) in enumerate(self.models):
                model_obj = model_class(**model_param)
                result[bal_ix, model_ix, fold_ix], models[bal_ix, model_ix, fold_ix] = self._learn_model(X_train_bal, y_train_bal, X_valid, y_valid, model_obj, classification)

        if valid_once and fold_ix > 0:
            result = result[:, :, [0]]
            models = models[:, :, [0]]
#            ret_train_ix = train_ix
#            ret_valid_ix = valid_ix
            break
    
        return result
    
    def run(self, X, y, n_folds, stratified=False, valid_once=False, classification=True, verbose=True):
        n_bals, n_models, n_folds = len(self.bal_methods), len(self.models), cv_obj.n_splits
        
        models = np.empty((n_bals, n_models, n_folds), dtype='object')
        result = np.empty((n_bals, n_models, n_folds), dtype='object')
        
        if n_folds > 1:
            cv_obj = (StratifiedKFold if stratified else KFold)(n_splits=n_folds, suffle=True)

#            ret_train_ix, ret_valid_ix = None, None
            for fold_ix, (train_ix, valid_ix) in enumerate(cv_obj.split(X, y)):
                X_train, y_train = X[train_ix], y[train_ix]
                X_valid, y_valid = X[valid_ix], y[valid_ix]

                result = self._run_fold(X_train, y_train, X_valid, y_valid, result, classification, verbose)
        else:
#            result = self.
            
        return result, models#, ret_train_ix, ret_valid_ix

    
        
    
    
    
    
    
    
    
    
        