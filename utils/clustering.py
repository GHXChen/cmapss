import numpy as np

class HBOS:
    def __init__(self):
        pass
    
    def fit(self, X):
        N, D = X.shape
        
        hist, bin_edges = [None]*D, [None]*D
        n_bins = np.sqrt(N, )
        for i in range(D):
            hist[i], bin_edges[i] = np.histogram(X[:,i], n_bins)
            hist[i] /= np.max(hist[i])
            
        self.hist = np.array(hist)
        self.bin_edges = np.array(bin_edges)
    
    def predict(self, X):
        hist = self.hist
        bin_edges = self.bin_edges
        
        D = hist.shape[0]
        
        bins = np.apply_along_axis(lambda x: np.digitize(bin_edges, x)-1, 0, X)
        heights = np.apply_along_axis(lambda b: hist[range(D),b], 1, bins)
        
        return np.sum(np.log(1/heights), axis=1)
        
        
            