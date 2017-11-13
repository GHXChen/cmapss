import keras

class KerasModelWrapper:
    def __init__(self, model_generator, input_dim, fit_param):
        self.model = model_generator(input_dim)
        self.fit_param = fit_param
    def fit(self, X, y):
        self.model.fit(X, y, **self.fit_param)
    def predict(self, X):
        return self.model.predict(X)