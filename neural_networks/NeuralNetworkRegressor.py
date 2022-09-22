import numpy as np

from itcs4156.assignments.neural_networks.NeuralNetwork import NeuralNetwork



class NeuralNetworkRegressor(NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._param_names = list(kwargs.keys())
    
    def get_params(self, deep=True):
        """ Gets all class variables
        
            This is a a method for compatibility with Sklearn's GridSearchCV 
        """
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        """ Sets all class variables
        
            This is a a method for compatibility with Sklearn's GridSearchCV 
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions using parameters learned during training.
        
            Args:
                X: Features/data to make predictions with 

            TODO:
                Finish this method by adding code to make a prediction. 
                Store the predicted labels into `y_hat`.
        """
        # TODO (REQUIRED) Add code below

        # TODO (REQUIRED) Store predictions below by replacing np.ones()
        y_hat = self.forward(X)
        # Makes sure predictions are given as a 2D array
        return y_hat.reshape(-1, 1)
