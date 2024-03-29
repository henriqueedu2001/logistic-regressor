import numpy as np

class LogisticRegressor():
    def __init__(self, dataset, weights: np.array = np.array([0, 1])) -> None:
        self.dataset = dataset
        self.weights = weights
        pass
    
    
    def predict_probabilty(self, input: float) -> float:
        """Predicts the probability of the given sample be from the positive class (1).
        In another words, P[Y = 1], given X = x, where Y is the class and x is the input.

        Args:
            input (float): input for the model

        Returns:
            float: the probability of the samble being from the positive class (1)
        """
        
        w_0, w_1 = self.weights[0], self.weights[1]
        probability = logit(input, w_0, w_1)
        
        return probability
    
    
    def predict_class(self, input: float, threshold: float = 0.5) -> float:
        """Predicts the class of the given sample. The prediction will be the positive class
        (1) if the probability of this particular example being from the positive class, acording
        to the model, is greater than the threshold (0.5 by default).

        Args:
            input (float): input for the model
            threshold (float, optional): the minimum probabilty required to consider the positive class
            as the most probable classification for the sample. Defaults to 0.5.

        Returns:
            float: the predicted class
        """
        w_0, w_1 = self.weights[0], self.weights[1]
        probability = logit(input, w_0, w_1)
        
        predicted_class = 0
        
        if probability >= threshold:
            return 1
        
        return predicted_class


def logit(input: float, w_0: float, w_1: float) -> float:
    """The logit function sigma(x) = 1/(1 + exp(-(w_0 + w_1*x))),
    for weights w_0 and w_1

    Args:
        input (float): input x of the logit function sigma(x)
        w_0 (float): the weight w_0
        w_1 (float): the weight w_1

    Returns:
        float: the output sigma(x) = 1/(1 + exp(-(w_0 + w_1*x)))
    """
    x = input
    sigma = 1/(1 + np.exp(-(w_0 + w_1*x)))
    
    return sigma


def binary_cross_entropy(y: float, y_hat: float, H_episilon: float = 1e-9) -> float:
    """Evaluates the binary cross entropy H for a single example (y, ŷ),
    where y is the true label and ŷ is the probability of the predicted label
    Args:
        y (float): the real label of the example
        y_hat (float): the label provided by the model
        H_episilon (float): the tolerance for the numeric evaluation, that prevents 0 in the logarithms

    Returns:
        _type_: _description_
    """
    
    H = -(y*np.log(y_hat + H_episilon) + (1 - y)*np.log(1 - y_hat + H_episilon))
    
    return H