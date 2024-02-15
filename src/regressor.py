import numpy as np

class LogisticRegressor():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        pass


def logit(input: float, w_0: float, w_1: float):
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


def binary_cross_entropy(y: float, y_hat: float, H_episilon: float = 1e-9):
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