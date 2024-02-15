import numpy as np

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