import unittest
import regressor
import numpy as np

class Test(unittest.TestCase):
    def test_logit(self):
        """Tests the logit function of the regressor
        """
        w_0, w_1 = 0, 1
        x_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        y_expected_values = [
            0.11920292202211755, 0.18242552380635635, 0.2689414213699951, 
            0.3775406687981454, 0.5000000000000000, 0.6224593312018546, 
            0.7310585786300049, 0.8175744761936437, 0.8807970779778823]
    
        for i, x_value in enumerate(x_values):
            y_calculated = regressor.logit(x_value, w_0, w_1)
            y_expected = y_expected_values[i]
            self.assertEqual(y_calculated, y_expected)
    
    
    def test_cross_entropy(self):
        zero_tolerance = 1e-6
        
        # low level of entropy, as the predicted labels are close to the real labels
        self.assertTrue(regressor.binary_cross_entropy(0, 0) < zero_tolerance)
        self.assertTrue(regressor.binary_cross_entropy(1, 1) < zero_tolerance)
        
        # high level of entropy, as the predicted labels are far from the real labels
        self.assertTrue(regressor.binary_cross_entropy(0, 1) > zero_tolerance)
        self.assertTrue(regressor.binary_cross_entropy(1, 0) > zero_tolerance)
        
        # medium level of entropy, as the predicted labels are far from the real labels
        self.assertTrue(regressor.binary_cross_entropy(0, 0.5) > zero_tolerance)
        self.assertTrue(regressor.binary_cross_entropy(1, 0.5) > zero_tolerance)

    
    def test_dataset(self):
        dataset = np.array([
            [ 0.0, 0], [ 0.5, 0], [ 1.0, 0], [ 1.5, 0], 
            [ 2.0, 0], [ 2.5, 0], [ 3.0, 0], [ 3.5, 0], 
            [ 4.0, 0], [ 4.5, 0], [ 5.0, 0], [ 5.5, 0], 
            [ 6.0, 1], [ 6.5, 1], [ 7.0, 1], [ 7.5, 1], 
            [ 8.0, 1], [ 8.5, 1], [ 9.0, 1], [ 9.5, 1], 
            [10.0, 1], [10.5, 1], [11.0, 1], [11.5, 1], 
            [12.0, 1], [12.5, 1], [13.0, 1], [13.5, 1]
        ])
        
        logistic_reg = regressor.LogisticRegressor(dataset)
        self.assertTrue(np.array_equal(dataset, logistic_reg.dataset))

unittest.main()