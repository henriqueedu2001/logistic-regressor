import unittest
import regressor

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

unittest.main()