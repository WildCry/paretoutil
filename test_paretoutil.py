import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import yfinance as yf
import paretoutil as pu


class TestLogDifference(unittest.TestCase):
    def test_log_difference_with_series(self):
        # Create a pandas Series for testing
        data = pd.Series([1, 2, 4, 8])

        # Expected result calculated manually
        expected = pd.Series([np.nan, np.log(2), np.log(2), np.log(2)])

        # Call the function under test
        result = pu.log_difference(data)

        # Verify the result
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_log_difference_with_dataframe(self):
        # Create a pandas DataFrame for testing
        data = pd.DataFrame({'A': [1, 2, 4, 8], 'B': [1, 3, 9, 27]})

        # Expected result calculated manually
        expected = pd.DataFrame({'A': [np.nan, np.log(2), np.log(2), np.log(2)], 'B': [
                                np.nan, np.log(3), np.log(3), np.log(3)]})

        # Call the function under test
        result = pu.log_difference(data)

        # Verify the result
        pd.testing.assert_frame_equal(result, expected)

    def test_log_difference_with_ndarray(self):
        # Create a numpy array for testing
        data = np.array([[1, 2], [4, 8], [16, 32]])

        # Expected result calculated manually
        expected = np.array(
            [[np.nan, np.nan], [np.log(4), np.log(4)], [np.log(4), np.log(4)]])

        # Call the function under test
        result = pu.log_difference(data)

        # Verify the result, using np.testing.assert_array_almost_equal because of potential floating-point errors
        np.testing.assert_array_almost_equal(result, expected)


# Add more test cases as needed


if __name__ == '__main__':
    unittest.main()
