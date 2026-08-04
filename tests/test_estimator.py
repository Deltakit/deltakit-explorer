import unittest
from unittest.mock import patch, MagicMock
import math


from src.deltakit_explorer.analysis.threshold import ThresholdEstimator, get_error_bar

class TestThresholdEstimator(unittest.TestCase):

    def test_get_error_bar(self):
        """Test that the math for the standard error works correctly."""
        # Test 1: 0 LEP should return 0 error
        self.assertEqual(get_error_bar(0.0, 100), 0.0)
        
        # Test 2: Normal mathematical calculation
        expected_error = math.sqrt((0.5 * 0.5) / 100)
        self.assertTrue(math.isclose(get_error_bar(0.5, 100), expected_error))

    @patch('your_file_name.Client')
    def test_run_simulation(self, MockClient):
        """Test the worker tool with the NEW shots parameter."""
        
        mock_instance = MockClient.get_instance.return_value
        mock_decode_result = MagicMock()
        mock_decode_result.get_logical_error_probability.return_value = 0.015
        mock_instance.decode_measurements.return_value = mock_decode_result
        mock_instance.simulate_stim_circuit.return_value = (MagicMock(), MagicMock())

        estimator = ThresholdEstimator()
        
        
        result = estimator.run_simulation(p_value=0.01, distance=3, shots=1000)
        self.assertEqual(result, 0.015)

    @patch.object(ThresholdEstimator, 'run_simulation')
    def test_run_search(self, mock_run_sim):
        """Test that the manager tool correctly steps left or right."""
        # We fake the simulation results to have a wide gap so the overlap loop doesn't get stuck
        def fake_simulation(p_value, distance, shots):
            if distance == 3: return p_value * 0.5
            if distance == 5: return p_value * 1.5
        
        mock_run_sim.side_effect = fake_simulation

        
        estimator = ThresholdEstimator(min_p=0.01, max_p=0.05, precision=0.01)
        threshold = estimator.run_search()
        
        self.assertIsInstance(threshold, float)
        self.assertTrue(mock_run_sim.called)

if __name__ == '__main__':
    unittest.main()
