import unittest
import random
from unittest.mock import MagicMock
from focus_opt.optimizers import RandomSearchOptimizer, HillClimbingOptimizer, GeneticAlgorithmOptimizer
from focus_opt.hp_space import HyperParameterSpace, BooleanHyperParameter, CategoricalHyperParameter
from focus_opt.config_candidate import ConfigCandidate

class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.hp_space = HyperParameterSpace("test_space", [
            BooleanHyperParameter("param1"),
            CategoricalHyperParameter("param2", ["a", "b", "c"])
        ])
        self.evaluation_function = MagicMock(return_value=1.0)

    def test_random_search_optimizer(self):
        optimizer = RandomSearchOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = optimizer.optimize(budget=10)
        self.assertIsInstance(best_candidate, ConfigCandidate)
        # evaluation function is called once more than budget to check for signature
        self.assertLessEqual(self.evaluation_function.call_count, 11)

    def test_hill_climbing_optimizer(self):
        optimizer = HillClimbingOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = optimizer.optimize(budget=10)
        self.assertIsInstance(best_candidate, ConfigCandidate)
        self.assertLessEqual(self.evaluation_function.call_count, 11)

    def test_genetic_algorithm_optimizer(self):
        optimizer = GeneticAlgorithmOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = optimizer.optimize(budget=10)
        self.assertIsInstance(best_candidate, ConfigCandidate)
        self.assertLessEqual(self.evaluation_function.call_count, 11)

    def test_optimizers_reach_optimum(self):
        """Test optimizers to reach an easy minimum"""
        def simple_evaluation_function(config):
            if config['param1'] == True and config['param2'] == 'b':
                return 0.0
            return 1.0

        self.evaluation_function.side_effect = simple_evaluation_function

        # Test Random Search Optimizer
        random_optimizer = RandomSearchOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = random_optimizer.optimize(budget=100)
        self.assertEqual(best_candidate.config['param1'], True)
        self.assertEqual(best_candidate.config['param2'], 'b')

        # Test Hill Climbing Optimizer
        hill_climbing_optimizer = HillClimbingOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = hill_climbing_optimizer.optimize(budget=100)
        self.assertEqual(best_candidate.config['param1'], True)
        self.assertEqual(best_candidate.config['param2'], 'b')

        # Test Genetic Algorithm Optimizer
        genetic_algorithm_optimizer = GeneticAlgorithmOptimizer(self.hp_space, self.evaluation_function)
        best_candidate = genetic_algorithm_optimizer.optimize(budget=100)
        self.assertEqual(best_candidate.config['param1'], True)
        self.assertEqual(best_candidate.config['param2'], 'b')

    def test_complex_multi_fidelity_optimization(self):
        """Test optimizers with a more complex multi-fidelity evaluation function"""
        def complex_multi_fidelity_evaluation_function(config, fidelity):
            base_score = random.uniform(-0.5, 0.5) + fidelity
            if config['param1'] == True:
                base_score += 0.5
            if config['param2'] == 'b':
                base_score += 0.5
            return 1.0 / (1.0 + base_score)  # Inverse to make it a minimization problem

        self.evaluation_function.side_effect = complex_multi_fidelity_evaluation_function

        # Test Random Search Optimizer
        random_optimizer = RandomSearchOptimizer(self.hp_space, self.evaluation_function, max_fidelity=5)
        best_candidate = random_optimizer.optimize(budget=100)
        self.assertTrue(best_candidate.config['param1'])
        self.assertEqual(best_candidate.config['param2'], 'b')

        # Test Hill Climbing Optimizer
        hill_climbing_optimizer = HillClimbingOptimizer(self.hp_space, self.evaluation_function, max_fidelity=5)
        best_candidate = hill_climbing_optimizer.optimize(budget=100)
        self.assertTrue(best_candidate.config['param1'])
        self.assertEqual(best_candidate.config['param2'], 'b')

        # Test Genetic Algorithm Optimizer
        genetic_algorithm_optimizer = GeneticAlgorithmOptimizer(self.hp_space, self.evaluation_function, max_fidelity=5)
        best_candidate = genetic_algorithm_optimizer.optimize(budget=100)
        self.assertTrue(best_candidate.config['param1'])
        self.assertEqual(best_candidate.config['param2'], 'b')

if __name__ == '__main__':
    unittest.main()
