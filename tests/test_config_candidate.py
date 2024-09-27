import unittest
from unittest.mock import MagicMock
from focus_opt.config_candidate import ConfigCandidate
from focus_opt.helpers import SessionContext

class TestConfigCandidate(unittest.TestCase):

    def setUp(self):
        self.config = {"param1": True, "param2": "a"}
        self.evaluation_function = MagicMock(return_value=1.0)
        self.candidate = ConfigCandidate(
            config=self.config,
            evaluation_function=self.evaluation_function,
            max_fidelity=3
        )

    def test_evaluate(self):
        self.session_context = SessionContext(budget=100)
        score = self.candidate.evaluate(self.session_context)
        self.assertEqual(score, 1.0)
        self.assertEqual(self.session_context.total_cost, 1)
        self.assertEqual(self.candidate.fidelity_level, 1)
        self.assertEqual(self.candidate.evaluation_score, 1.0)
        self.assertFalse(self.candidate.is_fully_evaluated)

    def test_full_evaluation(self):
        self.session_context = SessionContext(budget=100)
        score = self.candidate.full_evaluation(self.session_context)
        self.assertEqual(score, 1.0)
        self.assertEqual(self.session_context.total_cost, 3)
        self.assertEqual(self.candidate.fidelity_level, 3)
        self.assertEqual(self.candidate.evaluation_score, 1.0)
        self.assertTrue(self.candidate.is_fully_evaluated)

    def test_aggregate_evaluations(self):
        self.candidate.evaluations = {0: 1.0, 1: 2.0, 2: 3.0}
        score = self.candidate.aggregate_evaluations()
        self.assertEqual(score, 3.0)

if __name__ == '__main__':
    unittest.main()
