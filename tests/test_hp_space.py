import unittest
from focus_opt.hp_space import BooleanHyperParameter, CategoricalHyperParameter, OrdinalHyperParameter, HyperParameterSpace

class TestHyperParameters(unittest.TestCase):

    def test_boolean_hyperparameter(self):
        hp = BooleanHyperParameter("test_bool", proba_true=0.7)
        sample = hp.sample()
        self.assertIn(sample, [True, False])
        neighbors = hp.sample_neighbors(sample)
        self.assertEqual(neighbors, [not sample])

    def test_categorical_hyperparameter(self):
        hp = CategoricalHyperParameter("test_cat", ["a", "b", "c"])
        sample = hp.sample()
        self.assertIn(sample, ["a", "b", "c"])
        neighbors = hp.sample_neighbors(sample)
        self.assertTrue(all(n in ["a", "b", "c"] for n in neighbors))

    def test_ordinal_hyperparameter(self):
        hp = OrdinalHyperParameter("test_ord", [1, 2, 3])
        sample = hp.sample()
        self.assertIn(sample, [1, 2, 3])
        neighbors = hp.sample_neighbors(sample)
        self.assertTrue(all(n in [1, 2, 3] for n in neighbors))

class TestHyperParameterSpace(unittest.TestCase):

    def setUp(self):
        self.hp_space = HyperParameterSpace("test_space", [
            BooleanHyperParameter("param1"),
            CategoricalHyperParameter("param2", ["a", "b", "c"])
        ])

    def test_sample_config(self):
        config = self.hp_space.sample_config()
        self.assertIn(config["param1"], [True, False])
        self.assertIn(config["param2"], ["a", "b", "c"])

    def test_sample_configs(self):
        configs = self.hp_space.sample_configs(5)
        self.assertEqual(len(configs), 5)
        for config in configs:
            self.assertIn(config["param1"], [True, False])
            self.assertIn(config["param2"], ["a", "b", "c"])

    def test_sample_all_neighbors(self):
        config = self.hp_space.sample_config()
        neighbors = self.hp_space.sample_all_neighbors(config)
        self.assertTrue(len(neighbors) > 0)

if __name__ == '__main__':
    unittest.main()
