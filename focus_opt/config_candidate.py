from typing import Callable
import numpy as np
from functools import lru_cache
import logging
from focus_opt.helpers import SessionContext

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=None)
def cached_evaluation_function(config_tuple, fidelity_level, evaluation_function, accepts_fidelity):
    """Global cached evaluation function to share cache across instances."""
    config_dict = dict(config_tuple)
    if accepts_fidelity:
        return evaluation_function(config_dict, fidelity_level)
    else:
        return evaluation_function(config_dict)

class ConfigCandidate:
    def __init__(
            self,
            config: dict,
            evaluation_function: Callable,
            score_aggregation: str = "latest",
            score_aggregation_function: Callable = None,
            accepts_fidelity: bool = True,
            max_fidelity: int = 0,
        ):
        self.config = config
        self.fidelity_level = None
        self.evaluations = {}
        self.evaluation_score = None
        self.evaluation_function = evaluation_function
        self.score_aggregation = score_aggregation
        self.score_aggregation_function = score_aggregation_function
        self.accepts_fidelity = accepts_fidelity
        self.max_fidelity = max_fidelity
        self.cached_evaluation_function = lru_cache(maxsize=None)(self._evaluation_wrapper)
        self._is_fully_evaluated = False

    def _evaluation_wrapper(self, config_tuple, fidelity_level):
        """Wrapper function to allow caching with lru_cache."""
        # Convert tuple back to dictionary for evaluation
        config_dict = dict(config_tuple)
        if self.accepts_fidelity:
            return self.evaluation_function(config_dict, fidelity_level)
        else:
            return self.evaluation_function(config_dict)

    def evaluate(self, session_context: SessionContext):
        """Evaluates the candidate solution at the next fidelity level."""
        session_context.budget_error_checks()

        if self.fidelity_level is None:
            self.fidelity_level = 1
        else:
            new_fidelity = self.fidelity_level + 1
            if new_fidelity > self.max_fidelity:
                raise ValueError(f"{self.config} cannot be evaluated at a fidelity beyond {self.max_fidelity}")
            else:
                self.fidelity_level += 1

        config_tuple = tuple(sorted(self.config.items()))
        evaluation_score = cached_evaluation_function(
            config_tuple,
            self.fidelity_level,
            self.evaluation_function,
            self.accepts_fidelity
        )
        self.evaluations[self.fidelity_level] = evaluation_score
        self.evaluation_score = self.aggregate_evaluations()
        session_context.increment_total_cost()

        # Log the evaluation details
        logging.info(f"Evaluating at fidelity level {self.fidelity_level}: {self.config}")
        logging.info(f"Score: {evaluation_score}")

        # Check if fully evaluated
        if self.fidelity_level == self.max_fidelity:
            self._is_fully_evaluated = True
            session_context.log_performance(self.evaluation_score)

        return self.evaluation_score

    def full_evaluation(self, session_context: SessionContext):
        """Evaluates a solution with full fidelity"""
        for _ in range(self.max_fidelity):
            self.evaluate(session_context)

        return self.evaluation_score

    def aggregate_evaluations(self):
        if self.score_aggregation_function is not None:
            return self.score_aggregation_function(self.evaluations)
        elif self.score_aggregation == "average":
            return np.mean([score for score in self.evaluations.values()])
        elif self.score_aggregation == "latest":
            return self.evaluations[max(self.evaluations.keys())]

    @property
    def is_fully_evaluated(self):
        return self._is_fully_evaluated

    def __str__(self):
        return (f"ConfigCandidate(config={self.config}, "
                f"fidelity_level={self.fidelity_level}, "
                f"evaluation_score={self.evaluation_score}, "
                f"is_fully_evaluated={self.is_fully_evaluated})")
