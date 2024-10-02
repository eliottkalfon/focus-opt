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
    """Represents a candidate configuration for evaluation in an optimization process.

    This class is designed to handle the evaluation of a configuration at different fidelity levels,
    caching the results to optimize performance. It supports aggregation of evaluation scores using
    different strategies.

    :param config: The configuration dictionary to be evaluated.
    :type config: dict
    :param evaluation_function: A callable function that evaluates the configuration.
    :type evaluation_function: Callable
    :param score_aggregation: The method used to aggregate scores from multiple evaluations. Options are "latest" or "average", defaults to "latest".
    :type score_aggregation: str, optional
    :param score_aggregation_function: A custom function to aggregate scores, defaults to None.
    :type score_aggregation_function: Callable, optional
    :param accepts_fidelity: Indicates if the evaluation function accepts a fidelity level, defaults to True.
    :type accepts_fidelity: bool, optional
    :param max_fidelity: The maximum fidelity level for evaluation, defaults to 0.
    :type max_fidelity: int, optional
    """

    def __init__(
            self,
            config: dict,
            evaluation_function: Callable,
            score_aggregation: str = "latest",
            score_aggregation_function: Callable = None,
            accepts_fidelity: bool = True,
            max_fidelity: int = 0,
        ):
        """Constructor method
        """
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
        """Wrapper function to allow caching with lru_cache.

        :param config_tuple: The configuration as a tuple for caching purposes.
        :type config_tuple: tuple
        :param fidelity_level: The fidelity level at which to evaluate the configuration.
        :type fidelity_level: int
        :return: The evaluation score.
        :rtype: float
        """
        config_dict = dict(config_tuple)
        if self.accepts_fidelity:
            return self.evaluation_function(config_dict, fidelity_level)
        else:
            return self.evaluation_function(config_dict)

    def evaluate(self, session_context: SessionContext):
        """Evaluates the candidate solution at the next fidelity level.

        :param session_context: The session context for managing evaluation budget and logging.
        :type session_context: SessionContext
        :return: The aggregated evaluation score.
        :rtype: float
        :raises ValueError: If the fidelity level exceeds the maximum allowed fidelity.
        """
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

        logging.info(f"Evaluating at fidelity level {self.fidelity_level}: {self.config}")
        logging.info(f"Score: {evaluation_score}")

        if self.fidelity_level == self.max_fidelity:
            self._is_fully_evaluated = True
            session_context.log_performance(self.evaluation_score)

        return self.evaluation_score

    def full_evaluation(self, session_context: SessionContext):
        """Evaluates a solution with full fidelity.

        :param session_context: The session context for managing evaluation budget and logging.
        :type session_context: SessionContext
        :return: The aggregated evaluation score after full evaluation.
        :rtype: float
        """
        for _ in range(self.max_fidelity):
            self.evaluate(session_context)

        return self.evaluation_score

    def aggregate_evaluations(self):
        """Aggregates the evaluation scores based on the specified aggregation method.

        :return: The aggregated evaluation score.
        :rtype: float
        """
        if self.score_aggregation_function is not None:
            return self.score_aggregation_function(self.evaluations)
        elif self.score_aggregation == "average":
            return np.mean([score for score in self.evaluations.values()])
        elif self.score_aggregation == "latest":
            return self.evaluations[max(self.evaluations.keys())]

    @property
    def is_fully_evaluated(self):
        """Checks if the configuration has been fully evaluated.

        :return: `True` if fully evaluated, `False` otherwise.
        :rtype: bool
        """
        return self._is_fully_evaluated

    def __str__(self):
        """String representation of the ConfigCandidate object.

        :return: A string describing the current state of the ConfigCandidate.
        :rtype: str
        """
        return (f"ConfigCandidate(config={self.config}, "
                f"fidelity_level={self.fidelity_level}, "
                f"evaluation_score={self.evaluation_score}, "
                f"is_fully_evaluated={self.is_fully_evaluated})")
