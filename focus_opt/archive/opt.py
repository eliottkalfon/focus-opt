import math
from abc import ABC
from typing import Callable, List
from src.hp_space import HyperParameterSpace
from src.config_candidate import ConfigCandidate
import time

"""
TO DO
- max budget
- time out
- enqueue trial
- error handling with log, if eval fail, log failure
- log trials
"""

class BaseOptimizer(ABC):
    def __init__(self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        max_fidelity: int = 0,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "latest",
        score_aggregation_function: Callable = None,
    ):
        self.hp_space = hp_space
        self.evaluation_function = evaluation_function
        self.max_fidelity = max_fidelity
        self.sh_eta = sh_eta
        self.maximize = maximize
        self.best_candidate = None
        self.score_aggregation = score_aggregation
        self.score_aggregation_function = score_aggregation_function

    def config_to_candidate(self, config: dict):
        """Instantiates a ConfigCandidate from a config dict"""
        return ConfigCandidate(
            config=config,
            evaluation_function=self.evaluation_function,
            score_aggregation=self.score_aggregation,
            score_aggregation_function=self.score_aggregation_function
        )

    def configs_to_candidates(self, configs: List[dict]):
        """Instantitates a list of ConfigCandidates from a list of dicts"""
        return [self.config_to_candidate(config) for config in configs]

    def compare_candidates(
            self,
            candidate_1: ConfigCandidate,
            candidate_2: ConfigCandidate
        ) -> bool:
        """
        Returns true is the first candidate is 'better' than the second candidate
        based on the optimisation objective
        """
        if self.maximize:
            return candidate_1.evaluation_score > candidate_2.evaluation_score
        else:
            return candidate_1.evaluation_score < candidate_2.evaluation_score


    def successive_halving(self, candidates: List[configs_to_candidates], total_cost: int):
        for _ in range(self.max_fidelity+1):
            for candidate in candidates:
                candidate.evaluate()
                total_cost+=1
            sh_cutoff = math.ceil(len(candidates)*self.sh_eta)
            candidates = sorted(
                candidates,
                key = lambda x: x.evaluation_score,
                reverse = self.maximize
            )[:sh_cutoff]
        return candidates, total_cost

    def update_best(self, candidates: List[ConfigCandidate]):
        new_candidate = sorted(candidates, key=lambda x: x.evaluation_score, reverse = self.maximize)[0]
        if (
            self.best_candidate is None or
            self.compare_candidates(new_candidate, self.best_candidate)
        ):
            self.best_candidate = new_candidate



    def optimize(self):
        raise NotImplementedError

class RandomSearchOptimizer(BaseOptimizer):
    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        budget: int = 100,
        max_fidelity: int = 0,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "latest",
        score_aggregation_function: Callable = None,
    ):
        super().__init__(
            hp_space,
            evaluation_function,
            max_fidelity,
            sh_eta,
            maximize,
            score_aggregation,
            score_aggregation_function
        )
        self.budget = budget

    def optimize(self, population_size=10, max_time=None):
        start_time = time.time()
        total_cost = 0
        while total_cost < self.budget:
            if max_time and (time.time() - start_time) > max_time:
                break
            candidates = self.configs_to_candidates(
                self.hp_space.sample_configs(n_configs=population_size)
            )
            candidates, total_cost = self.successive_halving(candidates, total_cost)
            self.update_best(candidates)
        return self.best_candidate
