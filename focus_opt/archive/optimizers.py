import numpy as np
import math
import time
from abc import ABC
from typing import List, Tuple, Union, Callable, Dict

from src.hp_space import HyperParameterSpace

class BaseOptimizer(ABC):
    def __init__(
        self,
        space: HyperParameterSpace,
        eval_fold: Callable,
        folds: List[Tuple] = [(0,)],
        budgets: Union[None, List[int]] = None,
        eta: float = 0,
        max_budget: float = 1000,
        run_id: int = 0,
        run_name: str = "",
        memory_dict: dict = {},
        eval_kwargs: dict = {},
    ):
        self.space = space
        self.eval_fold = eval_fold
        self.folds = folds
        self.budgets = budgets if budgets is not None else [fold + 1 for fold in range(len(folds))]
        self.eta = eta
        self.max_budget = max_budget
        self.run_id = run_id
        self.run_name = run_name
        self.memory_dict = memory_dict
        self.eval_kwargs = eval_kwargs
        self.start_time = None
        self.total_cost = 0
        self.iteration_number = 0
        self.max_eval_budget = self.budgets[-1]
        self.log = []
        self.best_config = None
        self.best_eval = None

    def cached_eval_fold(self, config: Dict, fold_number: int) -> float:
        config_key = str(config)
        config_memory = self.memory_dict.get(config_key, {})
        fold_eval = config_memory.get(fold_number)

        if fold_eval is not None:
            return fold_eval

        indices = self.folds[fold_number]
        fold_eval = self.eval_fold(config, indices, **self.eval_kwargs)
        self.total_cost += 1
        self.memory_dict.setdefault(config_key, {})[fold_number] = fold_eval
        return fold_eval

    def update_best(self, config: Dict, result: float):
        if self.best_eval is None or result > self.best_eval:
            self.best_config, self.best_eval = config, result

    def log_eval(self, config: Dict, result: float):
        self.update_best(config, result)
        self.log.append(
            [self.run_id, self.run_name, self.iteration_number, self.best_eval, time.time() - self.start_time, self.total_cost]
        )

    def eval_config(self, config: Dict, budget: int) -> float:
        results = [self.cached_eval_fold(config, fold) for fold in range(budget)]
        result = np.mean(results)
        if budget == self.max_eval_budget:
            self.log_eval(config, result)
        return result

    def successive_halving(self, configs: List[Dict]) -> List[Dict]:
        population = configs.copy()
        for budget in self.budgets:
            scores = [self.eval_config(config, budget) for config in population]
            scored_pop = [{"config": config, "score": score} for config, score in zip(population, scores)]
            scored_pop.sort(key=lambda x: x["score"], reverse=True)
            population = [indiv["config"] for indiv in scored_pop[:math.ceil(len(scored_pop) * self.eta)]]
        return population

    def run_episode(self):
        raise NotImplementedError()

    def run(self) -> Tuple:
        self.start_time = time.time()
        while self.total_cost < self.max_budget:
            self.run_episode()
        return (self.best_config, self.best_eval, time.time() - self.start_time, self.total_cost, self.log)

class RandomSearch(BaseOptimizer):
    def __init__(
        self,
        space: HyperParameterSpace,
        eval_fold: Callable,
        folds: List[Tuple],
        budgets: Union[None, List[int]] = None,
        max_budget: float = 1000,
        eta: float = 0,
        pop_size: int = 10,
        run_id: int = 0,
        run_name: str = "RS",
        memory_dict: dict = {},
        eval_kwargs: dict = {},
    ):
        super().__init__(
            space,
            eval_fold,
            folds,
            budgets,
            eta=eta,
            max_budget=max_budget,
            run_id=run_id,
            run_name=run_name,
            memory_dict=memory_dict,
            eval_kwargs=eval_kwargs,
        )
        self.pop_size = pop_size

    def run_episode(self):
        """Samples a population and executes successive halving"""
        # Sample a population of configurations
        population = [self.space.sample_config() for _ in range(self.pop_size)]
        # Execute successive halving on the sampled population
        population = self.successive_halving(population)
        # Update the best configuration found in this episode
        for config in population:
            eval_result = self.eval_config(config, self.max_eval_budget)
            self.update_best(config, eval_result)

