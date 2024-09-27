import math
from abc import ABC
from typing import Callable, List
from focus_opt.hp_space import HyperParameterSpace
from focus_opt.config_candidate import ConfigCandidate
from focus_opt.helpers import OutOfBudgetError, SessionContext
import logging
import inspect
import random
import copy


logging.basicConfig(level=logging.INFO)

class BaseOptimizer(ABC):
    def __init__(self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Callable = None,
        initial_config: dict = None,
        log_results: bool = False,
    ):
        self.hp_space = hp_space
        self.evaluation_function = evaluation_function
        self.max_fidelity = max_fidelity
        self.sh_eta = sh_eta
        self.maximize = maximize
        self.best_candidate = None
        self.score_aggregation = score_aggregation
        self.score_aggregation_function = score_aggregation_function
        self.initial_config = initial_config
        self.log_results = log_results

        # Check if the evaluation function accepts a fidelity level
        self.accepts_fidelity = self._check_accepts_fidelity()

    def _check_accepts_fidelity(self):
        signature = inspect.signature(self.evaluation_function)
        parameters = list(signature.parameters.values())

        if len(parameters) != 2:
            return False

        try:
            self.evaluation_function({}, 1)
        except TypeError:
            return False
        except Exception as e:
            logging.error(f"An error occurred: {e}")

            pass

        return True

    def config_to_candidate(self, config: dict):
        """Instantiates a ConfigCandidate from a config dict"""
        return ConfigCandidate(
            config=config,
            evaluation_function=self.evaluation_function,
            score_aggregation=self.score_aggregation,
            score_aggregation_function=self.score_aggregation_function,
            accepts_fidelity=self.accepts_fidelity,
            max_fidelity=self.max_fidelity,
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
        Returns true if the first candidate is 'better' than the second candidate
        based on the optimisation objective
        """
        if self.maximize:
            return candidate_1.evaluation_score > candidate_2.evaluation_score
        else:
            return candidate_1.evaluation_score < candidate_2.evaluation_score

    def successive_halving(
            self,
            candidates: List[ConfigCandidate],
            session_context: SessionContext,
            min_population_size: int = None
        ):
        for _ in range(self.max_fidelity):
            for candidate in candidates:
                try:
                    candidate.evaluate(session_context)
                    log_trial(candidate, success=True)
                except OutOfBudgetError:
                    raise
                except TimeoutError:
                    raise
                except Exception as e:
                    logging.error(f"Evaluation failed for candidate {candidate.config}: {e}")
                    log_trial(candidate, success=False)
            if min_population_size:
                sh_cutoff = max(math.ceil(len(candidates) * self.sh_eta), min_population_size)
            else:
                sh_cutoff = math.ceil(len(candidates) * self.sh_eta)

            candidates = sorted(
                candidates,
                key=lambda x: x.evaluation_score,
                reverse=self.maximize
            )[:sh_cutoff]
        return candidates


    def update_best(self, candidates: List[ConfigCandidate]):
        fully_evaluated_candidates = [cand for cand in candidates if cand.is_fully_evaluated]
        if len(fully_evaluated_candidates) == 0:
            return
        new_candidate = sorted(
            fully_evaluated_candidates,
            key=lambda x: x.evaluation_score,
            reverse=self.maximize
        )[0]
        if (
            self.best_candidate is None or
            self.compare_candidates(new_candidate, self.best_candidate)
        ):
            self.best_candidate = new_candidate

    def optimize(self):
        raise NotImplementedError

def log_trial(candidate, success=True):
        if success:
            logging.info(f"Trial successful: {candidate}")
        else:
            logging.error(f"Trial failed: {candidate}")


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Callable = None,
        log_results: bool = False,
    ):
        super().__init__(
            hp_space,
            evaluation_function,
            max_fidelity,
            sh_eta,
            maximize,
            score_aggregation,
            score_aggregation_function,
            log_results=log_results,
        )

    def optimize(self, population_size = 10, budget: int = 100,  max_time = None):
        session_context = SessionContext(budget = budget, max_time = max_time, log_results = self.log_results)
        while session_context.can_continue_running():
            candidates = self.configs_to_candidates(
                self.hp_space.sample_configs(n_configs=population_size)
            )
            try:
                candidates = self.successive_halving(candidates, session_context)
            except Exception as e:
                logging.error(f"An exception occurred: {e}")
                self.update_best(candidates)
                break
            self.update_best(candidates)
        return self.best_candidate


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Callable = None,
        initial_config: dict = None,
        warm_start: int = 0,
        random_restarts: int = 5,
        log_results: bool = False,
    ):
        super().__init__(
            hp_space,
            evaluation_function,
            max_fidelity,
            sh_eta,
            maximize,
            score_aggregation,
            score_aggregation_function,
            initial_config=initial_config,
            log_results=log_results
        )
        self.warm_start = warm_start
        self.random_restarts = random_restarts

    def hill_climbing_round(self, session_context, restart_number: int = 0):

        starting_configs = []
        if self.initial_config and restart_number == 0:
            starting_configs.append(self.initial_config)

        starting_configs.extend(
            self.hp_space.sample_configs(n_configs=max(self.warm_start, 1))
        )

        starting_candidates = self.configs_to_candidates(starting_configs)
        try:
            starting_candidates = self.successive_halving(starting_candidates, session_context=session_context)
        except Exception as e:
            logging.error(f"An exception occurred during starting candidates evaluation {e}")
            self.update_best(starting_candidates)
            return self.best_candidate

        current_candidate = starting_candidates[0]

        while session_context.can_continue_running():

            neighbors = self.configs_to_candidates(
                self.hp_space.sample_all_neighbors(current_candidate.config)
            )

            try:
                candidates = self.successive_halving(neighbors, session_context)
            except Exception as e:
                logging.error(f"An exception occurred {e}")
                break

            best_neighbor = candidates[0]
            if self.compare_candidates(best_neighbor, current_candidate):
                current_candidate = best_neighbor
            else:
                logging.info(f"Local optimum achieved with candidate: {current_candidate}")
                break

            self.update_best([current_candidate])

        return current_candidate

    def optimize(self, max_time: int = None, budget: int = 100):
        session_context = SessionContext(budget=budget, max_time=max_time, log_results = self.log_results)

        for restart in range(self.random_restarts):
            logging.info(f"Random restart {restart + 1}/{self.random_restarts}")

            current_candidate = self.hill_climbing_round(session_context, restart_number=restart)

            if self.best_candidate is None or self.compare_candidates(current_candidate, self.best_candidate):
                self.best_candidate = current_candidate

            try:
                session_context.budget_error_checks()
            except Exception as e:
                logging.error(f"An exception occurred {e}")
                break

        return self.best_candidate


class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable,
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Callable = None,
        population_size: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int = 1,
        tournament_size: int = 3,
        min_population_size: int = 5,
        log_results: bool = False,
    ):
        super().__init__(
            hp_space,
            evaluation_function,
            max_fidelity,
            sh_eta,
            maximize,
            score_aggregation,
            score_aggregation_function,
            log_results=log_results
        )
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.min_population_size = min_population_size

    def crossover(self, parent1: dict, parent2: dict) -> dict:
        """Perform crossover between two parents to produce an offspring."""
        offspring = {}
        for key in parent1.keys():
            if random.random() < self.crossover_rate:
                offspring[key] = parent1[key]
            else:
                offspring[key] = parent2[key]
        return offspring

    def mutate(self, config: dict) -> dict:
        """Perform mutation on a configuration."""
        mutated_config = copy.deepcopy(config)
        for key in mutated_config.keys():
            if random.random() < self.mutation_rate:
                mutated_config[key] = self.hp_space.hp_dict[key].sample()
        return mutated_config

    def select_parents(self, candidates: List[ConfigCandidate]) -> List[ConfigCandidate]:
        """Select parents for crossover using tournament selection."""
        fully_evaluated_candidates = [c for c in candidates if c.is_fully_evaluated]
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(fully_evaluated_candidates, k=self.tournament_size)
            winner = sorted(tournament, key=lambda x: x.evaluation_score, reverse=self.maximize)[0]
            selected_parents.append(winner)
        return selected_parents

    def generate_offspring(self, parents: List[ConfigCandidate]) -> List[dict]:
        """Generate new population through crossover and mutation."""
        new_population_configs = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i].config
            parent2 = parents[(i + 1) % len(parents)].config
            offspring1 = self.crossover(parent1, parent2)
            offspring2 = self.crossover(parent2, parent1)
            new_population_configs.append(self.mutate(offspring1))
            new_population_configs.append(self.mutate(offspring2))
        return new_population_configs

    def optimize(self, budget:int = 100, max_time=None):
        session_context = SessionContext(budget=budget, max_time=max_time, log_results = self.log_results)

        # Initialise population
        population_configs = self.hp_space.sample_configs(n_configs=self.population_size)
        if self.initial_config:
            population_configs.append(self.initial_config)
        population_candidates = self.configs_to_candidates(population_configs)

        while session_context.can_continue_running():

            try:
                # Evaluate population
                population_candidates = self.successive_halving(population_candidates, session_context, min_population_size=self.min_population_size)
            except Exception as e:
                logging.error(e)
                self.update_best(population_candidates)
                break

            # Update the best candidate
            self.update_best(population_candidates)

            # Select parents
            parents = self.select_parents(population_candidates)

            # Generate new population through crossover and mutation
            new_population_configs = self.generate_offspring(parents)

            # Apply elitism
            elite_candidates = sorted(population_candidates, key=lambda x: x.evaluation_score, reverse=self.maximize)[:self.elitism]
            new_population_configs.extend([candidate.config for candidate in elite_candidates])

            # Ensure the population size remains constant
            new_population_configs = new_population_configs[:self.population_size]
            population_candidates = self.configs_to_candidates(new_population_configs)

        return self.best_candidate
