import math
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional
from focus_opt.hp_space import HyperParameterSpace
from focus_opt.config_candidate import ConfigCandidate
from focus_opt.helpers import OutOfBudgetError, SessionContext
import logging
import inspect
import random
import copy

logging.basicConfig(level=logging.INFO)


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers in the focus_opt package.

    This class provides common functionality and interfaces for different optimization
    algorithms. It handles hyperparameter space management, evaluation function integration,
    and tracking of the best candidate found during optimization.
    """

    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable[[Dict[str, Any], int], float],
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Optional[Callable[[List[float]], float]] = None,
        initial_config: Optional[dict] = None,
        log_results: bool = False,
    ):
        """
        Initialize the BaseOptimizer.

        Args:
            hp_space (HyperParameterSpace): The hyperparameter space to explore.
            evaluation_function (Callable[[Dict[str, Any], int], float]): 
                Function to evaluate a hyperparameter configuration. 
                It should accept a config dict and a fidelity level, returning a performance score.
            max_fidelity (int, optional): Maximum fidelity level for evaluations. Defaults to 1.
            sh_eta (float, optional): Successive halving proportion. Defaults to 0.5.
            maximize (bool, optional): Whether to maximize the evaluation score. Defaults to False.
            score_aggregation (str, optional): Method to aggregate scores across fidelities. 
                Options include "average". Defaults to "average".
            score_aggregation_function (Optional[Callable[[List[float]], float]], optional): 
                Custom function to aggregate scores. If None, defaults to the method specified 
                by score_aggregation. Defaults to None.
            initial_config (Optional[dict], optional): Initial hyperparameter configuration to start optimization. 
                Defaults to None.
            log_results (bool, optional): Whether to log the results of evaluations. Defaults to False.
        """
        self.hp_space = hp_space
        self.evaluation_function = evaluation_function
        self.max_fidelity = max_fidelity
        self.sh_eta = sh_eta
        self.maximize = maximize
        self.best_candidate: Optional[ConfigCandidate] = None
        self.score_aggregation = score_aggregation
        self.score_aggregation_function = score_aggregation_function
        self.initial_config = initial_config
        self.log_results = log_results

        # Check if the evaluation function accepts a fidelity level
        self.accepts_fidelity = self._check_accepts_fidelity()

    def _check_accepts_fidelity(self) -> bool:
        """
        Check if the evaluation function accepts a fidelity parameter.

        Returns:
            bool: True if the evaluation function accepts a fidelity level, False otherwise.
        """
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

    def config_to_candidate(self, config: dict) -> ConfigCandidate:
        """
        Instantiate a ConfigCandidate from a configuration dictionary.

        Args:
            config (dict): Hyperparameter configuration.

        Returns:
            ConfigCandidate: An instance of ConfigCandidate initialized with the given config.
        """
        return ConfigCandidate(
            config=config,
            evaluation_function=self.evaluation_function,
            score_aggregation=self.score_aggregation,
            score_aggregation_function=self.score_aggregation_function,
            accepts_fidelity=self.accepts_fidelity,
            max_fidelity=self.max_fidelity,
        )

    def configs_to_candidates(self, configs: List[dict]) -> List[ConfigCandidate]:
        """
        Instantiate a list of ConfigCandidates from a list of configuration dictionaries.

        Args:
            configs (List[dict]): List of hyperparameter configurations.

        Returns:
            List[ConfigCandidate]: List of ConfigCandidate instances.
        """
        return [self.config_to_candidate(config) for config in configs]

    def compare_candidates(
        self,
        candidate_1: ConfigCandidate,
        candidate_2: ConfigCandidate
    ) -> bool:
        """
        Compare two candidates based on the optimization objective.

        Args:
            candidate_1 (ConfigCandidate): The first candidate to compare.
            candidate_2 (ConfigCandidate): The second candidate to compare.

        Returns:
            bool: True if candidate_1 is better than candidate_2 based on the objective, False otherwise.
        """
        if self.maximize:
            return candidate_1.evaluation_score > candidate_2.evaluation_score
        else:
            return candidate_1.evaluation_score < candidate_2.evaluation_score

    def successive_halving(
        self,
        candidates: List[ConfigCandidate],
        session_context: SessionContext,
        min_population_size: Optional[int] = None
    ) -> List[ConfigCandidate]:
        """
        Perform the successive halving algorithm on a list of candidates.

        Args:
            candidates (List[ConfigCandidate]): List of candidates to evaluate.
            session_context (SessionContext): Context managing the optimization session.
            min_population_size (Optional[int], optional): Minimum population size to maintain. 
                If None, no lower bound is enforced. Defaults to None.

        Returns:
            List[ConfigCandidate]: Reduced list of candidates after successive halving.
        """
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

    def update_best(self, candidates: List[ConfigCandidate]) -> None:
        """
        Update the best candidate found so far based on the current list of candidates.

        Args:
            candidates (List[ConfigCandidate]): List of candidates to consider.
        """
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

    @abstractmethod
    def optimize(self) -> ConfigCandidate:
        """
        Abstract method to perform optimization.

        Must be implemented by subclasses.

        Returns:
            ConfigCandidate: The best candidate found during optimization.
        """
        raise NotImplementedError


def log_trial(candidate: ConfigCandidate, success: bool = True) -> None:
    """
    Log the result of a trial evaluation.

    Args:
        candidate (ConfigCandidate): The candidate that was evaluated.
        success (bool, optional): Whether the evaluation was successful. Defaults to True.
    """
    if success:
        logging.info(f"Trial successful: {candidate}")
    else:
        logging.error(f"Trial failed: {candidate}")


class RandomSearchOptimizer(BaseOptimizer):
    """
    Optimizer that performs random search over the hyperparameter space.

    This optimizer samples hyperparameter configurations randomly and evaluates them
    using successive halving to identify the best configuration within a given budget.
    """

    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable[[Dict[str, Any], int], float],
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Optional[Callable[[List[float]], float]] = None,
        log_results: bool = False,
    ):
        """
        Initialize the RandomSearchOptimizer.

        Args:
            hp_space (HyperParameterSpace): The hyperparameter space to explore.
            evaluation_function (Callable[[Dict[str, Any], int], float]): 
                Function to evaluate a hyperparameter configuration.
            max_fidelity (int, optional): Maximum fidelity level for evaluations. Defaults to 1.
            sh_eta (float, optional): Successive halving proportion. Defaults to 0.5.
            maximize (bool, optional): Whether to maximize the evaluation score. Defaults to False.
            score_aggregation (str, optional): Method to aggregate scores. Defaults to "average".
            score_aggregation_function (Optional[Callable[[List[float]], float]], optional): 
                Custom score aggregation function. Defaults to None.
            log_results (bool, optional): Whether to log evaluation results. Defaults to False.
        """
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

    def optimize(self, population_size: int = 10, budget: int = 100, max_time: Optional[int] = None) -> ConfigCandidate:
        """
        Perform random search optimization.

        Samples configurations randomly and evaluates them using successive halving
        until the budget or time is exhausted.

        Args:
            population_size (int, optional): Number of configurations to sample in each iteration. Defaults to 10.
            budget (int, optional): Total number of evaluations allowed. Defaults to 100.
            max_time (Optional[int], optional): Maximum time (in seconds) allowed for optimization. 
                If None, no time limit is imposed. Defaults to None.

        Returns:
            ConfigCandidate: The best candidate found during optimization.
        """
        session_context = SessionContext(budget=budget, max_time=max_time, log_results=self.log_results)
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
    """
    Optimizer that uses the hill climbing algorithm for hyperparameter optimization.

    This optimizer starts from an initial configuration (or random ones) and iteratively
    explores neighboring configurations to find local optima. It supports multiple random
    restarts to escape local optima and search for the global optimum.
    """

    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable[[Dict[str, Any], int], float],
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Optional[Callable[[List[float]], float]] = None,
        initial_config: Optional[dict] = None,
        warm_start: int = 0,
        random_restarts: int = 5,
        log_results: bool = False,
    ):
        """
        Initialize the HillClimbingOptimizer.

        Args:
            hp_space (HyperParameterSpace): The hyperparameter space to explore.
            evaluation_function (Callable[[Dict[str, Any], int], float]): 
                Function to evaluate a hyperparameter configuration.
            max_fidelity (int, optional): Maximum fidelity level for evaluations. Defaults to 1.
            sh_eta (float, optional): Successive halving proportion. Defaults to 0.5.
            maximize (bool, optional): Whether to maximize the evaluation score. Defaults to False.
            score_aggregation (str, optional): Method to aggregate scores. Defaults to "average".
            score_aggregation_function (Optional[Callable[[List[float]], float]], optional): 
                Custom score aggregation function. Defaults to None.
            initial_config (Optional[dict], optional): Initial hyperparameter configuration. 
                If None, starts with random configurations. Defaults to None.
            warm_start (int, optional): Number of initial configurations to explore randomly. Defaults to 0.
            random_restarts (int, optional): Number of random restarts to perform to escape local optima. Defaults to 5.
            log_results (bool, optional): Whether to log evaluation results. Defaults to False.
        """
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

    def hill_climbing_round(self, session_context: SessionContext, restart_number: int = 0) -> ConfigCandidate:
        """
        Perform a single hill climbing round.

        Starts from an initial configuration and iteratively explores neighboring configurations
        to find a better candidate. Stops when no better neighbors are found or when the budget/time is exhausted.

        Args:
            session_context (SessionContext): Context managing the optimization session.
            restart_number (int, optional): The current restart iteration number. Defaults to 0.

        Returns:
            ConfigCandidate: The best candidate found in this hill climbing round.
        """
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

    def optimize(self, max_time: Optional[int] = None, budget: int = 100) -> ConfigCandidate:
        """
        Perform hill climbing optimization with random restarts.

        Initiates multiple hill climbing rounds with random restarts to explore different regions
        of the hyperparameter space and avoid getting stuck in local optima.

        Args:
            max_time (Optional[int], optional): Maximum time (in seconds) allowed for optimization. 
                If None, no time limit is imposed. Defaults to None.
            budget (int, optional): Total number of evaluations allowed. Defaults to 100.

        Returns:
            ConfigCandidate: The best candidate found during optimization.
        """
        session_context = SessionContext(budget=budget, max_time=max_time, log_results=self.log_results)

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
    """
    Optimizer that uses a genetic algorithm for hyperparameter optimization.

    This optimizer maintains a population of candidate configurations and evolves them
    through selection, crossover, and mutation to find the best configuration within a given budget.
    """

    def __init__(
        self,
        hp_space: HyperParameterSpace,
        evaluation_function: Callable[[Dict[str, Any], int], float],
        max_fidelity: int = 1,
        sh_eta: float = 0.5,
        maximize: bool = False,
        score_aggregation: str = "average",
        score_aggregation_function: Optional[Callable[[List[float]], float]] = None,
        population_size: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int = 1,
        tournament_size: int = 3,
        min_population_size: int = 5,
        log_results: bool = False,
    ):
        """
        Initialize the GeneticAlgorithmOptimizer.

        Args:
            hp_space (HyperParameterSpace): The hyperparameter space to explore.
            evaluation_function (Callable[[Dict[str, Any], int], float]): 
                Function to evaluate a hyperparameter configuration.
            max_fidelity (int, optional): Maximum fidelity level for evaluations. Defaults to 1.
            sh_eta (float, optional): Successive halving proportion. Defaults to 0.5.
            maximize (bool, optional): Whether to maximize the evaluation score. Defaults to False.
            score_aggregation (str, optional): Method to aggregate scores. Defaults to "average".
            score_aggregation_function (Optional[Callable[[List[float]], float]], optional): 
                Custom score aggregation function. Defaults to None.
            population_size (int, optional): Number of individuals in the population. Defaults to 20.
            crossover_rate (float, optional): Probability of crossover between parents. Defaults to 0.8.
            mutation_rate (float, optional): Probability of mutation in offspring. Defaults to 0.1.
            elitism (int, optional): Number of top individuals to carry over to the next generation. Defaults to 1.
            tournament_size (int, optional): Number of individuals competing in tournament selection. Defaults to 3.
            min_population_size (int, optional): Minimum population size to maintain diversity. Defaults to 5.
            log_results (bool, optional): Whether to log evaluation results. Defaults to False.
        """
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
        """
        Perform crossover between two parent configurations to produce an offspring.

        For each hyperparameter, the offspring inherits the value from one of the parents
        based on the crossover rate.

        Args:
            parent1 (dict): The first parent configuration.
            parent2 (dict): The second parent configuration.

        Returns:
            dict: The offspring configuration resulting from crossover.
        """
        offspring = {}
        for key in parent1.keys():
            if random.random() < self.crossover_rate:
                offspring[key] = parent1[key]
            else:
                offspring[key] = parent2[key]
        return offspring

    def mutate(self, config: dict) -> dict:
        """
        Perform mutation on a configuration.

        Each hyperparameter has a chance to be mutated based on the mutation rate.
        Mutation replaces the hyperparameter value with a new sampled value from its space.

        Args:
            config (dict): The configuration to mutate.

        Returns:
            dict: The mutated configuration.
        """
        mutated_config = copy.deepcopy(config)
        for key in mutated_config.keys():
            if random.random() < self.mutation_rate:
                mutated_config[key] = self.hp_space.hp_dict[key].sample()
        return mutated_config

    def select_parents(self, candidates: List[ConfigCandidate]) -> List[ConfigCandidate]:
        """
        Select parents for crossover using tournament selection.

        Args:
            candidates (List[ConfigCandidate]): List of fully evaluated candidates.

        Returns:
            List[ConfigCandidate]: List of selected parent candidates.
        """
        fully_evaluated_candidates = [c for c in candidates if c.is_fully_evaluated]
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(fully_evaluated_candidates, k=self.tournament_size)
            winner = sorted(tournament, key=lambda x: x.evaluation_score, reverse=self.maximize)[0]
            selected_parents.append(winner)
        return selected_parents

    def generate_offspring(self, parents: List[ConfigCandidate]) -> List[dict]:
        """
        Generate new population configurations through crossover and mutation.

        Args:
            parents (List[ConfigCandidate]): List of parent candidates selected for reproduction.

        Returns:
            List[dict]: List of offspring configurations.
        """
        new_population_configs = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i].config
            parent2 = parents[(i + 1) % len(parents)].config
            offspring1 = self.crossover(parent1, parent2)
            offspring2 = self.crossover(parent2, parent1)
            new_population_configs.append(self.mutate(offspring1))
            new_population_configs.append(self.mutate(offspring2))
        return new_population_configs

    def optimize(self, budget: int = 100, max_time: Optional[int] = None) -> ConfigCandidate:
        """
        Perform genetic algorithm optimization.

        Evolves a population of configurations through selection, crossover, and mutation
        until the budget or time is exhausted.

        Args:
            budget (int, optional): Total number of evaluations allowed. Defaults to 100.
            max_time (Optional[int], optional): Maximum time (in seconds) allowed for optimization. 
                If None, no time limit is imposed. Defaults to None.

        Returns:
            ConfigCandidate: The best candidate found during optimization.
        """
        session_context = SessionContext(budget=budget, max_time=max_time, log_results=self.log_results)

        # Initialize population
        population_configs = self.hp_space.sample_configs(n_configs=self.population_size)
        if self.initial_config:
            population_configs.append(self.initial_config)
        population_candidates = self.configs_to_candidates(population_configs)

        while session_context.can_continue_running():

            try:
                # Evaluate population
                population_candidates = self.successive_halving(
                    population_candidates,
                    session_context,
                    min_population_size=self.min_population_size
                )
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
            elite_candidates = sorted(
                population_candidates,
                key=lambda x: x.evaluation_score,
                reverse=self.maximize
            )[:self.elitism]
            new_population_configs.extend([candidate.config for candidate in elite_candidates])

            # Ensure the population size remains constant
            new_population_configs = new_population_configs[:self.population_size]
            population_candidates = self.configs_to_candidates(new_population_configs)

        return self.best_candidate