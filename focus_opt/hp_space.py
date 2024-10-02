from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class HyperParameter(ABC):
    """Abstract base class for hyperparameters.

    :param name: The name of the hyperparameter.
    :type name: str
    :param values: The possible values for the hyperparameter, defaults to None.
    :type values: Union[List[Any], None], optional
    """

    def __init__(self, name: str, values: Union[List[Any], None] = None):
        """Constructor method
        """
        self.name = name
        self.values = values

    @property
    def param_type(self) -> str:
        """Returns the type of the hyperparameter.

        :return: The type of the hyperparameter.
        :rtype: str
        """
        return self.__class__.__name__

    def __str__(self):
        """String representation of the HyperParameter object.

        :return: A string describing the hyperparameter.
        :rtype: str
        """
        return f"{self.name} ({self.param_type}): {self.values}"

    @abstractmethod
    def sample(self) -> Any:
        """Abstract method to sample a value for the hyperparameter.

        :return: A sampled value.
        :rtype: Any
        """
        pass

    @abstractmethod
    def sample_neighbors(self, value: Any) -> List[Any]:
        """Abstract method to sample neighboring values for a given value.

        :param value: The current value of the hyperparameter.
        :type value: Any
        :return: A list of neighboring values.
        :rtype: List[Any]
        """
        pass

class BooleanHyperParameter(HyperParameter):
    """Boolean hyperparameter with a probability of being True.

    :param name: The name of the hyperparameter.
    :type name: str
    :param proba_true: Probability of sampling True, defaults to 0.5.
    :type proba_true: float, optional
    """

    def __init__(self, name: str, proba_true: float = 0.5):
        """Constructor method
        """
        super().__init__(name)
        self.proba_true = proba_true
        self.values = [True, False]

    def sample(self) -> bool:
        """Samples a boolean value based on the probability.

        :return: A sampled boolean value.
        :rtype: bool
        """
        return np.random.rand() < self.proba_true

    def sample_neighbors(self, value: bool) -> List[bool]:
        """Returns the opposite boolean value as the neighbor.

        :param value: The current boolean value.
        :type value: bool
        :return: A list containing the opposite value.
        :rtype: List[bool]
        """
        return [not value]

class CategoricalHyperParameter(HyperParameter):
    """Categorical hyperparameter with a list of possible values.

    :param name: The name of the hyperparameter.
    :type name: str
    :param values: The list of possible values.
    :type values: List[Any]
    """

    def __init__(self, name: str, values: List[Any]):
        """Constructor method
        """
        super().__init__(name, values)

    def sample(self) -> Any:
        """Samples a value from the list of possible values.

        :return: A sampled value.
        :rtype: Any
        """
        return np.random.choice(self.values)

    def sample_neighbors(self, value: Any) -> List[Any]:
        """Samples two random neighboring values from the list, excluding the current value.

        :param value: The current value.
        :type value: Any
        :return: A list of neighboring values.
        :rtype: List[Any]
        :raises ValueError: If no neighbors are available.
        """
        if len(self.values) <= 1:
            raise ValueError("No neighbors available for single-value parameters.")
        other_values = [val for val in self.values if val != value]
        return np.random.choice(other_values, size=min(2, len(other_values)), replace=False).tolist()

class OrdinalHyperParameter(CategoricalHyperParameter):
    """Ordinal hyperparameter with ordered values.

    :param name: The name of the hyperparameter.
    :type name: str
    :param values: The ordered list of possible values.
    :type values: List[Any]
    """

    def sample_neighbors(self, value: Any) -> List[Any]:
        """Samples neighboring values based on the order of values.

        :param value: The current value.
        :type value: Any
        :return: A list of neighboring values.
        :rtype: List[Any]
        """
        idx = self.values.index(value)
        neighbors = [self.values[i] for i in [idx - 1, idx + 1] if 0 <= i < len(self.values)]
        return neighbors

class ContinuousHyperParameter(HyperParameter):
    """Continuous hyperparameter with a range of values.

    :param name: The name of the hyperparameter.
    :type name: str
    :param min_value: The minimum value of the range.
    :type min_value: float
    :param max_value: The maximum value of the range.
    :type max_value: float
    :param is_int: Whether the values should be integers, defaults to False.
    :type is_int: bool, optional
    :param step_size: The step size for sampling neighbors, defaults to 0.1.
    :type step_size: float, optional
    """

    def __init__(self, name: str, min_value: float, max_value: float, is_int: bool = False, step_size: float = 0.1):
        """Constructor method
        """
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.is_int = is_int
        self.step_size = step_size

    def sample(self) -> float:
        """Samples a value within the range.

        :return: A sampled value.
        :rtype: float
        """
        value = np.random.uniform(self.min_value, self.max_value)
        return int(value) if self.is_int else value

    def sample_neighbors(self, value: float) -> List[float]:
        """Samples neighboring values based on the step size.

        :param value: The current value.
        :type value: float
        :return: A list of neighboring values.
        :rtype: List[float]
        """
        step = (self.max_value - self.min_value) * self.step_size
        neighbors = []

        if value - step >= self.min_value:
            neighbors.append(value - step)
        if value + step <= self.max_value:
            neighbors.append(value + step)

        if self.is_int:
            neighbors = list(map(int, neighbors))

        return neighbors

class HyperParameterSpace:
    """Represents a space of hyperparameters for configuration sampling.

    :param name: The name of the hyperparameter space.
    :type name: str
    :param hyper_parameters: A list of hyperparameters in the space, defaults to an empty list.
    :type hyper_parameters: List[HyperParameter], optional
    """

    def __init__(self, name: str, hyper_parameters: List[HyperParameter] = []):
        """Constructor method
        """
        self.name = name
        self.hps = hyper_parameters
        self.hp_dict = {hp.name: hp for hp in self.hps}

    def add_hp(self, hp: HyperParameter):
        """Adds a hyperparameter to the space.

        :param hp: The hyperparameter to add.
        :type hp: HyperParameter
        """
        self.hps.append(hp)
        self.hp_dict[hp.name] = hp

    def sample_config(self) -> Dict[str, Any]:
        """Samples a configuration from the hyperparameter space.

        :return: A dictionary representing the sampled configuration.
        :rtype: Dict[str, Any]
        """
        return {hp.name: hp.sample() for hp in self.hps}

    def sample_configs(self, n_configs: int) -> List[Dict[str, Any]]:
        """Samples multiple configurations from the hyperparameter space.

        :param n_configs: The number of configurations to sample.
        :type n_configs: int
        :return: A list of sampled configurations.
        :rtype: List[Dict[str, Any]]
        """
        configs = []
        while len(configs) < n_configs:
            config = self.sample_config()
            configs.append(config)
        return [dict(config) for config in configs]

    def sample_unique_configs(self, n_configs: int) -> List[Dict[str, Any]]:
        """Samples unique configurations from the hyperparameter space.

        :param n_configs: The number of unique configurations to sample.
        :type n_configs: int
        :return: A list of unique sampled configurations.
        :rtype: List[Dict[str, Any]]
        """
        configs = set()
        while len(configs) < n_configs:
            config = self.sample_config()
            configs.add(tuple(config.items()))
        return [dict(config) for config in configs]

    def sample_n_neighbors(self, config: Dict[str, Any], n_neighbors: int = 1) -> List[Dict[str, Any]]:
        """Samples neighboring configurations for a given configuration.

        :param config: The current configuration.
        :type config: Dict[str, Any]
        :param n_neighbors: The number of neighbors to sample, defaults to 1.
        :type n_neighbors: int, optional
        :return: A list of neighboring configurations.
        :rtype: List[Dict[str, Any]]
        """
        neighbors = []
        for _ in range(n_neighbors):
            new_config = config.copy()
            hp_to_modify = np.random.choice(self.hps)
            new_config[hp_to_modify.name] = hp_to_modify.sample_neighbors(config[hp_to_modify.name])[0]
            neighbors.append(new_config)
        return neighbors

    def sample_all_neighbors(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Samples all possible neighboring configurations for a given configuration.

        :param config: The current configuration.
        :type config: Dict[str, Any]
        :return: A list of all neighboring configurations.
        :rtype: List[Dict[str, Any]]
        """
        neighbors = []
        for hp in self.hps:
            current_value = config[hp.name]
            try:
                neighbor_values = hp.sample_neighbors(current_value)
                for neighbor_value in neighbor_values:
                    new_config = config.copy()
                    new_config[hp.name] = neighbor_value
                    neighbors.append(new_config)
            except ValueError as e:
                logging.warning(f"No neighbors available for hyperparameter {hp.name}: {e}")
        return neighbors

    def __str__(self):
        """String representation of the HyperParameterSpace object.

        :return: A string describing the hyperparameter space.
        :rtype: str
        """
        return f"HyperParameterSpace: {self.name}\n" + "\n".join(str(hp) for hp in self.hps)
