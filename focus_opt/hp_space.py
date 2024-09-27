from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class HyperParameter(ABC):
    def __init__(self, name: str, values: Union[List[Any], None] = None):
        self.name = name
        self.values = values

    @property
    def param_type(self) -> str:
        return self.__class__.__name__

    def __str__(self):
        return f"{self.name} ({self.param_type}): {self.values}"

    @abstractmethod
    def sample(self) -> Any:
        pass

    @abstractmethod
    def sample_neighbors(self, value: Any) -> List[Any]:
        pass

class BooleanHyperParameter(HyperParameter):
    def __init__(self, name: str, proba_true: float = 0.5):
        super().__init__(name)
        self.proba_true = proba_true
        self.values = [True, False]

    def sample(self) -> bool:
        return np.random.rand() < self.proba_true

    def sample_neighbors(self, value: bool) -> List[bool]:
        return [not value]

class CategoricalHyperParameter(HyperParameter):
    def __init__(self, name: str, values: List[Any]):
        super().__init__(name, values)

    def sample(self) -> Any:
        return np.random.choice(self.values)

    def sample_neighbors(self, value: Any) -> List[Any]:
        if len(self.values) <= 1:
            raise ValueError("No neighbors available for single-value parameters.")
        other_values = [val for val in self.values if val != value]
        return np.random.choice(other_values, size=min(2, len(other_values)), replace=False).tolist()

class OrdinalHyperParameter(CategoricalHyperParameter):
    def sample_neighbors(self, value: Any) -> List[Any]:
        idx = self.values.index(value)
        neighbors = [self.values[i] for i in [idx - 1, idx + 1] if 0 <= i < len(self.values)]
        return neighbors

class ContinuousHyperParameter(HyperParameter):
    def __init__(self, name: str, min_value: float, max_value: float, is_int: bool = False, step_size: float = 0.1):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.is_int = is_int
        self.step_size = step_size

    def sample(self) -> float:
        value = np.random.uniform(self.min_value, self.max_value)
        return int(value) if self.is_int else value

    def sample_neighbors(self, value: float) -> List[float]:
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
    def __init__(self, name: str, hyper_parameters: List[HyperParameter] = []):
        self.name = name
        self.hps = hyper_parameters
        self.hp_dict = {hp.name:hp for hp in self.hps}

    def add_hp(self, hp: HyperParameter):
        self.hps.append(hp)
        self.hp_dict[hp.name] = hp

    def sample_config(self) -> Dict[str, Any]:
        return {hp.name: hp.sample() for hp in self.hps}

    def sample_configs(self, n_configs: int) -> List[Dict[str, Any]]:
        configs = []
        while len(configs) < n_configs:
            config = self.sample_config()
            configs.append(config)
        return [dict(config) for config in configs]

    def sample_unique_configs(self, n_configs: int) -> List[Dict[str, Any]]:
        configs = set()
        while len(configs) < n_configs:
            config = self.sample_config()
            configs.add(tuple(config.items()))
        return [dict(config) for config in configs]

    def sample_n_neighbors(self, config: Dict[str, Any], n_neighbors: int = 1) -> List[Dict[str, Any]]:
        neighbors = []
        for _ in range(n_neighbors):
            new_config = config.copy()
            hp_to_modify = np.random.choice(self.hps)
            new_config[hp_to_modify.name] = hp_to_modify.sample_neighbors(config[hp_to_modify.name])[0]
            neighbors.append(new_config)
        return neighbors

    def sample_all_neighbors(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all neighbors, instead of randomly switching, get neighbors from each param
        reverse for bool, two adjacent for ordinal, two random for categorical etc..
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
                # Handle the case where no neighbors are available
                logging.warning(f"No neighbors available for hyperparameter {hp.name}: {e}")
        return neighbors

    def __str__(self):
        return f"HyperParameterSpace: {self.name}\n" + "\n".join(str(hp) for hp in self.hps)
