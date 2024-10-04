# **focus-opt**: Multi-Fidelity Hyperparameter Optimization

[![PyPI version](https://badge.fury.io/py/focus-opt.svg)](https://badge.fury.io/py/focus-opt)
[![Python versions](https://img.shields.io/pypi/pyversions/focus-opt.svg)](https://pypi.org/project/focus-opt/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)
[![Downloads](https://pepy.tech/badge/focus-opt)](https://pepy.tech/project/focus-opt)
[![Documentation Status](https://readthedocs.org/projects/focus-opt/badge/?version=latest)](https://focus-opt.readthedocs.io/en/latest/?badge=latest)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

`focus-opt` is a Python package for performing **multi-fidelity hyperparameter optimization** on machine learning models. It implements optimization algorithms such as **Hill Climbing** and **Genetic Algorithms** with support for multi-fidelity evaluations. This allows for efficient exploration of hyperparameter spaces by evaluating configurations at varying levels of fidelity, balancing computational cost and optimization accuracy.

The package is designed to be flexible and extensible, enabling users to define custom hyperparameter spaces and evaluation functions for different machine learning models. In this guide, we'll demonstrate how to install and use `focus-opt` with a **Decision Tree Classifier** on the **Breast Cancer Wisconsin** dataset.

## Installation

You can install `focus-opt` directly from PyPI:

```bash
pip install focus-opt
```

Alternatively, if you want to work with the latest version from the repository:

```bash
git clone https://github.com/eliottkalfon/focus_opt.git
cd focus_opt
pip install .
```

It's recommended to use a virtual environment to manage dependencies.

## Creating a Virtual Environment (Optional)

Create a virtual environment using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On Unix or Linux:**

  ```bash
  source venv/bin/activate
  ```

## Using `focus-opt` with a Decision Tree Classifier

Below is an example of how to use `focus-opt` to perform hyperparameter optimization on a Decision Tree Classifier using both Hill Climbing and Genetic Algorithms.

```python
import logging
from typing import Dict, Any

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Import classes from focus_opt package
from focus_opt.hp_space import (
    HyperParameterSpace,
    CategoricalHyperParameter,
    OrdinalHyperParameter,
    ContinuousHyperParameter,
)

from focus_opt.optimizers import HillClimbingOptimizer, GeneticAlgorithmOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the hyperparameter space for the Decision Tree Classifier
hp_space = HyperParameterSpace("Decision Tree Hyperparameter Space")

hp_space.add_hp(CategoricalHyperParameter(name="criterion", values=["gini", "entropy"]))
hp_space.add_hp(CategoricalHyperParameter(name="splitter", values=["best", "random"]))
hp_space.add_hp(
    OrdinalHyperParameter(name="max_depth", values=[None] + list(range(1, 21)))
)
hp_space.add_hp(
    ContinuousHyperParameter(
        name="min_samples_split", min_value=2, max_value=20, is_int=True
    )
)
hp_space.add_hp(
    ContinuousHyperParameter(
        name="min_samples_leaf", min_value=1, max_value=20, is_int=True
    )
)
hp_space.add_hp(
    ContinuousHyperParameter(name="max_features", min_value=0.0, max_value=1.0)
)

# Load the Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define the evaluation function
def dt_evaluation(config: Dict[str, Any], fidelity: int) -> float:
    """
    Evaluation function for a Decision Tree Classifier with cross-validation.

    Args:
        config (Dict[str, Any]): Hyperparameter configuration.
        fidelity (int): Fidelity level (index of the cross-validation fold).

    Returns:
        float: Accuracy for the specified cross-validation fold.
    """
    logging.info(f"Evaluating config: {config} at fidelity level: {fidelity}")

    # Initialize the classifier with the given hyperparameters
    clf = DecisionTreeClassifier(random_state=42, **config)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get the train and test indices for the specified fold
    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y)):
        if fold_index + 1 == fidelity:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            logging.info(f"Score for config {config} at fold {fidelity}: {score}")
            return score

    raise ValueError(f"Invalid fidelity level: {fidelity}")

# Instantiate the Hill Climbing Optimizer
hill_climbing_optimizer = HillClimbingOptimizer(
    hp_space=hp_space,
    evaluation_function=dt_evaluation,
    max_fidelity=5,       # Number of cross-validation folds
    maximize=True,        # We aim to maximize accuracy
    log_results=True,
    warm_start=20,        # Number of initial configurations to explore
    random_restarts=5,    # Number of random restarts to avoid local optima
)

# Run the Hill Climbing optimization
best_candidate_hill_climbing = hill_climbing_optimizer.optimize(budget=500)
print(
    f"Best candidate from Hill Climbing: {best_candidate_hill_climbing.config} "
    f"with score: {best_candidate_hill_climbing.evaluation_score}"
)

# Instantiate the Genetic Algorithm Optimizer
ga_optimizer = GeneticAlgorithmOptimizer(
    hp_space=hp_space,
    evaluation_function=dt_evaluation,
    max_fidelity=5,           # Number of cross-validation folds
    maximize=True,            # We aim to maximize accuracy
    population_size=20,       # Size of the population in each generation
    crossover_rate=0.8,       # Probability of crossover between parents
    mutation_rate=0.1,        # Probability of mutation in offspring
    elitism=1,                # Number of top individuals to carry over to the next generation
    tournament_size=3,        # Number of individuals competing in tournament selection
    min_population_size=5,    # Minimum population size to maintain diversity
    log_results=True,
)

# Run the Genetic Algorithm optimization
best_candidate_ga = ga_optimizer.optimize(budget=500)
print(
    f"Best candidate from Genetic Algorithm: {best_candidate_ga.config} "
    f"with score: {best_candidate_ga.evaluation_score}"
)
```

### Explanation

- **Importing `focus_opt`**: We import the necessary classes from the `focus_opt` package.
- **Hyperparameter Space Definition**: We define a hyperparameter space that includes parameters such as `criterion`, `splitter`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
- **Evaluation Function**: The `dt_evaluation` function evaluates a given hyperparameter configuration using cross-validation. The `fidelity` parameter corresponds to the cross-validation fold index, enabling multi-fidelity optimization.
- **Optimizers**: We use both `HillClimbingOptimizer` and `GeneticAlgorithmOptimizer` from `focus_opt.optimizers` to search for the best hyperparameter configuration within the defined budget.
- **Running the Optimization**: We specify a computational budget (e.g., `budget=500`), which limits the total number of evaluations performed during the optimization process.

## Usage Guide

### Defining Hyperparameter Spaces

`focus_opt` allows you to define a hyperparameter space by creating instances of different hyperparameter types:

- **CategoricalHyperParameter**: For hyperparameters that take on a set of discrete categories.
- **OrdinalHyperParameter**: For hyperparameters that have an inherent order.
- **ContinuousHyperParameter**: For hyperparameters with continuous values, including integers and floats.

Example:

```python
from focus_opt.hp_space import (
    HyperParameterSpace,
    CategoricalHyperParameter,
    ContinuousHyperParameter
)

hp_space = HyperParameterSpace("Model Hyperparameters")

hp_space.add_hp(
    CategoricalHyperParameter(
        name="activation_function", 
        values=["relu", "tanh", "sigmoid"]
    )
)

hp_space.add_hp(
    ContinuousHyperParameter(
        name="learning_rate", 
        min_value=0.0001, 
        max_value=0.1
    )
)
```

### Implementing Custom Evaluation Functions

Your evaluation function should accept a hyperparameter configuration and a fidelity level, then return a performance score. Here's a template:

```python
from typing import Dict, Any

def evaluation_function(config: Dict[str, Any], fidelity: int) -> float:
    """
    Custom evaluation function.

    Args:
        config (Dict[str, Any]): Hyperparameter configuration.
        fidelity (int): Fidelity level (e.g., amount of data or number of epochs).

    Returns:
        float: Performance score.
    """
    # Implement your model training and evaluation logic here
    pass
```

### Using the Hill Climbing Optimizer

```python
from focus_opt.optimizers import HillClimbingOptimizer

optimizer = HillClimbingOptimizer(
    hp_space=hp_space,
    evaluation_function=evaluation_function,
    max_fidelity=10,       # Adjust based on your fidelity levels
    maximize=True,         # Set to False if minimizing
    log_results=True,
    warm_start=10,         # Initial random configurations
    random_restarts=3,     # Number of restarts to avoid local optima
)

best_candidate = optimizer.optimize(budget=100)
print(f"Best configuration: {best_candidate.config}")
print(f"Best score: {best_candidate.evaluation_score}")
```

### Using the Genetic Algorithm Optimizer

```python
from focus_opt.optimizers import GeneticAlgorithmOptimizer

optimizer = GeneticAlgorithmOptimizer(
    hp_space=hp_space,
    evaluation_function=evaluation_function,
    max_fidelity=10,
    maximize=True,
    population_size=50,
    crossover_rate=0.7,
    mutation_rate=0.1,
    elitism=2,
    tournament_size=5,
    log_results=True,
)

best_candidate = optimizer.optimize(budget=500)
print(f"Best configuration: {best_candidate.config}")
print(f"Best score: {best_candidate.evaluation_score}")
```

### Customizing the Optimizers

You can adjust various parameters of the optimizers to suit your needs:

- **For `HillClimbingOptimizer`**:
  - `warm_start`: Number of random initial configurations.
  - `random_restarts`: Number of times the optimizer restarts from a new random position.
  - `neighbor_selection`: Strategy for selecting neighboring configurations.

- **For `GeneticAlgorithmOptimizer`**:
  - `population_size`: Number of configurations in each generation.
  - `crossover_rate`: Probability of crossover between parent configurations.
  - `mutation_rate`: Probability of mutation in offspring configurations.
  - `elitism`: Number of top configurations to carry over to the next generation.
  - `tournament_size`: Number of configurations competing during selection.

### Multi-Fidelity Optimization

`focus_opt` enables multi-fidelity optimization by allowing you to specify varying levels of fidelity in your evaluation function. This can help reduce computational costs by evaluating more configurations at lower fidelities and fewer configurations at higher fidelities.

#### Fidelity Scheduling

You can define your own fidelity scheduling within your evaluation function or rely on the built-in mechanisms:

```python
def evaluation_function(config: Dict[str, Any], fidelity: int) -> float:
    # Use 'fidelity' to adjust the evaluation, such as training epochs or data size
    pass
```

## Requirements

Ensure you have the following packages installed:

- **Python**: 3.8 or higher
- **numpy**: 1.26

These dependencies are automatically installed when you install `focus_opt` using `pip`.

## Contributing

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

To contribute:

1. **Fork** the repository.
2. **Create** a new branch (`git checkout -b feature/YourFeature`).
3. **Commit** your changes (`git commit -am 'Add YourFeature'`).
4. **Push** to the branch (`git push origin feature/YourFeature`).
5. **Open** a Pull Request.

Please ensure your code adheres to the existing style standards and includes appropriate tests.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://opensource.org/license/apache-2-0) for details.

---

**Author**: Eliott Kalfon

Feel free to reach out if you have any questions or need further assistance!

---

**Additional Notes**

- **Documentation**: Comprehensive documentation is available on [Read the Docs](https://focus-opt.readthedocs.io/en/latest/).
- **Continuous Integration**: The project uses GitHub Actions for automated testing and code quality checks.
- **Code Style**: The codebase follows the [Black](https://github.com/psf/black) code style for consistency.
