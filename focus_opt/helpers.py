import csv
import os
import time
import uuid

class OutOfBudgetError(Exception):
    """Exception raised when the budget is exceeded."""
    pass

class SessionContext:
    """Manages the session context for optimization processes, including budget and logging.

    :param budget: The total budget available for the session.
    :type budget: int
    :param start_time: The start time of the session, defaults to the current time.
    :type start_time: int, optional
    :param total_cost: The initial total cost, defaults to 0.
    :type total_cost: int, optional
    :param max_time: The maximum allowed time for the session, defaults to None.
    :type max_time: int, optional
    :param cost_increment: The cost increment for each operation, defaults to 1.
    :type cost_increment: int, optional
    :param log_dir: The directory where logs are stored, defaults to "logs/".
    :type log_dir: str, optional
    :param log_results: Whether to log results to a file, defaults to False.
    :type log_results: bool, optional
    """

    def __init__(self, budget: int, start_time: int = None, total_cost: int = 0, max_time: int = None, cost_increment: int = 1, log_dir: str = "logs/", log_results: bool = False):
        """Constructor method
        """
        self.start_time = start_time if start_time else time.time()
        self.total_cost = total_cost
        self.budget = budget
        self.max_time = max_time
        self.cost_increment = cost_increment
        self.log_results = log_results

        if log_results:
            self.run_id = str(uuid.uuid4())[:8]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file = os.path.join(log_dir, f"run_log_{timestamp}.csv")

            os.makedirs(log_dir, exist_ok=True)

            if not os.path.exists(self.log_file):
                with open(self.log_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Run ID", "Budget Used", "Best Score"])

    def time_out_check(self) -> bool:
        """Checks if the session has exceeded the maximum allowed time.

        :return: `True` if the session has timed out, `False` otherwise.
        :rtype: bool
        """
        return (self.max_time and (time.time() - self.start_time) > self.max_time)

    def out_of_budget_check(self) -> bool:
        """Checks if the session has exceeded the budget.

        :return: `True` if the budget is exceeded, `False` otherwise.
        :rtype: bool
        """
        return (self.total_cost + self.cost_increment) > self.budget

    def budget_error_checks(self):
        """Performs checks for timeouts and budget exceedance.

        :raises TimeoutError: If the session has timed out.
        :raises OutOfBudgetError: If the budget is exceeded.
        """
        if self.time_out_check():
            raise TimeoutError("Timeout reached")
        if self.out_of_budget_check():
            raise OutOfBudgetError("Budget exceeded")

    def can_continue_running(self) -> bool:
        """Determines if the session can continue running.

        :return: `True` if the session can continue, `False` otherwise.
        :rtype: bool
        """
        return not (self.time_out_check() or self.out_of_budget_check())

    def increment_total_cost(self):
        """Increments the total cost by the cost increment."""
        self.total_cost += self.cost_increment

    def log_performance(self, best_score: float):
        """Logs the performance of the session to a file.

        :param best_score: The best score achieved in the session.
        :type best_score: float
        """
        if self.log_results:
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.run_id, self.total_cost, best_score])
