import csv
import os
import time
import uuid
class OutOfBudgetError(Exception):
    pass

class SessionContext:
    def __init__(self, budget: int, start_time: int = None, total_cost: int = 0, max_time: int = None, cost_increment: int = 1, log_dir: str = "logs/", log_results: bool = False):
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

    def time_out_check(self):
        return (self.max_time and (time.time()-self.start_time) > self.max_time)

    def out_of_budget_check(self):
        return (self.total_cost + self.cost_increment) > self.budget

    def budget_error_checks(self):
        if self.time_out_check():
            raise TimeoutError("Timeout reached")
        if self.out_of_budget_check():
            raise OutOfBudgetError("Budget exceeded")

    def can_continue_running(self):
        return not (self.time_out_check() or self.out_of_budget_check())

    def increment_total_cost(self):
        self.total_cost += self.cost_increment

    def log_performance(self, best_score):
        if self.log_results:
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.run_id, self.total_cost, best_score])
