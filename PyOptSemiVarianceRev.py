from pypfopt.efficient_frontier import EfficientSemivariance
from pypfopt import objective_functions
import numpy as np
import cvxpy as cp


class EfficientSemivarianceRev(EfficientSemivariance):
    def __init__(self, expected_returns, returns):
        super().__init__(expected_returns, returns)

    def worst_risk(self, target_semideviation, market_neutral=False):
        self._objective = objective_functions.portfolio_return(
            self._w, -self.expected_returns
        )
        for obj in self._additional_objectives:
            self._objective += obj

        p = cp.Variable(self._T, nonneg=True)
        n = cp.Variable(self._T, nonneg=True)

        self._constraints.append(
            self.frequency * cp.sum(cp.square(n)) <= (target_semideviation ** 2)
        )
        B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
        self._constraints.append(B @ self._w - p + n == 0)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()
