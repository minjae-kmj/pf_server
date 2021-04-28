"""
The ``efficient_SemiAbsoluteDeviation`` submodule houses the EfficientSemiAbsoluteDeviation class, which
generates portfolios along the mean-SemiAbsoluteDeviation frontier.
"""


import numpy as np
import cvxpy as cp
import pandas as pd

from .. import objective_functions
from .efficient_frontier import EfficientFrontier


class EfficientSemiAbsoluteDeviation(EfficientFrontier):
    """
    EfficientSemiAbsoluteDeviation objects allow for optimization along the mean-SemiAbsoluteDeviation frontier.
    This may be relevant for users who are more concerned about downside deviation.

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``returns`` - pd.DataFrame
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str
        - ``solver_options`` - {str: str} dict


    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``min_SemiAbsoluteDeviation()`` minimises the portfolio SemiAbsoluteDeviation (downside deviation)
    - ``max_quadratic_utility()`` maximises the "downside quadratic utility", given some risk aversion.
    - ``efficient_risk()`` maximises return for a given target semiDeviation
    - ``efficient_return()`` minimises semiDeviation for a given target return
    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints

    - ``portfolio_performance()`` calculates the expected return, semiDeviation and Sortino ratio for
      the optimized portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(
        self,
        close_returns,
        predicted_close_returns,
        frequency=5,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        :param expected_returns: expected returns for each asset. Can be None if
                                optimising for semiDeviation only.
        :type expected_returns: pd.Series, list, np.ndarray
        :param close_returns: (historic) returns for all your assets (no NaNs).
                                 See ``expected_returns.returns_from_prices``.
        :type close_returns: pd.DataFrame or np.array
        :param frequency: number of time periods in a year, defaults to 252 (the number
                          of trading days in a year). This must agree with the frequency
                          parameter used in your ``expected_returns``.
        :type frequency: int, optional
        :param benchmark: the return threshold to distinguish "downside" and "upside".
                          This should match the frequency of your ``returns``,
                          i.e this should be a benchmark daily returns if your
                          ``returns`` are also daily.
        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair
                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)
                              for portfolios with shorting.
        :type weight_bounds: tuple OR tuple list, optional
        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`
        :type solver: str
        :param verbose: whether performance and debugging info should be printed, defaults to False
        :type verbose: bool, optional
        :param solver_options: parameters for the given solver
        :type solver_options: dict, optional
        :raises TypeError: if ``expected_returns`` is not a series, list or array
        """
        # Instantiate parent
        super().__init__(
            expected_returns=close_returns,
            cov_matrix=np.zeros((len(close_returns),) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        # self.close_returns = self._validate_returns(close_returns)
        self.close_returns = pd.DataFrame(close_returns)
        # self.predicted_returns = self._validate_returns(predicted_close_returns)
        self.predicted_close_returns = pd.DataFrame(predicted_close_returns)
        self.frequency = frequency
        self._T = self.close_returns.shape[0]

    def min_volatility(self):
        raise NotImplementedError("Please use min_semi_absolute_deviation instead.")

    def max_sharpe(self, risk_free_rate=0.02):
        raise NotImplementedError("Method not available in EfficientSemiAbsoluteDeviation")

    def min_semi_absolute_deviation(self, market_neutral=False):
        """
        Minimise portfolio SemiAbsoluteDeviation (see docs for further explanation).

        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :return: asset weights for the volatility-minimising portfolio
        :rtype: OrderedDict
        """
        predict_error = cp.sum((self.close_returns.values - self.predicted_close_returns.values) @ self._w)
        objective_func = cp.sum((cp.abs(predict_error) + predict_error) / (2 * self._T))
        self._objective = objective_func

        for obj in self._additional_objectives:
            self._objective += obj

        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= 0)
        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= cp.sum(predict_error))
        self._constraints.append(cp.sum(self.close_returns.values @ self._w) >= 0.001)
        self._constraints.append(cp.sum(self._w) == 1)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        """
        Maximise the given quadratic utility, using portfolio SemiAbsoluteDeviation instead
        of variance.

        :param risk_aversion: risk aversion parameter (must be greater than 0),
                              defaults to 1
        :type risk_aversion: positive float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :return: asset weights for the maximum-utility portfolio
        :rtype: OrderedDict
        """
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        predict_error = cp.sum((self.close_returns.values - self.predicted_close_returns.values) @ self._w)
        objective_func = cp.sum((cp.abs(predict_error) + predict_error) / (2 * self._T))
        # mu = objective_functions.portfolio_return(self._w, self.close_returns.values)
        mu = objective_functions.portfolio_return(
            self._w, np.array(self.close_returns.mean(axis=0)).T, negative=False
        )
        self._objective = mu + 0.5 * risk_aversion * objective_func

        for obj in self._additional_objectives:
            self._objective += obj

        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= 0)
        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= cp.sum(predict_error))
        self._constraints.append(cp.sum(self.close_returns.values @ self._w) >= 0.001)
        self._constraints.append(cp.sum(self._w) == 1)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_semi_deviation, market_neutral=False):
        """
        Maximise return for a target semiDeviation (downside standard deviation).
        The resulting portfolio will have a semiDeviation less than the target
        (but not guaranteed to be equal).

        :param target_semi_deviation: the desired maximum semiDeviation of the resulting portfolio.
        :type target_semi_deviation: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :return: asset weights for the efficient risk portfolio
        :rtype: OrderedDict
        """
        self._objective = objective_functions.portfolio_return(
            self._w, np.array(self.close_returns.mean(axis=0)).T, negative=False
        )
        print(self._objective)
        for obj in self._additional_objectives:
            self._objective += obj

        predict_error = cp.sum((self.close_returns.values - self.predicted_close_returns.values) @ self._w)
        objective_func = cp.sum((cp.abs(predict_error) + predict_error) / (2 * self._T))

        # ?? 왜 이것만 에러지
        self._constraints.append(self.frequency * objective_func <= (target_semi_deviation ** 2))

        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= 0)
        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= cp.sum(predict_error))
        self._constraints.append(cp.sum(self.close_returns.values @ self._w) >= 0.001)
        self._constraints.append(cp.sum(self._w) == 1)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """
        Minimise semiDeviation for a given target return.

        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :raises ValueError: if no portfolio can be found with return equal to ``target_return``
        :return: asset weights for the optimal portfolio
        :rtype: OrderedDict
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if target_return > np.abs(self.close_returns.values).max():
            raise ValueError(
                "target_return must be lower than the largest expected return"
            )

        predict_error = cp.sum((self.close_returns.values - self.predicted_close_returns.values) @ self._w)
        objective_func = cp.sum((cp.abs(predict_error) + predict_error) / (2 * self._T))
        self._objective = objective_func

        for obj in self._additional_objectives:
            self._objective += obj

        self._constraints.append(cp.sum(self.close_returns.values @ self._w) >= target_return)

        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= 0)
        # self._constraints.append((cp.abs(predict_error) + predict_error) / 2 >= cp.sum(predict_error))
        self._constraints.append(cp.sum(self.close_returns.values @ self._w) >= 0.001)
        self._constraints.append(cp.sum(self._w) == 1)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio, specifically: expected return, semiDeviation, Sortino ratio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                               The period of the risk-free rate should correspond to the
                               frequency of expected returns.
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, semiDeviation, Sortino ratio.
        :rtype: (float, float, float)
        """
        print(np.array(self.close_returns.mean(axis=0)))
        mu = objective_functions.portfolio_return(
            self.weights, np.array(self.close_returns.mean(axis=0)).T, negative=False
        )
        portfolio_returns = self.close_returns @ self.weights
        drops = np.fmin(portfolio_returns, 0)
        semi_absolute_deviation = np.sum(np.square(drops)) / self._T / self.frequency
        semi_deviation = np.sqrt(semi_absolute_deviation)
        sortino_ratio = (mu - risk_free_rate) / semi_deviation

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual semi-deviation: {:.1f}%".format(100 * semi_deviation))
            print("Sortino Ratio: {:.2f}".format(sortino_ratio))

        return mu, semi_deviation, sortino_ratio
