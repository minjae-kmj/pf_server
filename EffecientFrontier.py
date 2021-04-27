import numpy as np
from Ticker import ticker
import yfinance as yf
import math

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.efficient_frontier import EfficientSemivariance
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd


class EfficientFrontierCalculator:
    def __init__(self, codes):
        self.codes = codes
        self.names = [ticker.get_name(code) for code in codes]
        df = yf.download([code + ".ks" for code in codes], start='2011-01-01')['Adj Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.names[0])
        df = df.rename(columns={code + ".KS": name for code, name in zip(codes, self.names)})

        self.mu = expected_returns.mean_historical_return(df)
        self.S = risk_models.sample_cov(df)
        self.latest_prices = get_latest_prices(df)

    def get_weights_object(self, weight_dic: dict):
        items = []
        values = []
        for item, value in weight_dic.items():
            items.append(item)
            values.append(value)
        return {
            "items": items,
            "values": values
        }

    def get_maximum_sharpe(self):
        ef = EfficientFrontier(self.mu, self.S)
        # ef = EfficientSemivariance(self.mu, self.S)
        weights = ef.max_sharpe()
        # weights = ef.max_quadratic_utility()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_minimum_risk(self):
        ef = EfficientFrontier(self.mu, self.S)
        # ef = EfficientSemivariance(self.mu, self.S)
        weights = ef.min_volatility()
        # weights = ef.min_semivariance()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_maximum_return(self, target_volatility=100):
        ef = EfficientFrontier(self.mu, self.S)
        # ef = EfficientSemivariance(self.mu, self.S)
        weights = ef.efficient_risk(target_volatility=target_volatility)
        # weights = ef.efficient_risk(target_semideviation=target_volatility)
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_frontier(self):
        min_vol = self.get_minimum_risk()["risk"]
        max_vol = max(self.get_maximum_sharpe()["risk"], self.get_maximum_return()["risk"])

        min_vol = math.ceil(min_vol * 1000) / 1000
        max_vol = math.ceil(max_vol * 1000) / 1000

        rslt = []
        for vol in np.arange(min_vol, max_vol, 0.001):
            rslt.append(self.get_maximum_return(target_volatility=vol))

        return rslt

    def get_performance_by_weight(self, weights):
        ef = EfficientFrontier(self.mu, self.S)
        # ef = EfficientSemivariance(self.mu, self.S)
        weights_dic = {name: weight for name, weight in zip(self.names, weights)}
        ef.set_weights(weights_dic)
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(weights_dic)
        }

    def get_stock_amount(self, weights, cash=10000000):
        weights_dic = {name: weight for name, weight in zip(self.names, weights)}
        da = DiscreteAllocation(weights_dic, self.latest_prices, total_portfolio_value=cash)

        allocation, leftover = da.lp_portfolio()
        print(f"Discrete allocation: {allocation}")
        print(f"Funds remaining: {leftover:.2f}")

        return allocation, leftover
