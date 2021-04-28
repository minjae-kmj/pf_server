import datetime

import numpy as np

from PyPortfolioOpt.pypfopt.efficient_frontier.efficient_future_semi_absolute_deviation import \
    EfficientSemiAbsoluteDeviation
from Ticker import ticker
import yfinance as yf
import math

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.efficient_frontier import EfficientSemivariance
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import pandas as pd
from tensorflow import keras

START_DATE = '2011-01-01'


class EfficientFrontierCalculator:
    def __init__(self, codes):
        self.codes = codes
        self.names = [ticker.get_name(code) for code in codes]
        df = yf.download([code + ".ks" for code in codes], start=START_DATE)['Adj Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.names[0])
        df = df.rename(columns={code + ".KS": name for code, name in zip(codes, self.names)})

        self.mu = expected_returns.mean_historical_return(df)
        self.S = risk_models.sample_cov(df)
        self.latest_prices = get_latest_prices(df)

    @staticmethod
    def get_weights_object(weight_dic: dict):
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
        weights = ef.max_sharpe()
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
        weights = ef.min_volatility()
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
        weights = ef.efficient_risk(target_volatility=target_volatility)
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
            try:
                rslt.append(self.get_maximum_return(target_volatility=vol))
            except Exception as e:
                print(f'!!!Error!!! Target Vol: {vol}, Min Vol: {min_vol}')
                print(e)
        return rslt

    def get_performance_by_weight(self, weights):
        ef = EfficientFrontier(self.mu, self.S)
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


class EfficientFrontierSemiVarianceCalculator:
    def __init__(self, codes):
        self.codes = codes
        self.names = [ticker.get_name(code) for code in codes]
        df = yf.download([code + ".ks" for code in codes], start=START_DATE)['Adj Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.names[0])
        df = df.rename(columns={code + ".KS": name for code, name in zip(codes, self.names)})
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()

        self.mu = expected_returns.mean_historical_return(df)
        self.historical_returns = risk_models.returns_from_prices(df)
        self.latest_prices = get_latest_prices(df)

    @staticmethod
    def get_weights_object(weight_dic: dict):
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
        ef = EfficientSemivariance(self.mu, self.historical_returns)
        weights = ef.max_quadratic_utility()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_minimum_risk(self):
        ef = EfficientSemivariance(self.mu, self.historical_returns)
        weights = ef.min_semivariance()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_maximum_return(self, target_volatility=100):
        ef = EfficientSemivariance(self.mu, self.historical_returns)
        weights = ef.efficient_risk(target_semideviation=target_volatility)
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
            try:
                rslt.append(self.get_maximum_return(target_volatility=vol))
            except Exception as e:
                print(f'!!!Error!!! Target Vol: {vol}, Min Vol: {min_vol}')
                print(e)

        return rslt

    def get_performance_by_weight(self, weights):
        ef = EfficientSemivariance(self.mu, self.historical_returns)
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


class EfficientFrontierSemiAbsoluteCalculator:
    def __init__(self, codes, predict_period, absolute_error_period):
        X_train = {}
        self.codes = codes
        self.names = [ticker.get_name(code) for code in codes]
        self.predict_period = predict_period
        self.absolute_error_period = absolute_error_period

        # Load model & predict returns
        model = keras.models.load_model('dnn_model_{0}'.format(predict_period))

        for code in codes:
            df = yf.download(code + ".ks", start=START_DATE)
            df = df.rename(columns={code + ".KS": name for code, name in zip(self.codes, self.names)})
            df = expected_returns.returns_from_prices(df)
            df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()

            agg_data = []
            columns = df.columns
            for index, row in df.iterrows():
                tmp_data = []
                for column in columns:
                    tmp_data.append(round(row[column], 5))
                agg_data.append(tmp_data)
            X_train[ticker.get_name(code)] = np.array(agg_data).astype('float32')

        df = yf.download([code + ".ks" for code in codes], start=START_DATE)
        df = df.rename(columns={code + ".KS": name for code, name in zip(self.codes, self.names)})

        pct_df = expected_returns.returns_from_prices(df)
        pct_df = pct_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()

        total_data_count = pct_df.shape[0]
        real_returns = pct_df['Adj Close'].iloc[total_data_count - absolute_error_period:, :]

        # if mu is calculated based on "specified period"
        self.mu = expected_returns.mean_historical_return(df['Adj Close'].iloc[total_data_count - absolute_error_period:, :])
        # if mu : 1 year
        self.mu = expected_returns.mean_historical_return(df['Adj Close'])
        predicted_returns = real_returns.copy()

        for name in X_train:
            agg_data = []
            for i in range(absolute_error_period):
                tmp_data = []
                for j in range(len(X_train[name]) - predict_period - absolute_error_period + i + 1,
                               len(X_train[name]) - absolute_error_period + i + 1):
                    tmp_data.append(X_train[name][j])
                agg_data.append(np.array(tmp_data).flatten())
            predicted_returns[name] = model.predict(np.array(agg_data)).flatten() / 10

        self.real_returns = real_returns
        self.predicted_returns = predicted_returns

    @staticmethod
    def get_weights_object(weight_dic: dict):
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
        ef = EfficientSemiAbsoluteDeviation(self.mu, self.real_returns, self.predicted_returns)
        weights = ef.max_quadratic_utility()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_minimum_risk(self):
        ef = EfficientSemiAbsoluteDeviation(self.mu, self.real_returns, self.predicted_returns)
        weights = ef.min_semi_absolute_deviation()
        cleaned_weights = ef.clean_weights()
        rt, vol, shp = ef.portfolio_performance(verbose=False)
        return {
            "returns": rt,
            "risk": vol,
            "sharpe": shp,
            "weights": self.get_weights_object(cleaned_weights)
        }

    def get_maximum_return(self, target_volatility=100):
        ef = EfficientSemiAbsoluteDeviation(self.mu, self.real_returns, self.predicted_returns)
        weights = ef.efficient_risk(target_semi_deviation=target_volatility)
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
            try:
                rslt.append(self.get_maximum_return(target_volatility=vol))
            except Exception as e:
                print(f'!!!Error!!! Target Vol: {vol}, Min Vol: {min_vol}')
                print(e)
        return rslt

    def get_performance_by_weight(self, weights):
        ef = EfficientSemiAbsoluteDeviation(self.mu, self.real_returns, self.predicted_returns)
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