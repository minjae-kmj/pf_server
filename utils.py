import csv
from time import sleep
from collections import defaultdict
from EffecientFrontier import EfficientFrontierCalculator
from Ticker import ticker
import yfinance as yf


def get_history_yields(names, proportions):
    samsung = yf.Ticker("aapl")
    # print(samsung.dividends)
    print(samsung.info)
    print(samsung.get_dividends())

