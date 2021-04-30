import yfinance as yf
import pandas as pd
import numpy as np
from Ticker import ticker
from EffecientFrontier import START_DATE


class BackTester:
    @staticmethod
    def get_plot(codes: list, weights: list):
        names = [ticker.get_name(code) for code in codes]
        df = yf.download([code + ".ks" for code in codes], start=START_DATE)['Adj Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=names[0])
        df = df.rename(columns={code + ".KS": name for code, name in zip(codes, names)})
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').dropna()

        ror = df / df.shift(1)
        ror.iloc[0] = [1] * len(ror.columns)

        for col, weight in zip(ror.columns, weights):
            ror[col] = ror[col] * weight
        ror_rslt = ror.sum(axis=1)
        cump = ror_rslt.cumprod()
        results = {
            "days": [],
            "values": []
        }

        for index, value in cump.items():
            results["days"].append(index.strftime("%Y-%m-%d"))
            results["values"].append(value)

        return results
