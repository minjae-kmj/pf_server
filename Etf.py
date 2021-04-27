import csv
from time import sleep
from collections import defaultdict
from EffecientFrontier import EfficientFrontierCalculator
from Ticker import ticker


class Etf:
    def __init__(self):
        self.path = "./dat"
        self.composition_dict = None
        self.reverse_dict = None
        self.get_composition_info()

    def write_file(self):
        from pykrx import stock
        with open(f'{self.path}/etf.csv', 'w', encoding='euc-kr') as f:
            f.writelines(f'Code,Name,Comp. Code,Amount,Cash,Percent\n')
            tickers = stock.get_etf_ticker_list()
            for idx, code in enumerate(tickers):
                name = stock.get_etf_ticker_name(code)
                df = stock.get_etf_portfolio_deposit_file(code)
                print(f'[{code}] {name} -> {idx}')
                if df.iloc[0, 2] == 0:
                    continue
                for row in df.itertuples():
                    f.writelines(f'{code},{name},{row[0]},{row[1]},{row[2]},{row[3]}\n')
                sleep(0.3)

    def get_composition_info(self):
        self.composition_dict = defaultdict(list)
        self.reverse_dict = defaultdict(list)
        with open(f'{self.path}/etf.csv', encoding='euc-kr') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.composition_dict[row[0]].append({
                    "code": row[2],
                    "name": ticker.get_name(row[2]),
                    "amount": row[3],
                    "cash": row[4],
                    "percent": row[5]
                })
                self.reverse_dict[row[2]].append({
                    "parent": row[0],
                    "amount": row[3],
                    "cash": row[4],
                    "percent": row[5]
                })

    def calc_match_score(self, codes, limit_count=5):
        etf_dic = {}
        for code in codes:
            for info in self.reverse_dict[code]:
                if info["parent"] in etf_dic:
                    pct, cnt = etf_dic[info["parent"]]
                else:
                    pct, cnt = 0, 0
                etf_dic[info["parent"]] = [pct + float(info["percent"]), cnt + 1]

        match_etf = sorted(etf_dic.items(), key=lambda x: x[1][0], reverse=True)[:limit_count]
        results = []
        for code, comp in match_etf:
            ef = EfficientFrontierCalculator([code])
            perform = ef.get_maximum_sharpe()

            results.append({
                "code": code,
                "name": ticker.get_name(code),
                "match_weight": comp[0],
                "match_count": comp[1],
                "returns": perform["returns"],
                "risk": perform["risk"],
                "info": self.composition_dict[code]
            })
        return results


etf = Etf()

if __name__ == "__main__":
    etf.calc_match_score(["005930", "005380"])
