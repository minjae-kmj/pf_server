import csv


class Ticker:
    def __init__(self):
        self.path = "./dat"
        self.name_dic = {}

        with open(f'{self.path}/all.csv', encoding='euc-kr') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    self.name_dic[row[0]] = row[1]

    def write_file(self, market: str):
        from pykrx import stock
        tickers = stock.get_market_ticker_list(market="ALL")
        with open(f'{self.path}/{market.lower()}.csv', 'w', encoding='euc-kr') as f:
            for code in tickers:
                f.writelines(f'{code},{stock.get_market_ticker_name(code)}\n')
            if market == "ALL":
                tickers = stock.get_etf_ticker_list()
                for code in tickers:
                    f.writelines(f'{code},{stock.get_etf_ticker_name(code)}\n')

    def get_ticker_list(self, market: str):
        results = []
        with open(f'{self.path}/{market.lower()}.csv', encoding='euc-kr') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    results.append({
                        "code": row[0],
                        "name": row[1]
                    })
        return results

    def get_name(self, code):
        if code in self.name_dic:
            return self.name_dic[code]
        else:
            return code


ticker = Ticker()

if __name__ == "__main__":
    ticker.write_file(market="ALL")
