from flask import Flask, request, jsonify
from EffecientFrontier import EfficientFrontierSemiVarianceCalculator as EfficientFrontierCalculator, \
    EfficientFrontierSemiVarianceCalculator
from EffecientFrontier import EfficientFrontierSemiAbsoluteCalculator
from Ticker import ticker
import FinanceDataReader as fdr
from Etf import etf

app = Flask(__name__)


@app.route("/")
def home():
    return "home"


@app.route("/stocks", methods=["GET"])
def get_stock_list():
    market = request.args.get("market")
    if market is None:
        market = "KOSPI"
    return jsonify(ticker.get_ticker_list(market=market))


@app.route("/stocks/<code>", methods=["GET"])
def get_stock_info(code):
    row = fdr.DataReader(code, start="2021-04-10").iloc[-1]
    return jsonify({
        "code": code,
        "name": ticker.get_name(code),
        "open": row["Open"],
        "high": row["High"],
        "low": row["Low"],
        "close": row["Close"],
        "volume": row["Volume"],
        "change": row["Change"]
    })


@app.route("/frontier", methods=["POST"])
def calc_efficient_frontier():
    inputs = request.get_json()
    if inputs["mode"] == "original":
        ef = EfficientFrontierCalculator(inputs["codes"])
    elif inputs["mode"] == "semi_variance":
        ef = EfficientFrontierSemiVarianceCalculator(inputs["codes"])
    elif inputs["mode"] == "semi_absolute":
        ef = EfficientFrontierSemiAbsoluteCalculator(inputs["codes"], inputs["predict_period"], inputs["absolute_error_period"])
    else:
        return "please check your input: mode", 400

    results = {
        "frontier": ef.get_frontier(),
        "specific": {
            "min_risk": ef.get_minimum_risk(),
            "max_returns": ef.get_maximum_return(),
            "max_sharpe": ef.get_maximum_sharpe()
        }
    }
    return jsonify(results)


@app.route("/portfolio", methods=["POST"])
def calc_portfolio_performance():
    inputs = request.get_json()
    ef = EfficientFrontierCalculator(inputs["codes"])
    ef.get_performance_by_weight(inputs["weights"])
    perform = ef.get_performance_by_weight(inputs["weights"])
    enhance = ef.get_maximum_return(target_volatility=perform["risk"])
    results = {
        "performance": perform,
        "enhance": enhance
    }
    return jsonify(results)


@app.route("/discrete", methods=["POST"])
def calc_discrete_amount():
    inputs = request.get_json()
    ef = EfficientFrontierCalculator(inputs["codes"])
    amount, remain = ef.get_stock_amount(inputs["weights"], inputs["cash"])
    results = {
        "amounts": [],
        "remains": remain
    }
    for code, name, price in zip(ef.codes, ef.names, ef.latest_prices):
        results["amounts"].append({
            "code": code,
            "name": name,
            "price": price,
            "amount": int(amount[name])
        })
    return jsonify(results)


@app.route("/etf", methods=["POST"])
def get_similar_eft():
    inputs = request.get_json()
    limit = request.args.get("limit")
    if limit is None:
        limit = 5
    return jsonify(etf.calc_match_score(codes=inputs["codes"], limit_count=limit))

if __name__ == "__main__":
    app.run(debug=True)
