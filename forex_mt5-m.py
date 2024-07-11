import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Connect to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader 5 initialization failed")
    raise SystemExit("MetaTrader 5 initialization failed")

# Login to your account
account = 12345678  # replace with your account number
password = "your_password"  # replace with your account password
server = "MetaQuotes-Demo"  # replace with your server
if not mt5.login(account, password, server):
    logging.error("Failed to connect to account")
    mt5.shutdown()
    raise SystemExit("Failed to connect to account")

# Define symbols to trade
symbols = ["EURUSD", "GBPUSD"]

# Define trading parameters
lot_size = 0.1
risk_per_trade = 0.01  # Risk 1% of the account balance per trade
max_drawdown = 0.2  # Maximum drawdown limit

# Helper function to get historical data
def get_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    return data

# Feature engineering for machine learning
def add_features(data):
    data['SMA10'] = data['close'].rolling(window=10).mean()
    data['SMA50'] = data['close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['close'])
    data['upper_band'], data['lower_band'] = compute_bollinger_bands(data['close'])
    data.dropna(inplace=True)
    return data

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std_dev=2):
    sma = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

# Prepare dataset for machine learning
def prepare_dataset(symbol, timeframe, num_bars):
    data = get_data(symbol, timeframe, num_bars)
    data = add_features(data)
    data['target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # Next period's direction
    return data.dropna()

# Train machine learning model
def train_model(symbol):
    data = prepare_dataset(symbol, mt5.TIMEFRAME_M5, 1000)
    X = data[['SMA10', 'SMA50', 'RSI', 'upper_band', 'lower_band']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logging.info(f"Model trained for {symbol} with accuracy: {score:.2f}")
    dump(model, f'{symbol}_model.joblib')  # Save the model
    return model

# Load machine learning model
def load_model(symbol):
    return load(f'{symbol}_model.joblib')

# Predict signal using the model
def predict_signal(symbol, model, data):
    features = data[['SMA10', 'SMA50', 'RSI', 'upper_band', 'lower_band']].tail(1)
    signal = model.predict(features)[0]
    return signal

# Risk management: Calculate lot size based on account balance and risk per trade
def calculate_lot_size(symbol, risk_per_trade, stop_loss_pips):
    account_info = mt5.account_info()
    balance = account_info.balance
    risk_amount = balance * risk_per_trade
    price_per_pip = mt5.symbol_info(symbol).point
    lot_size = risk_amount / (stop_loss_pips * price_per_pip)
    return round(lot_size, 2)

# Execute trades based on signals
def execute_trades(symbol, signal):
    if signal == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
        sl_price = price - 50 * mt5.symbol_info(symbol).point  # 50 pips stop loss
    elif signal == -1:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
        sl_price = price + 50 * mt5.symbol_info(symbol).point  # 50 pips stop loss
    else:
        return

    lot_size = calculate_lot_size(symbol, risk_per_trade, 50)  # Assume 50 pips stop loss for risk calculation
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade execution failed for {symbol}: {result.comment}")
    else:
        logging.info(f"Trade executed for {symbol}: {result}")

# Check for maximum drawdown
def check_drawdown():
    account_info = mt5.account_info()
    equity = account_info.equity
    balance = account_info.balance
    drawdown = (balance - equity) / balance
    if drawdown > max_drawdown:
        logging.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
        return True
    return False

# Portfolio management: Adjust position sizes based on portfolio risk
def adjust_positions(symbols):
    account_info = mt5.account_info()
    balance = account_info.balance
    total_risk = 0
    for symbol in symbols:
        total_risk += mt5.positions_get(symbol=symbol)
    for symbol in symbols:
        position = mt5.positions_get(symbol=symbol)
        if position:
            risk = position[0].volume * position[0].price * 0.01  # Assume 1% risk per position
            if risk > (balance * risk_per_trade):
                close_position(symbol)

# Close position
def close_position(symbol):
    position = mt5.positions_get(symbol=symbol)
    if position:
        order_type = mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if position[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position[0].volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position for {symbol}: {result.comment}")
        else:
            logging.info(f"Position closed for {symbol}: {result}")

# Main trading loop
models = {}
for symbol in symbols:
    models[symbol] = train_model(symbol)  # Train and save models

while True:
    if check_drawdown():
        logging.info("Stopping trading due to maximum drawdown limit")
        break

    adjust_positions(symbols)

    for symbol in symbols:
        try:
            data = get_data(symbol, mt5.TIMEFRAME_M5, 100)
            data = add_features(data)
            model = models[symbol]
            signal = predict_signal(symbol, model, data)

            # Execute trades based on model prediction
            execute_trades(symbol, signal)

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")

    time.sleep(300)  # Wait for 5 minutes before next iteration

# Shutdown MetaTrader 5 connection
mt5.shutdown()
