"""
ScottiGotti OKX Cloud Bot (Full AI-learning version)
- Fetches 1 year OHLCV from OKX via ccxt for BTC/USDT and XRP/USDT
- Runs multiple strategies (SMA, EMA, RSI, MACD, Bollinger, Breakout)
- Ranks strategies, builds a lightweight ensemble (AI Hybrid)
- Paper-trades (simulated) and records ROI
- Daily scheduler (APScheduler) to run backtests and send Telegram report
- Config via environment variables for Render deployment
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import ccxt
import pandas as pd
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('scottigotti')

# ---- Config / Env ----
OKX_API_KEY = os.getenv('OKX_API_KEY', 'PLACEHOLDER_KEY')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', 'PLACEHOLDER_SECRET')
OKX_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', 'PLACEHOLDER_PASSPHRASE')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'PLACEHOLDER_TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'PLACEHOLDER_CHAT_ID')

PAIRS = ['BTC/USDT', 'XRP/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
DAYS_OF_HISTORY = 365
STARTING_BALANCE_USD = 10000.0  # paper trading starting capital

# ---- CCXT OKX instance ----
exchange = ccxt.okx({
    'apiKey': OKX_API_KEY,
    'secret': OKX_API_SECRET,
    'password': OKX_API_PASSPHRASE,
    'enableRateLimit': True,
})

# ---- Indicators ----
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, period=20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower

# ---- Data loader ----
def fetch_ohlcv(pair: str, timeframe: str, since: int = None, limit: int = None) -> pd.DataFrame:
    log.info(f'Fetching OHLCV {pair} {timeframe} since={since} limit={limit}')
    ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def fetch_past_days(pair: str, timeframe: str, days: int) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    try:
        df = fetch_ohlcv(pair, timeframe, since=since, limit=None)
    except Exception as e:
        log.warning('Direct fetch failed, trying iterative fetch: %s', e)
        # iterative fetch for long ranges (safe fallback)
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }.get(timeframe, 60)
        candles_needed = int((days * 24 * 60) / timeframe_minutes)
        all_rows = []
        since_local = since
        batch = 2000
        while True:
            try:
                chunk = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_local, limit=batch)
                if not chunk:
                    break
                all_rows += chunk
                last_ts = chunk[-1][0]
                since_local = last_ts + timeframe_minutes * 60 * 1000
                if len(chunk) < batch:
                    break
            except Exception as e2:
                log.exception('iterative fetch error: %s', e2)
                break
        df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
    return df

# ---- Strategy signals ----
def signals_sma(df: pd.DataFrame, fast=10, slow=30) -> pd.Series:
    a = sma(df['close'], fast)
    b = sma(df['close'], slow)
    sig = (a > b).astype(int)
    return sig.diff().fillna(0)

def signals_ema(df: pd.DataFrame, fast=8, slow=21) -> pd.Series:
    a = ema(df['close'], fast)
    b = ema(df['close'], slow)
    return (a > b).astype(int).diff().fillna(0)

def signals_rsi(df: pd.DataFrame, lower=30, upper=70, period=14) -> pd.Series:
    r = rsi(df['close'], period)
    sig = pd.Series(0, index=df.index)
    sig[(r < lower)] = 1
    sig[(r > upper)] = -1
    return sig.diff().fillna(0)

def signals_macd(df: pd.DataFrame) -> pd.Series:
    macd_line, signal_line, hist = macd(df['close'])
    sig = (macd_line > signal_line).astype(int).diff().fillna(0)
    return sig

def signals_bollinger(df: pd.DataFrame) -> pd.Series:
    upper, mid, lower = bollinger(df['close'])
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] < lower)] = 1
    sig[(df['close'] > upper)] = -1
    return sig.diff().fillna(0)

def signals_breakout(df: pd.DataFrame, lookback=20) -> pd.Series:
    high = df['high'].rolling(lookback).max()
    low = df['low'].rolling(lookback).min()
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > high.shift(1)] = 1
    sig[df['close'] < low.shift(1)] = -1
    return sig.diff().fillna(0)

# ---- Paper trading simulator ----
class PaperTrader:
    def __init__(self, starting_balance=STARTING_BALANCE_USD):
        self.cash = starting_balance
        self.position = 0.0
        self.entry_price = None
        self.history = []

    def buy(self, price: float, fraction=0.25):
        use = self.cash * fraction
        if use <= 0:
            return
        qty = use / price
        self.position += qty
        self.cash -= qty * price
        self.entry_price = price
        self.history.append({'action': 'buy', 'price': price, 'qty': qty, 'time': datetime.utcnow().isoformat()})
        log.debug('Paper buy %s @ %s', qty, price)

    def sell(self, price: float, fraction=1.0):
        qty = self.position * fraction
        if qty <= 0:
            return
        self.position -= qty
        self.cash += qty * price
        self.history.append({'action': 'sell', 'price': price, 'qty': qty, 'time': datetime.utcnow().isoformat()})
        log.debug('Paper sell %s @ %s', qty, price)

    def net_worth(self, market_price: float) -> float:
        return self.cash + self.position * market_price

# ---- Strategy runner & evaluator ----
STRATEGIES = {
    'SMA': signals_sma,
    'EMA': signals_ema,
    'RSI': signals_rsi,
    'MACD': signals_macd,
    'BOLL': signals_bollinger,
    'BREAK': signals_breakout,
}

def run_backtest(df: pd.DataFrame, strategy_fn, pair: str) -> Dict:
    trader = PaperTrader()
    # If strategy_fn is a precomputed series (for hybrid), handle that
    if isinstance(strategy_fn, pd.Series):
        signals = strategy_fn
    else:
        signals = strategy_fn(df)
    results = {'trades': 0, 'wins': 0, 'losses': 0}
    last_entry_price = None

    for ts, row in df.iterrows():
        s = signals.loc[ts] if ts in signals.index else 0
        price = float(row['close'])
        if s > 0 and trader.cash > 0:
            trader.buy(price, fraction=0.25)
            results['trades'] += 1
            last_entry_price = price
        elif s < 0 and trader.position > 0:
            entry = last_entry_price or trader.entry_price or price
            pnl = (price - entry) / (entry + 1e-9)
            if pnl > 0:
                results['wins'] += 1
            else:
                results['losses'] += 1
            trader.sell(price, fraction=1.0)

    final = trader.net_worth(float(df['close'].iloc[-1]))
    roi_pct = (final - STARTING_BALANCE_USD) / STARTING_BALANCE_USD * 100.0
    results.update({'final': final, 'roi_pct': roi_pct, 'history_len': len(trader.history)})
    return results

def evaluate_strategies(df: pd.DataFrame, pair: str) -> Dict:
    scores = {}
    for name, fn in STRATEGIES.items():
        try:
            res = run_backtest(df, fn, pair)
            scores[name] = res
        except Exception as e:
            log.exception('Error running strategy %s for %s: %s', name, pair, e)
            scores[name] = {'error': str(e)}
    ranked = sorted([(name, v) for name, v in scores.items() if 'roi_pct' in v], key=lambda x: x[1]['roi_pct'], reverse=True)
    return {'scores': scores, 'ranked': ranked}

# ---- Ensemble AI Hybrid ----
def ai_hybrid_signal(df: pd.DataFrame, top_strategies: List[str]) -> pd.Series:
    votes = pd.DataFrame(index=df.index)
    for sname in top_strategies:
        fn = STRATEGIES.get(sname)
        if fn is None:
            continue
        votes[sname] = fn(df)
    summed = votes.fillna(0).sum(axis=1)
    sig = summed.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return sig.diff().fillna(0)

# ---- Telegram helper ----
def send_telegram(text: str) -> bool:
    if TELEGRAM_TOKEN.startswith('PLACEHOLDER') or TELEGRAM_CHAT_ID.startswith('PLACEHOLDER'):
        log.warning('Telegram token or chat id is placeholder — message not sent')
        return False
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            log.warning('Telegram send failed: %s %s', r.status_code, r.text)
            return False
    except Exception as e:
        log.exception('Telegram send exception: %s', e)
        return False

# ---- Daily job ----
def daily_job():
    log.info('Starting daily backtest job')
    summary = {'date': datetime.utcnow().isoformat(), 'pairs': {}}
    for pair in PAIRS:
        pair_summary = {}
        for tf in TIMEFRAMES:
            try:
                df = fetch_past_days(pair, tf, DAYS_OF_HISTORY)
                if df.empty or len(df) < 50:
                    log.warning('No or insufficient data for %s %s', pair, tf)
                    pair_summary[tf] = {'error': 'no data'}
                    continue

                ev = evaluate_strategies(df, pair)
                top_two = [name for name, _ in ev['ranked'][:2]] if ev['ranked'] else []
                hybrid_res = None
                if top_two:
                    hybrid_sig = ai_hybrid_signal(df, top_two)
                    hybrid_res = run_backtest(df, hybrid_sig, pair)

                pair_summary[tf] = {
                    'ranked': [(n, round(v['roi_pct'], 3) if 'roi_pct' in v else None) for n, v in ev['ranked'][:5]],
                    'best': ev['ranked'][0][0] if ev['ranked'] else None,
                    'best_roi': round(ev['ranked'][0][1]['roi_pct'], 3) if ev['ranked'] else None,
                    'hybrid': hybrid_res,
                }
                log.info('Pair %s %s best: %s', pair, tf, pair_summary[tf]['best'])
            except Exception as e:
                log.exception('Failed for %s %s: %s', pair, tf, e)
                pair_summary[tf] = {'error': str(e)}

        summary['pairs'][pair] = pair_summary

    # save summary file
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    filename = f'summary_{stamp}.json'
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info('Saved summary %s', filename)

    # Build short telegram message (top-level)
    lines = [f'📊 ScottiGotti Daily Report ({summary["date"]})\\nExchange: OKX']
    for pair, pair_summary in summary['pairs'].items():
        tf_info = pair_summary.get('1d') or next(iter(pair_summary.values()), None)
        if tf_info and isinstance(tf_info, dict):
            best = tf_info.get('best')
            best_roi = tf_info.get('best_roi')
            if best and (best_roi is not None):
                lines.append(f'Pair: {pair} | TF: 1d | Best: {best} | ROI: {best_roi}%')
            elif best:
                lines.append(f'Pair: {pair} | TF: 1d | Best: {best}')
    send_telegram('\\n'.join(lines))

# ---- Scheduler ----
scheduler = BackgroundScheduler()
scheduler.add_job(daily_job, 'cron', hour=8, minute=0)  # runs daily at 08:00 UTC

def run_once():
    daily_job()

if __name__ == '__main__':
    log.info('Starting ScottiGotti OKX AI worker')
    scheduler.start()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        log.info('Shutdown')
