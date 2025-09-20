# -*- coding: utf-8 -*-
import asyncio
import os
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta

# قم بتغيير هذه القيم بمعلوماتك الخاصة
# Binance API Key & Secret
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

async def vwap_test():
    """
    Script to test pandas-ta VWAP calculation with ccxt data.
    """
    # 1. Connect to Binance
    print("Connecting to Binance...")
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
    })

    # 2. Fetch OHLCV data for BTC/USDT
    symbol = 'BTC/USDT'
    timeframe = '15m'
    limit = 200
    print(f"Fetching {limit} {timeframe} candles for {symbol}...")
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
    except Exception as e:
        print(f"Failed to fetch data from Binance: {e}")
        return

    # 3. Create and clean DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Crucial data cleaning steps
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['volume'] = df['volume'].replace(0, 1e-6)

    # 4. Calculate VWAP
    print("Calculating VWAP...")
    df.ta.vwap(append=True)
    
    # 5. Print result
    vwap_col_name = ta.utils.get_last_columns(df, "VWAP")
    if vwap_col_name and not df[vwap_col_name].iloc[-1:].isnull().values.any():
        print(f"Success! VWAP calculated. Last value: {df[vwap_col_name].iloc[-1]:.2f}")
    else:
        print("Error: VWAP column not found or contains NaN values.")
        print("DataFrame head after calculation:\n", df.tail())
    
    print("\nTest completed.")

if __name__ == '__main__':
    asyncio.run(vwap_test())
