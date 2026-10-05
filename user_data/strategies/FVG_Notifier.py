"""
SMC Fair Value Gap (FVG) Notification Strategy for Freqtrade

The strategy tracks SMC swing structure, TRAMA(99)/HMA(21) and MACD on the
15-minute base timeframe, notifying through the configured alert channel
(RPC/Telegram) and the log when the latest closed candle prints an HMA/TRAMA
cross or a MACD/signal cross.
"""

import atexit
import logging
import os
import re
import sys
import time
import pandas as pd
from freqtrade.strategy import IStrategy
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
import numpy as np
from tabulate import tabulate

# Technical analysis library (talib.abstract for MACD + HMA/WMA).
try:
    import talib.abstract as ta
except ImportError:  # pragma: no cover
    ta = None

# Import Smart Money Concepts library
try:
    from smartmoneyconcepts import smc
except ImportError:
    smc = None

logger = logging.getLogger(__name__)


def _format_volume(vol: float | None) -> str:
    """Format a 24h quote volume (e.g. 359.5M, 1.24M, 650K). Returns '-' when unknown."""
    if not vol or vol != vol:  # None or NaN
        return "-"
    if vol >= 1e9:
        return f"{vol / 1e9:.2f}B"
    if vol >= 1e6:
        return f"{vol / 1e6:.2f}M"
    if vol >= 1e3:
        return f"{vol / 1e3:.1f}K"
    return f"{vol:.0f}"


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted moving average matching Pine's ta.wma (most recent value has the largest weight)."""
    if length < 1:
        return series.astype(float)
    weights = np.arange(1, length + 1, dtype=float)

    def _calc(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / weights.sum())

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def hma(dataframe: pd.DataFrame, length: int = 21) -> pd.Series:
    """
    Hull Moving Average matching Pine's ta.hma:
    WMA(2 * WMA(close, floor(length/2)) - WMA(close, length), round(sqrt(length))).

    Uses talib's HMA function when available; talib 0.7.x only ships WMA, so the
    same canonical formula is composed from ta.WMA. Falls back to a pure-pandas
    WMA if talib is not installed.
    """
    close = dataframe['close']
    half = max(1, length // 2)
    sq = max(1, int(round(np.sqrt(length))))
    if ta is not None:
        if hasattr(ta, 'HMA'):
            return pd.Series(ta.HMA(close, timeperiod=length), index=close.index)
        fast = pd.Series(ta.WMA(close, timeperiod=half), index=close.index)
        slow = pd.Series(ta.WMA(close, timeperiod=length), index=close.index)
        raw = 2.0 * fast - slow
        return pd.Series(ta.WMA(raw, timeperiod=sq), index=close.index)
    fast = _wma(close, half)
    slow = _wma(close, length)
    raw = 2.0 * fast - slow
    return _wma(raw, sq)


def trama(dataframe: pd.DataFrame, length: int = 99) -> pd.Series:
    """
    TRAMA (LuxAlgo) port of the Pine source:

        hh = max(sign(change(highest(length))), 0)
        ll = max(sign(-change(lowest(length))), 0)
        tc = sma((hh > 0 or ll > 0) ? 1 : 0, length) ^ 2
        ama := nz(ama[1] + tc * (src - ama[1]), src)

    Rolling windows include the current bar (ta.highest/ta.lowest/ta.sma), and the
    extremes come from the HIGH (ta.highest) and LOW (ta.lowest) series, while the
    recursion source defaults to CLOSE (trama_src). During the warm-up, where tc is
    na, ama falls back to the source value (Pine's nz).
    """
    close = dataframe['close']
    high = dataframe['high']
    low = dataframe['low']
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()

    # 1 when highest made a new high (hh) or a new lower low was made (ll).
    hh = (highest.diff() > 0).astype(float)
    ll = (lowest.diff() < 0).astype(float)
    cond = ((hh + ll) > 0).astype(float)
    tc = cond.rolling(length, min_periods=length).mean() ** 2

    c = close.to_numpy(dtype=float)
    t = tc.to_numpy(dtype=float)
    ama = np.empty(len(close), dtype=float)
    ama[0] = c[0]
    for i in range(1, len(close)):
        if np.isnan(t[i]):
            ama[i] = c[i]
        else:
            ama[i] = ama[i - 1] + t[i] * (c[i] - ama[i - 1])

    return pd.Series(ama, index=dataframe.index)


def pine_swing_highs_lows(dataframe: pd.DataFrame, size: int = 50) -> pd.DataFrame:
    """
    Replicates the swing high/low logic of the LuxAlgo 'Smart Money Concepts'
    Pine indicator (getCurrentStructure).

    A pivot at bar j is confirmed once `size` bars have followed it:
      swing-high candidate: high[j] > max(high[j+1 .. j+size])
      swing-low candidate:  low[j]  < min(low[j+1  .. j+size])
    A leg state machine (starting bearish) alternates highs and lows; a swing is
    recorded only when the leg flips (0 <-> 1). Same-direction confirmed pivots
    within a leg are ignored, and no synthetic swings are added at the ends.

    Returns a DataFrame with columns 'HighLow' (1 = swing high, -1 = swing low,
    NaN otherwise) and 'Level' (the pivot price) - the same format
    smc.swing_highs_lows() produces, so smc.ob() can consume it directly.
    """
    high = dataframe["high"].to_numpy(dtype=float)
    low = dataframe["low"].to_numpy(dtype=float)
    n = len(high)
    size = int(size)

    highlow = np.full(n, np.nan, dtype=np.float64)
    level = np.full(n, np.nan, dtype=np.float64)

    if n <= size + 1:
        return pd.DataFrame({"HighLow": highlow, "Level": level})

    from numpy.lib.stride_tricks import sliding_window_view

    # For every bar j with j+size < n: max/min of the next `size` bars (j+1..j+size)
    next_high_max = sliding_window_view(high[1:], size).max(axis=1)
    next_low_min = sliding_window_view(low[1:], size).min(axis=1)

    high_candidate = high[: n - size] > next_high_max
    low_candidate = low[: n - size] < next_low_min

    # Legacy state machine over "current bars" i; pivot j = i - size gets recorded
    # when the leg flips. leg: 0 = bearish, 1 = bullish (mirrors Pine's constants).
    leg = 0
    for i in range(size, n):
        j = i - size
        if high_candidate[j]:
            new_leg = 0
        elif low_candidate[j]:
            new_leg = 1
        else:
            new_leg = leg
        if new_leg != leg:
            if new_leg == 1:  # 0 -> 1: last leg was down, a swing LOW was made
                highlow[j] = -1.0
                level[j] = low[j]
            else:             # 1 -> 0: a swing HIGH was made
                highlow[j] = 1.0
                level[j] = high[j]
        leg = new_leg

    return pd.DataFrame({"HighLow": highlow, "Level": level})

class SMCAlert(BaseModel):
    """
    Pydantic model representing a smart money concept alert data structure.
    """
    type: str = Field(..., description="Type of SMC alert (e.g., 'FVG' or 'OB')")
    pair: str = Field(..., description="Trading pair symbol")
    direction: str = Field(..., description="'Bullish' or 'Bearish'")
    percentage: float = Field(..., description="Size as a percentage")

    def format_entry(self) -> str:
        """
        Formats the alert data into Markdown for Telegram dispatch.
        """
        emoji = "🟢" if self.direction == 'Bullish' else "🔴"
        return f"{self.type} {emoji} {self.pair} `{self.percentage:.2f}%`"

class FVGAlert(SMCAlert):
    """
    Specialized Pydantic model for Fair Value Gap alerts.
    """
    pass

class OBAlert(SMCAlert):
    """
    Specialized Pydantic model for Order Block alerts.
    """
    pass

class FVG_Notifier(IStrategy):
    """
    Freqtrade strategy dedicated to detecting SMC Fair Value Gaps and alerting via Telegram.
    """
    # Processing options
    process_only_new_candles = True
    use_exit_signal = False
    timeframe = '1h'
    
    # Minimal ROI and Stoploss (Placeholders required by Freqtrade IStrategy)
    minimal_roi = {
        "0": 100
    }
    stoploss = -0.99
    # TRAMA needs 99 bars on the 15m base timeframe; 200 candles give margin
    # over that warm-up (and the HMA/MACD lookbacks are well inside it).
    startup_candle_count = 200

    # Base-timeframe indicator settings for the crossover notifications.
    hma_length = 21
    trama_length = 99
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # Minimum seconds to wait before sending another FVG alert for the same pair.
    # Prevents the Telegram bot from being saturated when a forming FVG repaints or
    # when a pair keeps producing candidate gaps. FVG detection intentionally uses
    # the forming (unfinished) candle, so alerts can repaint by design.
    fvg_pair_cooldown_seconds = 300  # 5 minutes

    # Number of bars used to confirm a swing high/low (Pine 'swingsLengthInput',
    # the LuxAlgo indicator default). A pivot at bar j is confirmed once `swing_size`
    # candles have followed it without exceeding it.
    swing_size = 50

    notification_whitelist: list[str] = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ZEC/USDT']  # Only listed pairs are notified.

    # Minimum number of minutes between scoreboard prints. Can be overridden from
    # the strategy config with the "scoreboard_interval_minutes" key.
    scoreboard_interval_minutes = 15

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if smc is None:
            logger.warning(
                "The 'smartmoneyconcepts' package is not installed. "
                "Please run 'pip install smartmoneyconcepts' in your Freqtrade environment."
            )
        self.pending_smc_alerts: list[SMCAlert] = []
        # Remembers the last FVG alert time per pair (epoch seconds) for rate limiting.
        self._last_fvg_alert_time: dict[str, float] = {}
        # Per-pair metrics/score for the end-of-loop scoreboard table.
        self.pair_metrics: dict[str, dict] = {}
        # Last printed scoreboard string, to avoid re-printing identical tables.
        self._last_scoreboard: str | None = None
        # 24h quote volume per pair, refreshed once per bot loop (see bot_loop_start).
        self._vol24h: dict[str, float] = {}
        # Epoch-second deadline for the next scoreboard print (see bot_loop_start).
        now = time.time()
        self._next_scoreboard_print = now + 1 * 60.0
        # Base-candle timestamps of the last HMA/TRAMA and MACD cross alerts per
        # pair, so a single cross produces exactly one notification.
        self._last_hma_cross_alert: dict[str, pd.Timestamp] = {}
        self._last_macd_cross_alert: dict[str, pd.Timestamp] = {}

    def _check_hma_trama_cross(self, pair: str, dataframe: pd.DataFrame) -> None:
        """
        Notifies when HMA(21) crosses above/below TRAMA(99) on the LATEST CLOSED
        base candle (iloc[-2]; iloc[-1] is still forming). One notification per
        crossed candle.
        """
        if len(dataframe) < 3:
            return
        h = dataframe['hma']
        t = dataframe['trama']
        i = -2
        h_i, t_i = h.iloc[i], t.iloc[i]
        h_j, t_j = h.iloc[i - 1], t.iloc[i - 1]
        if pd.isna(h_i) or pd.isna(t_i) or pd.isna(h_j) or pd.isna(t_j):
            return

        crossed_up = h_i > t_i and h_j <= t_j
        crossed_down = h_i < t_i and h_j >= t_j
        if crossed_up == crossed_down:
            return

        candle_time = dataframe['date'].iloc[i]
        if self._last_hma_cross_alert.get(pair) == candle_time:
            return

        msg = (
            f"{pair}: HMA crossed {'above' if crossed_up else 'below'} TRAMA "
            f"({self.timeframe}): HMA {h_i:.2f} vs TRAMA {t_i:.2f}"
        )
        logger.info(msg)
        #if self.dp:
        #    self.dp.send_msg(msg)
        self._last_hma_cross_alert[pair] = candle_time

    def _check_macd_cross(self, pair: str, dataframe: pd.DataFrame) -> None:
        """
        Notifies when MACD crosses above/below its signal line on the LATEST
        CLOSED base candle. One notification per crossed candle.
        """
        if len(dataframe) < 3:
            return
        m = dataframe['macd']
        s = dataframe['macdsignal']
        i = -2
        m_i, s_i = m.iloc[i], s.iloc[i]
        m_j, s_j = m.iloc[i - 1], s.iloc[i - 1]
        if pd.isna(m_i) or pd.isna(s_i) or pd.isna(m_j) or pd.isna(s_j):
            return

        crossed_up = m_i > s_i and m_j <= s_j
        crossed_down = m_i < s_i and m_j >= s_j
        if crossed_up == crossed_down:
            return

        candle_time = dataframe['date'].iloc[i]
        if self._last_macd_cross_alert.get(pair) == candle_time:
            return

        msg = (
            f"{pair}: MACD crossed {'above' if crossed_up else 'below'} signal "
            f"({self.timeframe}): MACD {m_i:.4f} vs Signal {s_i:.4f}"
        )
        logger.info(msg)
        #if self.dp:
        #    self.dp.send_msg(msg)
        self._last_macd_cross_alert[pair] = candle_time

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Computes TRAMA(99), HMA(21) and MACD on the 15m base dataframe and fires
        notifications on HMA/TRAMA and MACD/signal crosses (last closed candle).
        """
        if smc is None:
            return dataframe

        is_live = self.dp and self.dp.runmode.value in ('live', 'dry_run')
        if not is_live:
            return dataframe
        
        pair = metadata.get('pair', 'Unknown Pair')

        if pair in self.notification_whitelist:
            logger.info(f"Processing {pair} for SMC indicators and notifications.")
            # ---------------------------------------------------------------------
            # Base timeframe (15m) indicators: TRAMA(99), HMA(21), MACD
            # ---------------------------------------------------------------------
            dataframe['trama'] = trama(dataframe, self.trama_length)
            dataframe['hma'] = hma(dataframe, self.hma_length)
            self._check_hma_trama_cross(pair, dataframe)
            if ta is not None:
                macd = ta.MACD(dataframe)
                dataframe['macd'] = macd['macd']
                dataframe['macdsignal'] = macd['macdsignal']
                dataframe['macdhist'] = macd['macdhist']
                self._check_macd_cross(pair, dataframe)
            

        # Pine-style (LuxAlgo) swing high/low detection - see pine_swing_highs_lows.
        swing_df = pine_swing_highs_lows(dataframe, size=self.swing_size)


        # ---------------------------------------------------------------------
        # Swing High and Low Detection
        # ---------------------------------------------------------------------

        # Extract raw swing signals
        # swing_df['HighLow']: 1 = Swing High, -1 = Swing Low, 0 = None
        # swing_df['Level']: Price of the swing point
        dataframe['swing_type'] = swing_df['HighLow']
        dataframe['swing_level'] = swing_df['Level']

        # Separate levels into distinct High and Low series
        dataframe['latest_swing_high'] = np.where(
            dataframe['swing_type'] == 1, dataframe['swing_level'], np.nan
        )
        dataframe['latest_swing_low'] = np.where(
            dataframe['swing_type'] == -1, dataframe['swing_level'], np.nan
        )

        # Forward-fill to project the latest known swing levels onto current candles
        dataframe['latest_swing_high'] = dataframe['latest_swing_high'].ffill()
        dataframe['latest_swing_low'] = dataframe['latest_swing_low'].ffill()

        # Relative Distance % (Range / Low)
        dataframe['swing_range_percentage'] = (
            (dataframe['latest_swing_high'] - dataframe['latest_swing_low']) / dataframe['latest_swing_low']
        ) * 100

        # ---------------------------------------------------------------------
        # Per-pair scoreboard metric
        # ---------------------------------------------------------------------

        # Count of confirmed, unmitigated FVGs during the last CLOSED base candle
        # (iloc[-2]; iloc[-1] is the still-forming candle).
        ratio = float(dataframe['swing_range_percentage'].iloc[-1])
        high = float(dataframe['latest_swing_high'].iloc[-1])
        low = float(dataframe['latest_swing_low'].iloc[-1])
        self.pair_metrics[pair] = {
            'pair': pair,
            'ratio': round(ratio, 2),
            'high': high,
            'low': low,
        }

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        No automatic entries — notification mode only.
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        No automatic exits.
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def get_ny_time(self) -> str:
        # 1. Get the current time in the New York timezone
        ny_time = datetime.now(ZoneInfo("America/New_York"))

        # 2. Format the datetime object into a string
        # This gives a format like: "2026-09-02 21:05:32"
        ny_string = ny_time.strftime("%b %d %Y %H:%M:%S")

        return ny_string

    def _print_scoreboard(self) -> None:
        """
        Renders the per-pair SMC scoreboard.

        On a real terminal the table is redrawn in place (top-like behavior):
        the screen is cleared and rewritten on every bot loop. The same table is
        normally mirrored to the log file only when its content actually changes,
        to avoid spamming the log on every throttle tick. Setting ``force`` makes
        it log unconditionally; the periodic timer in bot_loop_start uses that so
        the scoreboard is printed at regular intervals even when nothing moved.
        """
        rows = []
        for m in sorted(
            self.pair_metrics.values(), key=lambda x: (self._vol24h.get(x['pair']), x['ratio']), reverse=True
        ):
            short_pair = re.sub('/.*', '', m['pair'])
            rows.append([
                short_pair,
                _format_volume(self._vol24h.get(m['pair'])),
                f"{m['ratio']:.2f}",
                f"{m['high']}/{m['low']}",
                
            ])

        if rows:
            frame = tabulate(
                rows,
                headers=["Pair", "24H Vol", "Ratio", "High/Low"],
                tablefmt="github",
            )

            if frame != self._last_scoreboard:
                self._last_scoreboard = frame
                logger.info(f"SMC Scoreboard ({self.get_ny_time()} {self.timeframe}):\n{frame}")

    def bot_loop_start(self, **kwargs) -> None:
        """
        Executed once at the beginning of each bot iteration, AFTER all pair 
        data frames have finished running through populate_indicators().
        """
        #if not self.pending_smc_alerts:
        #    return
        
        if self.dp and self.dp.runmode.value in ('live', 'dry_run'):
            # Refresh 24h quote volume once per bot loop and cache it, so the
            # scoreboard never triggers repeated network calls. dp.ticker()
            # returns {} on error; the previous value is then kept.
            for pair in self.pair_metrics:
                ticker = self.dp.ticker(pair)
                vol = ticker.get('quoteVolume') if ticker else None
                if vol is not None:
                    self._vol24h[pair] = float(vol)

            # Print the scoreboard at most once every N minutes (N = interval_minutes),
            # regardless of how fast the bot loop throttles.
            interval_minutes = float(
                self.config.get('scoreboard_interval_minutes', self.scoreboard_interval_minutes)
            )
            now = time.time()
            if now >= self._next_scoreboard_print:
                # The very first loop prints immediately (i.e. on app start once
                # the pair metrics are available), and the interval is then
                # counted from that print: every N minutes the scoreboard is
                # re-printed even if its content did not change.
                self._print_scoreboard()
                self._next_scoreboard_print = now + interval_minutes * 60.0

        # Clear the queue for the next loop
        self.pending_smc_alerts.clear()
 