"""
SMC Fair Value Gap (FVG) Notification Strategy for Freqtrade

This strategy detects Fair Value Gaps (FVG) on a 1-minute timeframe using the
`smartmoneyconcepts` library and sends real-time Telegram notifications when a new FVG
forms on the latest closed candle.
"""

import atexit
import logging
import os
import re
import sys
import pandas as pd
from freqtrade.strategy import IStrategy
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
import numpy as np
from tabulate import tabulate

# Import Smart Money Concepts library
try:
    from smartmoneyconcepts import smc
except ImportError:
    smc = None

logger = logging.getLogger(__name__)


def _enable_vt() -> None:
    """Enable ANSI/VT escape sequence processing on the Windows console."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def _restore_cursor() -> None:
    """Re-show the terminal cursor on process exit."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


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


def dynamic_delta_fvg(
    dataframe: pd.DataFrame,
    atr_mult: float = 0.2,
    vol_mult: float = 0.0,
    mit_mode: str = 'touch',
    filter_overlap: bool = True,
    atr_period: int = 14,
    vol_period: int = 20,
) -> pd.DataFrame:
    """
    Port of the LuxAlgo 'Dynamic Delta FVG' detection logic.

    A gap is detected at the CURRENT bar i (only on confirmed candles in Pine,
    which maps to rows < len(dataframe) here) when bar i is separated from bar
    i-2:

      bullish: low[i]   > high[i-2]   -> top=low[i],     bottom=high[i-2]
      bearish: high[i]  <  low[i-2]   -> top=low[i-2],   bottom=high[i]

    Filters (Pine defaults):
      - volume: volume[i-1] >= SMA(volume, 20)[i-1] * vol_mult   (0 = disabled)
      - size:   gap depth >= ATR(14) * atr_mult
      - overlap: when a new gap overlaps an older unmitigated gap, the older one
        is removed (only the most recent imbalance in a zone stays active).

    Mitigation ('touch' = gap borders touched, 'full' = fully filled):
      - bullish gap mitigated the first bar j>i with low[j]  < level
      - bearish gap mitigated the first bar j>i with high[j] > level

    Volume delta uses the Pine fallback (no lower-TF data): candle direction
    (close>open -> 100% buy). The delta shown belongs to bar i-1 (buyPct[1]).

    Returns a DataFrame with columns:
      FVG (1 bullish, -1 bearish, NaN), Top, Bottom, MitigatedIndex
      (0 = unmitigated, else bar index that filled it), Active (post
      mitigation + overlap filtering), BuyPct, SellPct.
    """
    df = dataframe
    n = len(df)

    # ATR with Wilder smoothing; fall back to TR on early bars (nz(ta.atr, ta.tr)).
    prev_close = df['close'].shift(1)
    tr = pd.concat(
        [df['high'] - df['low'],
         (df['high'] - prev_close).abs(),
         (df['low'] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean().fillna(tr)

    vol_avg = df['volume'].rolling(vol_period).mean()

    is_bull = df['low'] > df['high'].shift(2)
    is_bear = df['high'] < df['low'].shift(2)
    # vol_mult == 0 disables the volume filter in Pine; any other value gates on
    # volume[1] >= SMA(volume, vol_period)[1] * vol_mult.
    if vol_mult == 0:
        vol_filter = pd.Series(True, index=df.index)
    else:
        vol_filter = df['volume'].shift(1) >= vol_avg.shift(1) * vol_mult
    size_bull = (df['low'] - df['high'].shift(2)) >= atr * atr_mult
    size_bear = (df['low'].shift(2) - df['high']) >= atr * atr_mult

    bull = is_bull & vol_filter & size_bull
    bear = is_bear & vol_filter & size_bear

    fvg = np.where(bull, 1.0, np.where(bear, -1.0, np.nan))
    top = np.where(bull, df['low'], np.where(bear, df['low'].shift(2), np.nan))
    bottom = np.where(bull, df['high'].shift(2), np.where(bear, df['high'], np.nan))

    # Volume delta (Pine fallback: candle direction of the previous bar).
    delta_buy = np.full(n, 100.0)
    close_arr = df['close'].to_numpy()
    open_arr = df['open'].to_numpy()
    delta_buy[close_arr < open_arr] = 0.0
    delta_buy[close_arr == open_arr] = 50.0
    buy_pct = pd.Series(delta_buy, index=df.index).shift(1)
    sell_pct = 100.0 - buy_pct

    # Mitigation.
    low_arr = df['low'].to_numpy()
    high_arr = df['high'].to_numpy()
    touch = mit_mode == 'touch'
    mitigated_index = np.zeros(n, dtype=np.int32)
    det = np.flatnonzero(~np.isnan(fvg))
    for i in det:
        if fvg[i] == 1.0:
            level = top[i] if touch else bottom[i]
            hits = np.flatnonzero(low_arr[i + 1:] < level)
        else:
            level = bottom[i] if touch else top[i]
            hits = np.flatnonzero(high_arr[i + 1:] > level)
        if hits.size:
            mitigated_index[i] = int(i + 1 + hits[0])

    # Overlap filtering (optional): remove a gap when any LATER detected gap
    # overlaps its price zone, so only the most recent imbalance in a zone stays
    # active (Pine's filterOverlapInput).
    active_det = np.ones(len(det), dtype=bool)
    if filter_overlap:
        for a in range(len(det)):
            i = det[a]
            for b in range(a + 1, len(det)):
                j = det[b]
                if max(bottom[i], bottom[j]) < min(top[i], top[j]):
                    active_det[a] = False
                    break
    active_det &= mitigated_index[det] == 0

    active_out = np.zeros(n, dtype=bool)
    active_out[det] = active_det

    return pd.DataFrame({
        'FVG': fvg,
        'Top': top,
        'Bottom': bottom,
        'MitigatedIndex': np.where(np.isnan(fvg), np.nan, mitigated_index),
        'Active': active_out,
        'BuyPct': buy_pct,
        'SellPct': sell_pct,
    })

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
    timeframe = '15m'
    
    # Minimal ROI and Stoploss (Placeholders required by Freqtrade IStrategy)
    minimal_roi = {
        "0": 100
    }
    stoploss = -0.99
    startup_candle_count = 100  # Ensures adequate historical candles for SMC calculations

    # Minimum seconds to wait before sending another FVG alert for the same pair.
    # Prevents the Telegram bot from being saturated when a forming FVG repaints or
    # when a pair keeps producing candidate gaps. FVG detection intentionally uses
    # the forming (unfinished) candle, so alerts can repaint by design.
    fvg_pair_cooldown_seconds = 300  # 5 minutes

    # Number of bars used to confirm a swing high/low (Pine 'swingsLengthInput',
    # the LuxAlgo indicator default). A pivot at bar j is confirmed once `swing_size`
    # candles have followed it without exceeding it.
    swing_size = 50

    # Dynamic Delta FVG detection filters (LuxAlgo 'Dynamic Delta FVG' defaults).
    fvg_atr_mult = 0.2          # min gap depth as a multiple of ATR(14)
    fvg_vol_mult = 0.0          # min previous-bar volume vs SMA(20); 0 = disabled
    fvg_mit_mode = 'touch'      # 'touch' = borders touched, 'full' = fully filled
    fvg_filter_overlap = True   # keep only the most recent gap in a price zone
    fvg_atr_period = 14
    fvg_vol_period = 20

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

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculates Fair Value Gaps using the smartmoneyconcepts library and fires
        Telegram alerts for newly formed gaps on the latest closed candle.
        """
        if smc is None:
            return dataframe

        is_live = self.dp and self.dp.runmode.value in ('live', 'dry_run')
        if not is_live:
            return dataframe
        
        pair = metadata.get('pair', 'Unknown Pair')

        # ---------------------------------------------------------------------
        # 1m timeframe FVG counts (base timeframe)
        # ---------------------------------------------------------------------
        fvg_df = dynamic_delta_fvg(
            dataframe,
            atr_mult=self.fvg_atr_mult,
            vol_mult=self.fvg_vol_mult,
            mit_mode=self.fvg_mit_mode,
            filter_overlap=self.fvg_filter_overlap,
            atr_period=self.fvg_atr_period,
            vol_period=self.fvg_vol_period,
        )

        fvg = fvg_df['FVG'].iloc[-1]
        if fvg in [1, -1]:
            logger.info(f"Pair {pair}: has new FVG")


        # Active gaps = detected, unmitigated and not overlapped by a later gap.
        dataframe['bullish_fvg_count'] = (
            (fvg_df['FVG'] == 1) & fvg_df['Active']).astype(int)
        dataframe['bearish_fvg_count'] = (
            (fvg_df['FVG'] == -1) & fvg_df['Active']).astype(int)

        # Total confirmed, unmitigated FVGs (both directions) per base candle.
        dataframe['fvg_total_count'] = (
            dataframe['bullish_fvg_count'] + dataframe['bearish_fvg_count'])

        # Pine-style (LuxAlgo) swing high/low detection - see pine_swing_highs_lows.
        swing_df = pine_swing_highs_lows(dataframe, size=self.swing_size)
        #ob_df = smc.ob(dataframe, swing_df)

        # Define lookback window
        n = 12

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
        fvg_count = int(dataframe['fvg_total_count'].iloc[-1])
        ratio = float(dataframe['swing_range_percentage'].iloc[-1])
        high = float(dataframe['latest_swing_high'].iloc[-1])
        low = float(dataframe['latest_swing_low'].iloc[-1])
        self.pair_metrics[pair] = {
            'pair': pair,
            'fvg_count': fvg_count,
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
        mirrored to the log file only when its content actually changes, to avoid
        spamming the log on every throttle tick.
        """
        rows = []
        for m in sorted(
            self.pair_metrics.values(), key=lambda x: (x['ratio'], x['fvg_count']), reverse=True
        ):
            short_pair = re.sub('/.*', '', m['pair'])
            rows.append([
                short_pair,
                f"{m['ratio']:.2f}",
                f"{m['high']}/{m['low']}",
            ])
        frame = tabulate(
            rows,
            headers=["Pair", "Ratio", "High/Low"],
            tablefmt="github",
        )

        if frame != self._last_scoreboard:
            if sys.stdout.isatty():
                header = (
                    f"SMC Scoreboard  {self.get_ny_time()}  "
                    f"timeframe={self.timeframe}  pairs={len(rows)}"
                )
                self._render_top(header, frame)

            self._last_scoreboard = frame
            logger.info(f"SMC Scoreboard ({self.get_ny_time()} {self.timeframe}):\n{frame}")

    def _render_top(self, header: str, body: str) -> None:
        """Redraw the terminal in place (top-style) with the given frame."""
        if not getattr(self, "_top_initialized", False):
            _enable_vt()
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
            atexit.register(_restore_cursor)
            self._top_initialized = True
        sys.stdout.write(f"\033[2J\033[H{header}\n\n{body}\n")
        sys.stdout.flush()

    def bot_loop_start(self, **kwargs) -> None:
        """
        Executed once at the beginning of each bot iteration, AFTER all pair 
        data frames have finished running through populate_indicators().
        """
        #if not self.pending_smc_alerts:
        #    return
        
        if self.dp and self.dp.runmode.value in ('live', 'dry_run'):
            self._print_scoreboard()

            # Send a single Telegram notification
            # Sort Pydantic FVGAlert objects by gap percentage descending
            #self.pending_smc_alerts.sort(key=lambda x: x.percentage, reverse=True)

            # Format entries into Markdown strings
            #formatted_alerts = [alert.format_entry() for alert in self.pending_smc_alerts]

            # Build a single unified message        
            #header = f"{self.get_ny_time()} {self.timeframe}\n"
            #combined_body = header + "\n".join(formatted_alerts)

            #self.dp.send_msg(combined_body)

        # Clear the queue for the next loop
        self.pending_smc_alerts.clear()
