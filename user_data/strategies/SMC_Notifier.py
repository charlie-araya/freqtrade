"""
SMC Fair Value Gap (FVG) Notification Strategy for Freqtrade

This strategy detects Fair Value Gaps (FVG) on a 15-minute timeframe using the
`smartmoneyconcepts` library and sends real-time Telegram notifications when a new FVG
forms on the latest closed candle.
"""

import logging
import re
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

class SMC_Notifier(IStrategy):
    """
    Freqtrade strategy dedicated to detecting SMC Fair Value Gaps and alerting via Telegram.
    """
    # Processing options
    process_only_new_candles = True
    use_exit_signal = False
    timeframe = '15m'
    informative_timeframe = '5m'
    
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

    # --- Score configuration (0-100, see _compute_pair_score) -----------------
    # Weights express the priority of each component (sum = 1.0).
    score_weight_forming = 0.30   # fresh / forming FVG (immediate trigger)
    score_weight_ob      = 0.35   # confirmed order blocks (rarest, strongest anchor)
    score_weight_fvg     = 0.20   # confirmed FVGs (confluence)
    score_weight_range   = 0.15   # swing_range_percentage (volatility / context)
    # Normalization caps: number in the 12-candle window that awards full credit.
    score_ob_cap   = 2            # OBs  (median 0 / p90 1 on 5m -> cap 2)
    score_fvg_cap  = 5            # FVGs (median 3 / p90 5 on 5m -> cap 5)
    score_range_ref = 1.25        # % swing range giving full credit (≈p90 on 5m)

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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

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
        # 5m timeframe FVG counts (informative)
        # ---------------------------------------------------------------------
        inf_tf = self.informative_timeframe

        # NOTE: freqtrade's merge_informative_pair() (a standalone helper since
        # 2026.7, no longer an IStrategy method) rejects a FASTER informative
        # timeframe - and our 5m informative is faster than the 15m base. So we
        # aggregate the 5m FVG counts into base-timeframe buckets manually.
        informative_df = self.dp.get_pair_dataframe(pair=pair, timeframe=inf_tf)
        if not informative_df.empty:
            fvg_df = smc.fvg(informative_df)

            # A FVG is "confirmed" when fvg() assigns it 1/-1, i.e. the confirming
            # candle (via shift(-1)) exists. FVGs are "unmitigated" when the gap is
            # not yet filled: the library uses MitigatedIndex == 0 for that (positive
            # index = filled by a later candle, NaN = no FVG at that row).
            unmitigated = fvg_df['MitigatedIndex'] == 0
            informative_df['bullish_fvg_count'] = (
                (fvg_df['FVG'] == 1) & unmitigated).astype(int)
            informative_df['bearish_fvg_count'] = (
                (fvg_df['FVG'] == -1) & unmitigated).astype(int)
            base_minutes = int(pd.Timedelta(self.timeframe).total_seconds() // 60)
            aggregated = informative_df.resample(
                f"{base_minutes}min", on='date'
            )[['bullish_fvg_count', 'bearish_fvg_count']].sum().reset_index()
            dataframe = pd.merge(dataframe, aggregated, on='date', how='left')
        else:
            dataframe['bullish_fvg_count'] = 0
            dataframe['bearish_fvg_count'] = 0

        # Merge rows that predate the informative data are NaN -> 0.
        dataframe['bullish_fvg_count'] = dataframe['bullish_fvg_count'].fillna(0)
        dataframe['bearish_fvg_count'] = dataframe['bearish_fvg_count'].fillna(0)
        # Total confirmed, unmitigated FVGs (both directions) per base candle.
        dataframe['fvg_total_count'] = (
            dataframe['bullish_fvg_count'] + dataframe['bearish_fvg_count'])

        #fvg_df = smc.fvg(dataframe)
        # Pine-style (LuxAlgo) swing high/low detection - see pine_swing_highs_lows.
        swing_df = pine_swing_highs_lows(dataframe, size=self.swing_size)
        ob_df = smc.ob(dataframe, swing_df)

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
        fvg_count = int(dataframe['fvg_total_count'].iloc[-2])
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
        Prints the per-pair SMC scoreboard as a table (tabulate), sorted by forming_fvg.
        Skips re-printing when nothing changed since the last loop, to avoid log spam.
        """
        pair_pattern = re.compile('/.*')
        rows = []
        for m in sorted(self.pair_metrics.values(), key=lambda x: ( x['ratio'], x['fvg_count']), reverse=True):
            short_pair = pair_pattern.sub('', m['pair'])
            rows.append([
                short_pair,
                f"{m['ratio']:.2f}",
                f"{m['fvg_count']}",                
                f"{m['high']}/{m['low']}",                
            ])
        table = tabulate(
            rows,
            headers=["Pair", "Ratio", "FVG", "High/Low"],
            tablefmt="github",
        )
        if table == self._last_scoreboard:
            return
        self._last_scoreboard = table
        logger.info(f"SMC Scoreboard ({self.get_ny_time()} {self.timeframe}):\n{table}")

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
