"""
SMC Fair Value Gap (FVG) Notification Strategy for Freqtrade

This strategy detects Fair Value Gaps (FVG) on a 15-minute timeframe using the
`smartmoneyconcepts` library and sends real-time Telegram notifications when a new FVG
forms on the latest closed candle.
"""

import logging
import time
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
        
        fvg_df = smc.fvg(dataframe)
        swing_df = smc.swing_highs_lows(dataframe, swing_length=5)
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
        # Fair Value Gap (FVG) Detection
        # ---------------------------------------------------------------------

        # Merge FVG results back to main dataframe
        dataframe['fvg_type'] = fvg_df['FVG']
        dataframe['fvg_top'] = fvg_df['Top']
        dataframe['fvg_bottom'] = fvg_df['Bottom']
        # Calculate gap size percentage for context in notifications
        dataframe['fvg_percentage'] = (dataframe['fvg_top'] - dataframe['fvg_bottom']) / dataframe['fvg_bottom'] * 100

        dataframe['ob_type'] = ob_df['OB']

        # Count how many occurred within the window (confirmed = both directions)
        dataframe['bullish_fvg_count'] = (dataframe['fvg_type'] == 1).astype(int).rolling(window=n).sum()
        dataframe['bullish_ob_count'] = (dataframe['ob_type'] == 1).astype(int).rolling(window=n).sum()
        dataframe['fvg_total_count'] = (dataframe['fvg_type'].abs() == 1).astype(int).rolling(window=n).sum()
        dataframe['ob_total_count'] = (dataframe['ob_type'].abs() == 1).astype(int).rolling(window=n).sum()

        # Detect forming FVG on c1 and c2 candles, c3 is in progress        
        c1 = dataframe.iloc[-2]
        c2 = dataframe.iloc[-1]

        c1_high = c1['high']
        c1_low = c1['low']
        c2_close = c2['close']

        possible_bullish_fvg = c2_close > c1_high
        #possible_bearish_fvg = c2_close < c1_low
        #forming_fvg = bool(possible_bullish_fvg or possible_bearish_fvg)
        forming_fvg = possible_bullish_fvg
        #if forming_fvg:
            # if c2_close > c1_high:
            #     gap_pct = (c2_close - c1_high) / c1_high * 100
            #     direction = "bullish"
            # else:
            #     gap_pct = (c1_low - c2_close) / c1_low * 100
            #     direction = "bearish"
            #fvg_gap_percentage = (c2_close - c1_high) / c1_high * 100
            #direction = "bullish"
        fvg_gap_percentage = (c2_close - c1_high) / c1_high * 100 if forming_fvg else 0.0

        # ---------------------------------------------------------------------
        # Per-pair scoreboard metric
        # ---------------------------------------------------------------------
        ob_count = int(dataframe['ob_total_count'].iloc[-1])
        fvg_count = int(dataframe['fvg_total_count'].iloc[-1])
        ratio = float(dataframe['swing_range_percentage'].iloc[-1])
        score = self._compute_pair_score(forming_fvg, ob_count, fvg_count, ratio)
        self.pair_metrics[pair] = {
            'pair': pair,
            'score': round(score, 2),
            'forming_fvg': fvg_gap_percentage,
            'ob_count': ob_count,
            'fvg_count': fvg_count,
            'ratio': round(ratio, 2),
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

    def _compute_pair_score(self, forming_fvg: bool, ob_count: int, fvg_count: int, ratio: float) -> float:
        """
        Weighted score in [0, 100].

        Each component is normalized to [0, 1] before being multiplied by its weight:
          - forming FVG (0 or 1)                            * 0.30  immediate trigger
          - min(ob_count / score_ob_cap, 1)   confirmed OBs * 0.35  strongest anchor
          - min(fvg_count / score_fvg_cap, 1) confirmed FVGs * 0.20  confluence
          - min(ratio / score_range_ref, 1)   swing range % * 0.15  volatility context

        A score near 100 only happens when ALL components are strong, matching the
        "textbook strongest setup" (OB + confirmed FVGs + forming FVG + wide range).
        """
        c_forming = 1.0 if forming_fvg else 0.0
        c_ob = min(ob_count / self.score_ob_cap, 1.0)
        c_fvg = min(fvg_count / self.score_fvg_cap, 1.0)
        c_range = min(ratio / self.score_range_ref, 1.0) if ratio == ratio else 0.0
        return 100.0 * (
            c_forming * self.score_weight_forming
            + c_ob * self.score_weight_ob
            + c_fvg * self.score_weight_fvg
            + c_range * self.score_weight_range
        )

    def _print_scoreboard(self) -> None:
        """
        Prints the per-pair SMC scoreboard as a table (tabulate), sorted by score.
        Skips re-printing when nothing changed since the last loop, to avoid log spam.
        """
        rows = []
        for m in sorted(self.pair_metrics.values(), key=lambda x: x['score'], reverse=True):
            rows.append([
                m['pair'],
                f"{m['score']:.2f}",
                f"{m['forming_fvg']:.2f}",
                m['ob_count'],
                m['fvg_count'],
                f"{m['ratio']:.2f}",
            ])
        table = tabulate(
            rows,
            headers=["Pair", "Score", "Forming FVG", "OBs", "FVGs", "Ratio"],
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
