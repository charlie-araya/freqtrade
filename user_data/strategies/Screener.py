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


def _format_mcap(mcap: float | None) -> str:
    """Format a CoinGecko market cap for the scoreboard. Reuses volume formatting."""
    return _format_volume(mcap)


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

class Screener(IStrategy):
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

    # Highlighting tiers for top coins by Ratio (terminal only, ANSI). 0 disables.
    # Tier 1: yellow background, black font for the very top coins.
    top_ratio_tier1_count = 10
    # Tier 2: white background, black font for the next coins after tier 1.
    top_ratio_tier2_count = 20

    # Scoreboard sort order: 'ratio' or 'volume'. Defaults to 'ratio'.
    scoreboard_sort_order = 'ratio'

    plot_config = {
        "main_plot": {
            "trama": {},
            "hma": {},
        },
        "subplots": {
            "MACD": {
                "macd": {},
                "macdsignal": {},
            },
        },
    }

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
        self._next_scoreboard_print = now + 0.5 * 60.0
        # Scoreboard sort order resolved from config/env (default 'ratio').
        so = (os.environ.get('FVG_SCOREBOARD_SORT_ORDER')
              or self.config.get('scoreboard_sort_order')
              or self.scoreboard_sort_order
              or 'ratio')
        so = str(so).strip().lower()
        self._scoreboard_sort_order = 'volume' if so == 'volume' else 'ratio'
        # Base-candle timestamps of the last HMA/TRAMA and MACD cross alerts per
        # pair, so a single cross produces exactly one notification.
        self._last_hma_cross_alert: dict[str, pd.Timestamp] = {}
        self._last_macd_cross_alert: dict[str, pd.Timestamp] = {}
        # CoinGecko market caps per pair (e.g. {"BTC/USDT": 2.1e12}), loaded once
        # at startup via _load_market_caps_once() and reused by _print_scoreboard.
        self._market_caps: dict[str, float] = {}
        self._market_caps_loaded: bool = False
        # Cached GET /coins/list result: symbol.lower() -> [coingecko ids].
        # Cached so later batches for newly-seen pairs don't refetch the list.
        self._coingecko_symbol_to_ids: dict[str, list[str]] | None = None
        # Base symbols already proven unresolvable (no id, or markets returned
        # no mcap). Skipped on later loops to avoid log/API spam; the cached
        # /coins/list won't change within one bot run anyway.
        self._mcap_unresolved_bases: set[str] = set()
        self._mcap_probed: bool = False

    # ------------------------------------------------------------------
    # CoinGecko market caps (loaded once at startup, reused by scoreboard)
    # ------------------------------------------------------------------
    def _coingecko_base_url(self) -> str:
        """CoinGecko API base URL from env, defaulting to the public v3 API."""
        url = (
            os.environ.get('COINGECKO_API_URL')
            or os.environ.get('COINGECKO_BASE_URL')
            or os.environ.get('CG_API_URL')
            or "https://api.coingecko.com/api/v3"
        )
        return str(url).strip().rstrip('/')

    def _coingecko_api_key(self) -> str | None:
        """CoinGecko API key from env (demo or pro). Returns None when unset."""
        key = (
            os.environ.get('COINGECKO_API_KEY')
            or os.environ.get('COINGECKO_DEMO_API_KEY')
            or os.environ.get('CG_API_KEY')
            or os.environ.get('X_CG_DEMO_API_KEY')
            or os.environ.get('X_CG_PRO_API_KEY')
        )
        if key is not None:
            key = str(key).strip()
            return key or None
        return None

    def _coingecko_headers(self) -> dict:
        """Auth header for CoinGecko - exactly ONE header matching the base URL.

        Sending a demo key in the pro header slot (or vice versa) makes
        CoinGecko reject even valid /coins/markets calls with HTTP 400, so we
        never send both. pro-api.coingecko.com -> x-cg-pro-api-key, everything
        else -> x-cg-demo-api-key. Override with COINGECKO_KEY_TYPE=demo|pro.
        """
        key = self._coingecko_api_key()
        if not key:
            return {}
        force = (os.environ.get('COINGECKO_KEY_TYPE') or '').strip().lower()
        base = self._coingecko_base_url().lower()
        if force == 'pro' or (not force and 'pro-api' in base):
            return {"x-cg-pro-api-key": key}
        return {"x-cg-demo-api-key": key}

    def _screener_pairs_for_mcap(self) -> list[str]:
        """All pairs whose market cap we want: whitelist + already-seen pairs."""
        pairs: set[str] = set(self.notification_whitelist or [])
        try:
            cfg = self.config or {}
            exch_whitelist = cfg.get('exchange', {}).get('pair_whitelist')
            if isinstance(exch_whitelist, list):
                pairs.update(exch_whitelist)
            cfg_whitelist = cfg.get('pair_whitelist')
            if isinstance(cfg_whitelist, list):
                pairs.update(cfg_whitelist)
        except Exception:
            pass
        try:
            pairs.update(self.pair_metrics.keys())
        except Exception:
            pass
        return sorted(p for p in pairs if isinstance(p, str) and p)

    def _get_coingecko_symbol_to_ids(self) -> dict[str, list[str]] | None:
        """
        GET /coins/list once and cache symbol.lower() -> [ids].
        Returns None on failure (caller logs and aborts that batch).
        """
        if self._coingecko_symbol_to_ids is not None:
            return self._coingecko_symbol_to_ids
        try:
            import requests  # type: ignore
        except ImportError:
            logger.warning("CoinGecko: 'requests' package not installed, market caps unavailable.")
            return None
        base_url = self._coingecko_base_url()
        headers = self._coingecko_headers()
        if not self._coingecko_api_key():
            logger.warning(
                "CoinGecko: no API key found in env (COINGECKO_API_KEY / "
                "COINGECKO_DEMO_API_KEY). Continuing unauthenticated with low rate limits."
            )
        logger.info(
            f"CoinGecko: fetching /coins/list from {base_url} "
            f"(key {'present' if headers else 'MISSING'})."
        )
        try:
            resp = requests.get(f"{base_url}/coins/list", headers=headers, timeout=30)
            resp.raise_for_status()
            coin_list = resp.json()
        except Exception as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = ""
            try:
                body = str(getattr(getattr(e, 'response', None), 'text', ''))[:300]
            except Exception:
                pass
            logger.warning(f"CoinGecko: GET /coins/list failed (status={status}): {e} {body}")
            return None
        symbol_to_ids: dict[str, list[str]] = {}
        try:
            count = 0
            for c in coin_list:
                sym = str(c.get('symbol', '')).strip().lower()
                cid = c.get('id')
                if sym and cid:
                    symbol_to_ids.setdefault(sym, []).append(str(cid))
                    count += 1
            logger.info(f"CoinGecko: /coins/list returned {count} coins ({len(symbol_to_ids)} symbols).")
        except Exception as e:
            logger.warning(f"CoinGecko: could not parse /coins/list response: {e}")
            return None
        self._coingecko_symbol_to_ids = symbol_to_ids
        return symbol_to_ids

    def _probe_coingecko_markets(self) -> None:
        """Minimal 1-id markets call to tell auth/param errors apart from bad ids.

        Logs (once per process) whether bitcoin-only query works with the
        current base URL + header. If it works, the 400 comes from the ids
        list; if it also 400s, the key/header/base-URL combo is wrong.
        """
        if getattr(self, '_mcap_probed', False):
            return
        self._mcap_probed = True
        try:
            import requests  # type: ignore
            base_url = self._coingecko_base_url()
            headers = self._coingecko_headers()
            r = requests.get(
                f"{base_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": "bitcoin",
                    "order": "market_cap_desc",
                    "per_page": 1,
                    "sparkline": "false",
                },
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            logger.info("CoinGecko: probe ids=bitcoin OK - base URL/key/params are fine.")
        except Exception as e:
            resp = getattr(e, 'response', None)
            status = getattr(resp, 'status_code', None)
            body = ""
            try:
                body = str(getattr(resp, 'text', ''))[:500]
            except Exception:
                pass
            header_names = sorted(self._coingecko_headers().keys())
            logger.warning(
                f"CoinGecko: probe ids=bitcoin also failed (status={status}): {e} "
                f"body={body}. base={self._coingecko_base_url()} headers={header_names} "
                "(demo keys only work with https://api.coingecko.com, pro keys only "
                "with https://pro-api.coingecko.com; check COINGECKO_KEY_TYPE)."
            )

    def _fetch_mcaps_for_bases(self, bases: list[str]) -> dict[str, float]:
        """
        Batch GET /coins/markets?ids=... for the given base symbols.
        Resolves each base via the cached /coins/list map, queries markets in
        small chunks (<=50 ids so the URL stays short and one bad chunk can't
        fail the whole batch), and returns base.lower() -> best market_cap.
        Ambiguous symbols (many ids share e.g. 's', 'w') include every id and
        keep the highest market_cap, which is the most-traded token.
        """
        import requests  # type: ignore
        symbol_to_ids = self._get_coingecko_symbol_to_ids()
        if not symbol_to_ids:
            return {}
        wanted = [b.strip().lower() for b in bases if b and b.strip()]
        wanted = sorted(set(wanted))
        if not wanted:
            return {}
        try:
            chunk_size = int(os.environ.get('COINGECKO_MARKETS_CHUNK_SIZE', '50'))
        except Exception:
            chunk_size = 50
        chunk_size = max(10, min(100, chunk_size))
        try:
            max_ids_per_symbol = int(os.environ.get('COINGECKO_MAX_IDS_PER_SYMBOL', '25'))
        except Exception:
            max_ids_per_symbol = 25
        max_ids_per_symbol = max(5, min(100, max_ids_per_symbol))
        candidate_ids: list[str] = []
        missing: list[str] = []
        dup_info: dict[str, int] = {}
        truncated: dict[str, int] = {}
        for base in wanted:
            ids = symbol_to_ids.get(base)
            if not ids:
                missing.append(base.upper())
            else:
                dup_info[base] = len(ids)
                if len(ids) > max_ids_per_symbol:
                    truncated[base] = len(ids)
                    candidate_ids.extend(ids[:max_ids_per_symbol])
                else:
                    candidate_ids.extend(ids)
        if missing:
            logger.warning(f"CoinGecko: no CoinGecko id found for symbols: {sorted(missing)}")
        if truncated:
            logger.warning(
                "CoinGecko: capping ids per symbol to "
                f"{max_ids_per_symbol} (raise via COINGECKO_MAX_IDS_PER_SYMBOL): {truncated}."
            )
        noisy = {k: v for k, v in dup_info.items() if v > 20}
        if noisy:
            logger.warning(
                "CoinGecko: highly ambiguous symbols (many ids share the ticker, "
                f"keeping highest mcap each): {noisy}. Results for these may be approximate."
            )
        seen: set[str] = set()
        uniq_ids: list[str] = []
        for cid in candidate_ids:
            if cid not in seen:
                seen.add(cid)
                uniq_ids.append(cid)
        if not uniq_ids:
            return {}
        base_url = self._coingecko_base_url()
        headers = self._coingecko_headers()
        wanted_set = set(wanted)
        best: dict[str, float] = {}
        n_chunks = (len(uniq_ids) + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            chunk = uniq_ids[ci * chunk_size:(ci + 1) * chunk_size]
            # NOTE: sparkline must be lowercase "false" string - Python False
            # encodes as "False" and CoinGecko rejects it with HTTP 400.
            params = {
                "vs_currency": "usd",
                "ids": ",".join(chunk),
                "order": "market_cap_desc",
                "per_page": len(chunk),
                "sparkline": "false",
            }
            if n_chunks > 1:
                logger.info(
                    f"CoinGecko: /coins/markets chunk {ci + 1}/{n_chunks} ({len(chunk)} ids)."
                )
            try:
                r2 = requests.get(
                    f"{base_url}/coins/markets", params=params, headers=headers, timeout=30
                )
                r2.raise_for_status()
            except Exception as e:
                resp = getattr(e, 'response', None)
                status = getattr(resp, 'status_code', None)
                body = ""
                try:
                    body = str(getattr(resp, 'text', ''))[:500]
                except Exception:
                    pass
                logger.warning(
                    f"CoinGecko: /coins/markets chunk {ci + 1}/{n_chunks} failed "
                    f"(status={status}, {len(chunk)} ids): {e} body={body}. "
                    "Continuing with other chunks."
                )
                if ci == 0 and status == 400:
                    # All chunks share the same params/headers, so a 400 on the
                    # very first chunk is systematic (bad header/key/param), not
                    # a bad id. Probe with a single known id to isolate it.
                    self._probe_coingecko_markets()
                continue
            try:
                payload = r2.json()
            except Exception as e:
                logger.warning(f"CoinGecko: could not parse /coins/markets chunk {ci + 1}: {e}")
                continue
            for m in payload:
                try:
                    sym = str(m.get('symbol', '')).strip().lower()
                    mcap = m.get('market_cap')
                except Exception:
                    continue
                if not sym or sym not in wanted_set or mcap is None:
                    continue
                try:
                    mcap_f = float(mcap)
                except (TypeError, ValueError):
                    continue
                if sym not in best or mcap_f > best[sym]:
                    best[sym] = mcap_f
        unresolved = sorted(b.upper() for b in wanted if b not in best)
        if unresolved:
            logger.warning(
                f"CoinGecko: /coins/markets returned no market_cap for: {unresolved}. "
                "Ticker is likely ambiguous or the coin is untracked."
            )
        return best

    def _load_market_caps_once(self, force: bool = False) -> None:
        """
        Fetch CoinGecko market caps for the startup universe and cache them in
        ``self._market_caps`` (pair -> market_cap in USD).

        Flow:
          1. GET /coins/list -> map each screener base symbol (e.g. BTC from
             BTC/USDT) to CoinGecko's internal id.
          2. Single GET /coins/markets?ids=<comma-joined ids> -> market_cap.
        Results are reused by _print_scoreboard; failures only log warnings
        and never raise, so the strategy keeps running without Mcap data.
        Pairs discovered later (pair_metrics) are filled in by
        _ensure_market_caps_for_current_pairs(), which reuses the cached
        /coins/list and only queries markets for the missing symbols.
        """
        if self._market_caps_loaded and not force:
            return
        pairs = self._screener_pairs_for_mcap()
        if not pairs:
            logger.warning("CoinGecko: no screener pairs found, skipping market-cap load.")
            return
        base_to_pairs: dict[str, list[str]] = {}
        for pair in pairs:
            try:
                base = re.split(r'[/:]', pair)[0].strip().lower()
            except Exception:
                continue
            if base:
                base_to_pairs.setdefault(base, []).append(pair)
        try:
            best = self._fetch_mcaps_for_bases(list(base_to_pairs.keys()))
        except Exception as e:
            logger.warning(f"CoinGecko: market-cap load failed: {e}")
            return
        loaded = 0
        for base, plist in base_to_pairs.items():
            mcap = best.get(base)
            if mcap is not None:
                for p in plist:
                    self._market_caps[p] = mcap
                    loaded += 1
            else:
                self._mcap_unresolved_bases.add(base)
        self._market_caps_loaded = True
        logger.info(
            f"CoinGecko: loaded market caps for {loaded}/{len(pairs)} pairs "
            f"({len(best)} symbols with mcap)."
        )

    def _ensure_market_caps_for_current_pairs(self) -> None:
        """
        Fill gaps for pairs seen in pair_metrics after startup (the scoreboard
        universe is dynamic). Only the missing base symbols are queried, in one
        /coins/markets batch; everything already cached is reused untouched.
        Symbols proven unresolvable are skipped via _mcap_unresolved_bases so a
        bot loop every few seconds does not spam the API/logs.
        """
        try:
            missing_pairs = [p for p in self.pair_metrics if p not in self._market_caps]
        except Exception:
            return
        if not missing_pairs:
            return
        base_to_pairs: dict[str, list[str]] = {}
        for pair in missing_pairs:
            try:
                base = re.split(r'[/:]', pair)[0].strip().lower()
            except Exception:
                continue
            if base and base not in self._mcap_unresolved_bases:
                base_to_pairs.setdefault(base, []).append(pair)
        if not base_to_pairs:
            return
        logger.info(
            f"CoinGecko: fetching market caps for {len(missing_pairs)} newly-seen pairs "
            f"({len(base_to_pairs)} symbols)."
        )
        try:
            best = self._fetch_mcaps_for_bases(list(base_to_pairs.keys()))
        except Exception as e:
            logger.warning(f"CoinGecko: gap-fill market-cap load failed: {e}")
            return
        n = 0
        for base, plist in base_to_pairs.items():
            mcap = best.get(base)
            if mcap is not None:
                for p in plist:
                    self._market_caps[p] = mcap
                    n += 1
            else:
                # No id or no mcap with the cached /coins/list -> retrying every
                # loop would never succeed, so remember and stay quiet next time.
                self._mcap_unresolved_bases.add(base)
        if n:
            logger.info(f"CoinGecko: gap-filled market caps for {n}/{len(missing_pairs)} pairs.")

    def bot_start(self, **kwargs) -> None:
        """Called once after bot instantiation: fetch market caps a single time."""
        try:
            self._load_market_caps_once()
        except Exception as e:
            logger.warning(f"CoinGecko: unexpected error loading market caps at startup: {e}")

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
        swing_df = pine_swing_highs_lows(dataframe, self.swing_size)
        #swing_df = swing_highs_lows(dataframe, self.swing_size)


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
        dataframe['swing_range_percentage'] = dataframe['swing_range_percentage'].fillna(0.0).replace([float('inf'), float('-inf')], 0.0)

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
        if self._scoreboard_sort_order == 'volume':
            sorted_metrics = sorted(
                self.pair_metrics.values(),
                key=lambda x: (self._vol24h.get(x['pair']) or 0.0, x.get('ratio') or 0.0),
                reverse=True,
            )
        else:
            sorted_metrics = sorted(
                self.pair_metrics.values(),
                key=lambda x: (x.get('ratio') or 0.0, self._vol24h.get(x['pair']) or 0.0),
                reverse=True,
            )
        for m in sorted_metrics:
            short_pair = re.sub('/.*', '', m['pair'])
            rows.append([
                short_pair,
                f"{m['ratio']:.2f}",
                _format_mcap(self._market_caps.get(m['pair'])),
                _format_volume(self._vol24h.get(m['pair'])),                
                f"{m['high']}/{m['low']}",

            ])

        if rows:
            frame = tabulate(
                rows,
                headers=["Pair", "Ratio", "Mcap", "24H Vol","High/Low"],
                tablefmt="github",
            )
            #frame = self._apply_ratio_highlight(frame, rows)

            if frame != self._last_scoreboard:
                self._last_scoreboard = frame
                logger.info(f"SMC Scoreboard ({self.get_ny_time()} {self.timeframe}):\n{frame}")

    def _apply_ratio_highlight(self, frame: str, rows: list) -> str:
        """
        Highlights the Ratio cell (terminal only) for the top coins sorted by Ratio:
        - Tier 1 (top N): yellow background, black font.
        - Tier 2 (next M): white background, black font.
        Log files are left unchanged.
        """
        tier1 = int(self.config.get('top_ratio_tier1_count', self.top_ratio_tier1_count))
        tier2 = int(self.config.get('top_ratio_tier2_count', self.top_ratio_tier2_count))
        if tier1 <= 0 and tier2 <= 0:
            return frame
        try:
            if not (sys.stdout.isatty() or sys.stderr.isatty()):
                return frame
        except Exception:
            return frame

        ranked = sorted(
            (m for m in self.pair_metrics.values()
             if isinstance(m.get('ratio'), (int, float)) and m.get('ratio') == m.get('ratio')),
            key=lambda m: m['ratio'],
            reverse=True,
        )
        top_by_tier: dict[str, str] = {}
        for i, m in enumerate(ranked):
            short = re.sub('/.*', '', m['pair'])
            if i < tier1 and tier1 > 0:
                top_by_tier[short] = 'Y'
            elif tier1 <= i < tier1 + tier2 and tier2 > 0:
                top_by_tier[short] = 'W'

        if not top_by_tier:
            return frame

        ansi_y = "\x1b[43;30m"  # yellow bg, black fg
        ansi_w = "\x1b[47;30m"  # white bg, black fg
        ansi_z = "\x1b[0m"
        out = []
        for line in frame.split('\n'):
            if not line.startswith('| ') or line.startswith('|-'):
                out.append(line)
                continue
            cells = line.split('|')
            body = [c for c in cells[1:-1]]
            if len(body) < 4 or body[0].strip() not in top_by_tier:
                out.append(line)
                continue
            ratio_cell = body[2]
            pad_l = len(ratio_cell) - len(ratio_cell.lstrip(' '))
            pad_r = len(ratio_cell) - len(ratio_cell.rstrip(' '))
            color = ansi_y if top_by_tier[body[0].strip()] == 'Y' else ansi_w
            new_body = []
            for i, c in enumerate(body):
                if i == 2:
                    new_body.append((' ' * pad_l) + color + ratio_cell.strip() + ansi_z + (' ' * pad_r))
                else:
                    new_body.append(c)
            out.append('|'.join([''] + new_body + ['']))
            continue
        return '\n'.join(out)

    def bot_loop_start(self, **kwargs) -> None:
        """
        Executed once at the beginning of each bot iteration, AFTER all pair 
        data frames have finished running through populate_indicators().
        """
        #if not self.pending_smc_alerts:
        #    return
        
        if self.dp and self.dp.runmode.value in ('live', 'dry_run'):
            # Startup batch runs once in bot_start; the scoreboard universe is
            # dynamic (pair_metrics fills in later), so gap-fill any newly-seen
            # pairs here. Cached values are reused; only missing symbols trigger
            # one extra /coins/markets batch.
            try:
                if not self._market_caps_loaded:
                    self._load_market_caps_once()
                self._ensure_market_caps_for_current_pairs()
            except Exception as e:
                logger.warning(f"CoinGecko: lazy market-cap load failed: {e}")
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
 