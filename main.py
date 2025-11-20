import os
import time
import json
import logging
from datetime import datetime, timezone

import requests
import gspread
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Config / Environment
# -------------------------

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")

if not OANDA_API_KEY or not OANDA_ACCOUNT_ID or not GOOGLE_CREDS_JSON:
    raise RuntimeError("Missing required env vars: OANDA_API_KEY, OANDA_ACCOUNT_ID, GOOGLE_CREDS_JSON")

# practice | live
OANDA_ENV = os.getenv("OANDA_ENV", "practice").lower()  # "practice" or "live"

# Google Sheet names
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "Active-Investing")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Oanda-Trader")

# Risk / trade logic
# Values are percentages, e.g. 0.3 = 0.3% move on price
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.3"))  # hard stop, e.g. 0.3%

# New trailing logic:
# TRAIL_GIVEBACK_FRACTION: fraction of peak profit we allow to be given back, e.g. 0.3 = 30%
# MIN_TRAIL_DROP_PCT: minimum absolute drawdown between ATH and exit, e.g. 0.2 = 0.2%
TRAIL_GIVEBACK_FRACTION = float(os.getenv("TRAIL_GIVEBACK_FRACTION", "0.3"))
MIN_TRAIL_DROP_PCT = float(os.getenv("MIN_TRAIL_DROP_PCT", "0.2"))

# Approximate commission settings.
# This is a *rough* approximation for accounts with a commission structure,
# expressed as cost per 1,000,000 units (per side) in account/quote currency.
# Set to 0.0 if your account is spread-only.
COMMISSION_PER_MILLION = float(os.getenv("COMMISSION_PER_MILLION", "50.0"))

# Loop interval
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

# Closed-trades log starts in this column (side-by-side)
CLOSED_TRADES_START_COL = os.getenv("CLOSED_TRADES_START_COL", "K")  # default: column K

OPEN_HEADER = [
    "Instrument",
    "Side",
    "Units",
    "EntryPrice",
    "CurrentPrice",
    "ProfitPct",
    "Armed",
    "AllTimeHighPct",
    "LastUpdatedUTC",
]

CLOSED_HEADER = [
    "ClosedInstrument",
    "ClosedSide",
    "ClosedUnits",
    "ClosedEntryPrice",
    "ClosedExitPrice",
    "ClosedProfitPct",
    "ClosedArmed",
    "CloseReason",            # e.g. STOP_LOSS, TRAILING_GIVEBACK
    "ClosedAllTimeHighPct",
    "ClosedAtUTC",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info(
    f"Using risk settings: "
    f"STOP_LOSS_PCT={STOP_LOSS_PCT}%, "
    f"TRAIL_GIVEBACK_FRACTION={TRAIL_GIVEBACK_FRACTION}, "
    f"MIN_TRAIL_DROP_PCT={MIN_TRAIL_DROP_PCT}%, "
    f"COMMISSION_PER_MILLION={COMMISSION_PER_MILLION}, "
    f"POLL_INTERVAL_SECONDS={POLL_INTERVAL_SECONDS}"
)

# -------------------------
# Oanda API helpers
# -------------------------


def oanda_base_url() -> str:
    if OANDA_ENV == "live":
        return "https://api-fxtrade.oanda.com/v3"
    else:
        return "https://api-fxpractice.oanda.com/v3"


def oanda_headers() -> dict:
    return {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }


def fetch_current_prices(instruments):
    """
    Fetch bid/ask prices for a list of instruments.
    Returns: {instrument: {"bid": float, "ask": float}}
    """
    if not instruments:
        return {}

    url = f"{oanda_base_url()}/accounts/{OANDA_ACCOUNT_ID}/pricing"
    params = {
        "instruments": ",".join(instruments)
    }
    resp = requests.get(url, headers=oanda_headers(), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    prices_map = {}
    for p in data.get("prices", []):
        instrument = p["instrument"]
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        if not bids or not asks:
            continue
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])

        prices_map[instrument] = {
            "bid": bid,
            "ask": ask,
        }

    return prices_map


def fetch_open_positions():
    """
    Returns a list of position dicts:
    {
        "instrument": str,
        "side": "LONG" | "SHORT",
        "units": float,
        "entry_price": float,
        "current_price": float,
        "profit_pct": float,   # approximate, net of estimated commission
    }
    """
    url = f"{oanda_base_url()}/accounts/{OANDA_ACCOUNT_ID}/openPositions"
    resp = requests.get(url, headers=oanda_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()

    raw_positions = data.get("positions", [])
    if not raw_positions:
        return []

    # Collect instruments for pricing lookup
    instruments = sorted({p["instrument"] for p in raw_positions})
    prices_map = fetch_current_prices(instruments)

    positions = []

    for p in raw_positions:
        instrument = p["instrument"]
        price_info = prices_map.get(instrument)
        if price_info is None:
            logging.warning(f"No current price for instrument {instrument}, skipping.")
            continue

        for side_key, side_label in (("long", "LONG"), ("short", "SHORT")):
            side_data = p.get(side_key)
            if not side_data:
                continue

            units = float(side_data.get("units", "0"))
            if units == 0:
                continue

            entry_price = float(side_data["averagePrice"])

            # Use side-appropriate price (bid for longs, ask for shorts)
            if side_label == "LONG":
                current_price = price_info["bid"]
                raw_profit_pct = (current_price - entry_price) / entry_price * 100.0
            else:  # SHORT
                current_price = price_info["ask"]
                raw_profit_pct = (entry_price - current_price) / entry_price * 100.0

            # --- Approximate commission impact ---
            # We estimate commission already paid on the opening side as:
            # commission_cash = COMMISSION_PER_MILLION * (abs(units) / 1_000_000)
            # and express it as a percent of notional (units * entry_price).
            notional = abs(units) * entry_price
            commission_pct = 0.0
            if COMMISSION_PER_MILLION > 0.0 and notional > 0.0:
                commission_cash = COMMISSION_PER_MILLION * (abs(units) / 1_000_000.0)
                commission_pct = (commission_cash / notional) * 100.0

            profit_pct = raw_profit_pct - commission_pct

            positions.append(
                {
                    "instrument": instrument,
                    "side": side_label,
                    "units": units,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "profit_pct": profit_pct,
                }
            )

    return positions


def close_position(instrument: str, side: str):
    """
    Close entire position for an instrument and side.
    """
    url = f"{oanda_base_url()}/accounts/{OANDA_ACCOUNT_ID}/positions/{instrument}/close"

    if side.upper() == "LONG":
        payload = {"longUnits": "ALL"}
    else:
        payload = {"shortUnits": "ALL"}

    logging.info(f"Closing {side} position for {instrument}")
    resp = requests.put(url, headers=oanda_headers(), data=json.dumps(payload), timeout=10)
    try:
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Error closing position {instrument} {side}: {e} - {resp.text}")
        raise
    logging.info(f"Successfully closed {instrument} {side}. Response: {resp.text}")


# -------------------------
# Google Sheets helpers
# -------------------------


def _col_to_index(col_letter: str) -> int:
    """
    Convert a column letter (e.g. 'A', 'K') to a 1-based index.
    """
    col_letter = col_letter.upper()
    idx = 0
    for c in col_letter:
        idx = idx * 26 + (ord(c) - ord("A") + 1)
    return idx


def _index_to_col(index: int) -> str:
    """
    Convert a 1-based column index to letters (e.g. 1 -> 'A', 11 -> 'K').
    """
    result = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        result.append(chr(rem + ord("A")))
    return "".join(reversed(result))


def get_gspread_client():
    """
    Build a gspread client from the GOOGLE_CREDS_JSON env var.

    IMPORTANT:
    - Drive scope is added so gspread can search by spreadsheet title
      (client.open(SPREADSHEET_NAME)) without 403 errors.
    """
    creds_info = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def init_worksheet(ws):
    """
    Ensure the worksheet has:
    - Open positions header in A1:I1
    - Closed trades header starting at CLOSED_TRADES_START_COL row 1
    """
    # Open positions header (A1:I1)
    try:
        open_header_row = ws.row_values(1)
    except Exception as e:
        logging.warning(f"Could not read row 1 for init: {e}")
        open_header_row = []

    if not open_header_row:
        logging.info("Initializing open-positions header in A1.")
        ws.update("A1", [OPEN_HEADER])

    # Closed trades header (e.g. K1:??1)
    start_col = CLOSED_TRADES_START_COL
    start_idx = _col_to_index(start_col)
    end_idx = start_idx + len(CLOSED_HEADER) - 1
    end_col = _index_to_col(end_idx)
    header_range = f"{start_col}1:{end_col}1"

    try:
        closed_header_values = ws.get(header_range)
    except Exception as e:
        logging.warning(f"Could not read closed-trades header row: {e}")
        closed_header_values = []

    # If that range is empty or first row blank, write header
    if not closed_header_values or not closed_header_values[0]:
        logging.info(f"Initializing closed-trades header in {header_range}.")
        ws.update(header_range, [CLOSED_HEADER])


def get_or_create_worksheet(client):
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=2000, cols=30)

    init_worksheet(ws)
    return ws


def load_prior_state(ws):
    """
    Reads open-positions snapshot in columns A..I (all rows) and builds:
        key -> {"armed": bool, "ath_pct": float}
    where key = f"{Instrument}|{Side}"
    """
    try:
        values = ws.get_all_values()
    except Exception as e:
        logging.warning(f"Could not read existing sheet state: {e}")
        return {}

    if not values:
        return {}

    # Only consider the first len(OPEN_HEADER) columns (A..I) for state.
    header_row = values[0][:len(OPEN_HEADER)]
    col_idx = {name: i for i, name in enumerate(header_row)}

    def safe_get(row, name, default=""):
        idx = col_idx.get(name)
        if idx is None:
            return default
        if idx >= len(row):
            return default
        return row[idx]

    state = {}
    for row in values[1:]:
        row = row[:len(OPEN_HEADER)]
        instr = safe_get(row, "Instrument")
        side = safe_get(row, "Side")
        if not instr or not side:
            continue

        key = f"{instr}|{side}"

        armed_raw = str(safe_get(row, "Armed", "")).strip().lower()
        armed = armed_raw in ("true", "1", "yes", "y")

        ath_raw = safe_get(row, "AllTimeHighPct", 0)
        try:
            ath_pct = float(ath_raw) if ath_raw not in ("", None) else 0.0
        except ValueError:
            ath_pct = 0.0

        state[key] = {"armed": armed, "ath_pct": ath_pct}

    return state


def write_positions_to_sheet(ws, positions, state_map):
    """
    positions: list of current open positions (after selling logic)
    state_map: key -> {"armed": bool, "ath_pct": float} for those still open

    Writes the live-snapshot table in A..I, without touching the closed-trades
    log on the right.
    """
    now_str = datetime.now(timezone.utc).isoformat()

    rows = [OPEN_HEADER]
    for pos in positions:
        key = f"{pos['instrument']}|{pos['side']}"
        state = state_map.get(key, {"armed": False, "ath_pct": 0.0})

        rows.append([
            pos["instrument"],
            pos["side"],
            pos["units"],
            round(pos["entry_price"], 6),
            round(pos["current_price"], 6),
            round(pos["profit_pct"], 4),
            str(state["armed"]),
            round(state["ath_pct"], 4),
            now_str,
        ])

    # Determine how many existing rows are in the open-positions area (column A)
    try:
        existing_rows = len(ws.col_values(1))  # number of non-empty rows in column A
    except Exception as e:
        logging.warning(f"Could not determine existing open-positions rows: {e}")
        existing_rows = 0

    # Clear only the existing open-positions block in A..I (rows 2..existing_rows)
    if existing_rows > 1:
        try:
            ws.batch_clear([f"A2:I{existing_rows}"])
        except AttributeError:
            # Fallback if batch_clear isn't available
            blank_rows = [[""] * len(OPEN_HEADER)] * (existing_rows - 1)
            ws.update(f"A2:I{existing_rows}", blank_rows)

    # Rewrite header + open positions
    if rows:
        ws.update("A1", rows, value_input_option="USER_ENTERED")


def log_closed_trade(ws, pos, armed: bool, ath_pct: float, close_reason: str):
    """
    Append a closed-trade row into the log area on the right side of the sheet.

    pos: position dict from fetch_open_positions()
    armed: whether the position has ever been > 0%
    ath_pct: all-time-high profit percentage
    close_reason: 'STOP_LOSS' or 'TRAILING_GIVEBACK'
    """
    closed_at = datetime.now(timezone.utc).isoformat()

    row = [
        pos["instrument"],
        pos["side"],
        pos["units"],
        round(pos["entry_price"], 6),
        round(pos["current_price"], 6),   # snapshot exit price
        round(pos["profit_pct"], 4),
        str(armed),
        close_reason,
        round(ath_pct, 4),
        closed_at,
    ]

    start_col = CLOSED_TRADES_START_COL
    start_idx = _col_to_index(start_col)
    end_idx = start_idx + len(CLOSED_HEADER) - 1
    end_col = _index_to_col(end_idx)

    table_range = f"{start_col}1:{end_col}1"

    ws.append_row(
        row,
        table_range=table_range,
        value_input_option="USER_ENTERED",
    )


# -------------------------
# Trading logic
# -------------------------


def apply_trading_logic(positions, prior_state, ws):
    """
    New trailing logic with closed-trade logging:

    - Always enforce a hard STOP_LOSS_PCT.
    - Track all-time-high profit pct (ath_pct) for each position side.
    - Once a position has been in positive territory (ath_pct > 0),
      allow it to give back only a fraction of that peak profit before closing:
         allowed_drawdown = max(MIN_TRAIL_DROP_PCT, ath_pct * TRAIL_GIVEBACK_FRACTION)
         exit_level       = ath_pct - allowed_drawdown
      If current profit_pct <= exit_level, close the position.

    On every close, log to the closed-trades section with:
    - profit_pct at close
    - armed (ever > 0%)
    - close_reason: 'STOP_LOSS' or 'TRAILING_GIVEBACK'
    """
    remaining_positions = []
    new_state = {}

    for pos in positions:
        key = f"{pos['instrument']}|{pos['side']}"
        prev = prior_state.get(key, {"armed": False, "ath_pct": 0.0})

        prev_ath = float(prev.get("ath_pct", 0.0))
        profit_pct = pos["profit_pct"]

        # Whether this position has ever been positive based on prior ATH
        previously_armed = prev_ath > 0.0 or bool(prev.get("armed", False))

        # 1) Hard stop loss always active
        if profit_pct <= -STOP_LOSS_PCT:
            logging.info(
                f"{key}: Profit {profit_pct:.2f}% <= -STOP_LOSS_PCT (-{STOP_LOSS_PCT}%), "
                f"triggering STOP LOSS sell."
            )
            try:
                # Log closed trade BEFORE sending close request
                log_closed_trade(ws, pos, previously_armed, prev_ath, "STOP_LOSS")
            except Exception as e:
                logging.error(f"Failed to log STOP_LOSS close for {key}: {e}")

            close_position(pos["instrument"], pos["side"])
            continue

        # 2) Update ATH: only track positive territory
        ath_pct = prev_ath
        if profit_pct > 0.0 and profit_pct > prev_ath:
            ath_pct = profit_pct
            logging.info(f"{key}: New all-time high profit {ath_pct:.2f}%.")

        # Determine if position has ever been positive
        has_been_positive = ath_pct > 0.0

        # 3) If never positive, just leave it open with hard SL only
        if not has_been_positive:
            remaining_positions.append(pos)
            new_state[key] = {
                "armed": False,   # has not been > 0% yet
                "ath_pct": ath_pct,
            }
            continue

        # 4) Percentage giveback trailing
        allowed_drawdown = max(
            MIN_TRAIL_DROP_PCT,
            ath_pct * TRAIL_GIVEBACK_FRACTION,
        )
        exit_level = ath_pct - allowed_drawdown

        if profit_pct <= exit_level:
            logging.info(
                f"{key}: Profit {profit_pct:.2f}% <= ATH({ath_pct:.2f}%) - "
                f"allowed_drawdown({allowed_drawdown:.2f}%), triggering "
                f"TRAILING GIVEBACK sell at level {exit_level:.2f}%."
            )
            try:
                log_closed_trade(ws, pos, True, ath_pct, "TRAILING_GIVEBACK")
            except Exception as e:
                logging.error(f"Failed to log TRAILING_GIVEBACK close for {key}: {e}")

            close_position(pos["instrument"], pos["side"])
            continue

        # 5) Still open; record updated state
        remaining_positions.append(pos)
        new_state[key] = {
            "armed": True,      # has been > 0% at some point
            "ath_pct": ath_pct,
        }

    return remaining_positions, new_state


# -------------------------
# Main loop
# -------------------------


def main_loop():
    logging.info("Starting Oanda-Trader loop with percentage-giveback trailing logic.")

    client = get_gspread_client()
    ws = get_or_create_worksheet(client)

    while True:
        try:
            logging.info("Fetching open positions from Oanda...")
            positions = fetch_open_positions()

            if not positions:
                logging.info("No open positions. Updating empty sheet snapshot.")
                write_positions_to_sheet(ws, [], {})
            else:
                logging.info(f"Found {len(positions)} open position side(s).")

                prior_state = load_prior_state(ws)
                remaining_positions, new_state = apply_trading_logic(positions, prior_state, ws)

                logging.info(f"{len(remaining_positions)} position side(s) remain open after logic.")
                write_positions_to_sheet(ws, remaining_positions, new_state)

        except Exception as e:
            logging.exception(f"Error in main loop: {e}")

        logging.info(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
