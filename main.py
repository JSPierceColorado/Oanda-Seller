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
# STOP_LOSS_PCT is the env var you requested (positive number, e.g. 5 for -5%)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "5"))

# Arming and trailing TP thresholds (can be env tuned if you like)
ARM_THRESHOLD_PCT = float(os.getenv("ARM_THRESHOLD_PCT", "5"))   # +5% -> becomes armed
TRAIL_OFFSET_PCT = float(os.getenv("TRAIL_OFFSET_PCT", "3"))     # sell when 3% below ATH

# Loop interval
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
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


def fetch_open_positions():
    """
    Returns a list of position dicts:
    {
        "instrument": str,
        "side": "LONG" | "SHORT",
        "units": float,
        "entry_price": float,
        "current_price": float,
        "profit_pct": float,
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
        current_price = prices_map.get(instrument)
        if current_price is None:
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

            if side_label == "LONG":
                profit_pct = (current_price - entry_price) / entry_price * 100.0
            else:  # SHORT
                profit_pct = (entry_price - current_price) / entry_price * 100.0

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


def fetch_current_prices(instruments):
    """
    Fetch mid prices for a list of instruments.
    Returns: {instrument: mid_price}
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
        mid = (bid + ask) / 2.0
        prices_map[instrument] = mid

    return prices_map


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

def get_gspread_client():
    creds_info = json.loads(GOOGLE_CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return gspread.authorize(creds)


def get_or_create_worksheet(client):
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=200, cols=20)
    return ws


def load_prior_state(ws):
    """
    Reads current sheet and builds a state map:
    key -> {"armed": bool, "ath_pct": float}
    where key = f"{Instrument}|{Side}"
    """
    try:
        records = ws.get_all_records()
    except Exception as e:
        logging.warning(f"Could not read existing sheet state: {e}")
        return {}

    state = {}
    for row in records:
        instr = row.get("Instrument")
        side = row.get("Side")
        if not instr or not side:
            continue
        key = f"{instr}|{side}"

        armed_raw = str(row.get("Armed", "")).strip().lower()
        armed = armed_raw in ("true", "1", "yes", "y")

        ath_raw = row.get("AllTimeHighPct", 0)
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
    """
    header = [
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

    now_str = datetime.now(timezone.utc).isoformat()

    rows = [header]
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

    # Clear & overwrite
    ws.clear()
    ws.update("A1", rows, value_input_option="USER_ENTERED")


# -------------------------
# Trading logic
# -------------------------

def apply_trading_logic(positions, prior_state):
    """
    Takes raw positions and prior state, decides what to close, and
    returns:
      - remaining_positions: list of positions that stay open
      - new_state: updated state map for remaining positions
    """
    remaining_positions = []
    new_state = {}

    for pos in positions:
        key = f"{pos['instrument']}|{pos['side']}"
        prev = prior_state.get(key, {"armed": False, "ath_pct": 0.0})
        armed = prev["armed"]
        ath_pct = prev["ath_pct"]
        profit_pct = pos["profit_pct"]

        # --- Stop Loss (applies always) ---
        if profit_pct <= -STOP_LOSS_PCT:
            logging.info(
                f"{key}: Profit {profit_pct:.2f}% <= -STOP_LOSS_PCT (-{STOP_LOSS_PCT}%), triggering STOP LOSS sell."
            )
            close_position(pos["instrument"], pos["side"])
            continue  # do not keep in remaining_positions

        # --- Arming logic ---
        if not armed and profit_pct >= ARM_THRESHOLD_PCT:
            armed = True
            ath_pct = profit_pct
            logging.info(
                f"{key}: Position became ARMED at {profit_pct:.2f}% (threshold {ARM_THRESHOLD_PCT}%)."
            )

        # --- ATH tracking for armed positions ---
        if armed:
            if profit_pct > ath_pct:
                ath_pct = profit_pct
                logging.info(
                    f"{key}: New all-time high profit {ath_pct:.2f}%."
                )

            # Trailing TP: sell if we've fallen TRAIL_OFFSET_PCT below ATH
            if profit_pct <= ath_pct - TRAIL_OFFSET_PCT:
                logging.info(
                    f"{key}: Profit {profit_pct:.2f}% <= ATH({ath_pct:.2f}%) - TRAIL_OFFSET({TRAIL_OFFSET_PCT}%), "
                    f"triggering TRAILING TP sell."
                )
                close_position(pos["instrument"], pos["side"])
                continue  # sold, do not keep

        # Still open
        remaining_positions.append(pos)
        new_state[key] = {"armed": armed, "ath_pct": ath_pct}

    return remaining_positions, new_state


# -------------------------
# Main loop
# -------------------------

def main_loop():
    logging.info("Starting Oanda-Trader loop.")

    client = get_gspread_client()
    ws = get_or_create_worksheet(client)

    # Ensure header exists at least once
    if not ws.get_all_values():
        logging.info("Initializing worksheet header.")
        write_positions_to_sheet(ws, [], {})

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
                remaining_positions, new_state = apply_trading_logic(positions, prior_state)

                logging.info(f"{len(remaining_positions)} position side(s) remain open after logic.")
                write_positions_to_sheet(ws, remaining_positions, new_state)

        except Exception as e:
            logging.exception(f"Error in main loop: {e}")

        logging.info(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
