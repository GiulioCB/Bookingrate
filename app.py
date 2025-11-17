# app.py — Streamlit UI for Booking.com RateChecker (fast classic + UX tweaks)
import os, re, asyncio, calendar
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
# ---- One-time Playwright browser bootstrap (Streamlit Cloud friendly) ----
import os, sys, subprocess, pathlib, contextlib
import streamlit as st

def _ensure_chromium_installed() -> None:
    """
    Ensure Playwright's Chromium is present.
    Tries a normal install, then falls back to --with-deps.
    Keeps the UI clean by not throwing big warnings unless both attempts fail.
    """
    cache_dir = pathlib.Path.home() / ".cache" / "ms-playwright"
    chromium_dir = cache_dir / "chromium"

    # If already installed, we're done
    if chromium_dir.exists():
        return

    # Prefer an idempotent install without extra apt deps (we already supply packages.txt)
    commands = [
        [sys.executable, "-m", "playwright", "install", "chromium"],
        [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
    ]

    last_err = None
    for cmd in commands:
        try:
            # Keep env, but set a stable browsers path
            env = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(cache_dir)}
            res = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            # Success: break
            return
        except subprocess.CalledProcessError as e:
            last_err = e

    # If both attempts failed, don't spam the UI—just give a compact note.
    # Full details will be in Streamlit Cloud logs.
    with contextlib.suppress(Exception):
        st.info("Preparing browser… if this is the first run, it may take a minute.")
    if last_err:
        # Send concise failure line to the log; you can view it under Manage app → Logs
        print("[playwright-install] failed:", last_err.stdout[-500:] if last_err.stdout else last_err)

_ensure_chromium_installed()
# -------------------------------------------------------------------------

from scraper import scrape_hotels_for_dates, ddmmyyyy

st.set_page_config(page_title="RateChecker • Booking.com", page_icon="🔎", layout="wide")


TITLE = "RateChecker — Booking.com"
INTRO = "Fast price checker for Booking.com properties (cheapest public rates)."
HOTEL_INFO = "Hotels"
DATE_SECTION = "Dates"
ADVANCED = "Advanced options"

# ============ Session defaults ============
if "custom_dates_text" not in st.session_state:
    st.session_state.custom_dates_text = "01.01.2026\n17.01.2026"
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
# persisted results
for k in ("last_results", "last_out_df", "last_debug_df",
          "last_currency", "last_hotels", "last_dates"):
    st.session_state.setdefault(k, None)

# ============ Helpers ============
def _canon_booking_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    u = re.sub(r"^https?://m\.booking\.com", "https://www.booking.com", u, flags=re.I)
    u = re.sub(r"^https?://[^/]*booking\.com", "https://www.booking.com", u, flags=re.I)
    return u.split("#")[0].split("?")[0]

def parse_ddmmyyyy(s: str) -> date:
    return datetime.strptime(s, "%d.%m.%Y").date()

def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))

def generate_every_day_for_months(start: date, months: int) -> List[datetime]:
    end = add_months(start, months)
    days = (end - start).days
    return [datetime.combine(start + timedelta(days=i), datetime.min.time()) for i in range(days)]

# ---------- New monthly rules helpers ----------
def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])

def _next_month(y: int, m: int) -> Tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)

def _first_sun_to_thu_on_or_after(d: date) -> Optional[date]:
    """First weekday that is Sunday–Thursday on/after d (Mon=0..Sun=6)."""
    month = d.month
    while d.month == month:
        if d.weekday() not in (4, 5):  # not Fri(4) or Sat(5)
            return d
        d += timedelta(days=1)
    return None

def _first_weekend_on_or_after(d: date) -> Optional[date]:
    """First Friday or Saturday on/after d."""
    month = d.month
    while d.month == month:
        if d.weekday() in (4, 5):      # Fri(4) or Sat(5)
            return d
        d += timedelta(days=1)
    return None

def _weekday_default_in_month(y: int, m: int) -> date:
    """Pick a stable Sun–Thu within the month (start from the 15th)."""
    anchor = date(y, m, min(15, calendar.monthrange(y, m)[1]))
    d = _first_sun_to_thu_on_or_after(anchor)
    return d or anchor

def _first_friday_of_month(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)  # 4 = Friday

def generate_weekday_and_weekend_per_month(start: date, months: int) -> List[datetime]:
    """
    Two dates per month:
      - Weekday: Sun–Thu
      - Weekend: Fri (for first month: first Fri or Sat AFTER start)
    First month relative to 'start':
      • if days left in month < 8 → skip that month
      • else pick both dates strictly AFTER 'start' within that month
    """
    out: List[datetime] = []
    y, m = start.year, start.month
    produced = 0
    first_month_done = False

    while produced < months:
        if not first_month_done:
            remaining_days = (_month_end(y, m) - start).days  # strictly after start
            if remaining_days < 8:
                y, m = _next_month(y, m)  # skip this month entirely
                first_month_done = True
                continue

            scan_start = start + timedelta(days=1)
            wd = _first_sun_to_thu_on_or_after(scan_start)
            wk = _first_weekend_on_or_after(scan_start)  # Fri or Sat (earliest)
            if wd:
                out.append(datetime.combine(wd, datetime.min.time()))
            if wk:
                out.append(datetime.combine(wk, datetime.min.time()))

            produced += 1
            first_month_done = True
            y, m = _next_month(y, m)
            continue

        # Subsequent months
        wd = _weekday_default_in_month(y, m)
        wk = _first_friday_of_month(y, m)
        out.append(datetime.combine(wd, datetime.min.time()))
        out.append(datetime.combine(wk, datetime.min.time()))
        produced += 1
        y, m = _next_month(y, m)

    return sorted(list({d: None for d in out}.keys()))

def format_dates_ddmmyyyy(dts: List[datetime]) -> str:
    return "\n".join(ddmmyyyy(d) for d in sorted(dts))

#def normalize_custom_text():
    #"""Read, parse, unique+sort the textarea → write back."""
    #txt = st.session_state.get("custom_dates_text", "").strip()
    #if not txt:
        #return
    #cleaned: List[datetime] = []
    #for line in txt.splitlines():
        #s = line.strip()
        #if not s:
            #continue
        #try:
            #cleaned.append(datetime.combine(parse_ddmmyyyy(s), datetime.min.time()))
        #except Exception:
            #pass
    #cleaned = sorted(list({d: None for d in cleaned}.keys()))
    #st.session_state["custom_dates_text"] = "\n".join(ddmmyyyy(d) for d in cleaned)

def normalize_custom_text():
    txt = st.session_state.get("custom_dates_text", "").strip()
    if not txt:
        return
    cleaned = []
    for line in txt.splitlines():
        try:
            cleaned.append(datetime.combine(parse_ddmmyyyy(line.strip()), datetime.min.time()))
        except Exception:
            pass
    cleaned = sorted({d for d in cleaned})
    st.session_state["normalized_custom_dates"] = "\n".join(ddmmyyyy(d) for d in cleaned)
    

# ============ UI ============
st.title(TITLE)
st.caption(INTRO)

with st.container():
    st.subheader(HOTEL_INFO)

    default_df = pd.DataFrame([{"Hotel name":"", "Booking.com property URL":""}])
    hotels_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hotel name": st.column_config.TextColumn(
                "Hotel name", required=True, help="Property name (for search)"),
            "Booking.com property URL": st.column_config.TextColumn(
                "Booking.com property URL", help="Optional: paste the property's Booking.com page URL"),
        },
        key="hotels_editor",
    )

    hotels_input: List[Dict] = []
    for _, row in hotels_df.iterrows():
        name = (row.get("Hotel name") or "").strip()
        url  = _canon_booking_url((row.get("Booking.com property URL") or "").strip())
        if name:
            hotels_input.append({"name": name, "url": url})

with st.container():
    st.subheader(DATE_SECTION)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        start_date = st.date_input("Start date", value=date.today())

        date_mode = st.radio(
            "Mode",
            ["Weekday + Weekend per month", "Every day for N months", "Custom list"],
            index=0,
        )

        months_to_generate = st.slider("Months", min_value=1, max_value=24, value=12, step=1)

        if st.button("Generate"):
            if date_mode == "Weekday + Weekend per month":
                dts = generate_weekday_and_weekend_per_month(start_date, months_to_generate)
            elif date_mode == "Every day for N months":
                dts = generate_every_day_for_months(start_date, months_to_generate)
            else:
                dts = []
            if dts:
                st.session_state["custom_dates_text"] = format_dates_ddmmyyyy(dts)
                st.rerun()

    with col2:
        st.markdown("**Custom dates (DD.MM.YYYY, one per line)**")
        st.text_area(
            "Custom dates (DD.MM.YYYY, one per line)",
            key="custom_dates_text",
            height=220,
            label_visibility="visible",  # or "collapsed" if you want it hidden
            help="You can edit freely. Dates are auto-sorted when you Generate or Run scraping.",
        )


with st.expander(ADVANCED, expanded=False):
    currencies = ["EUR","USD","GBP","CHF","PLN","CZK","SEK","NOK","DKK","HUF"]
    selected_currency = st.selectbox("Currency (default EUR)", options=currencies, index=currencies.index("EUR"))
    debug_flag = st.toggle("Show debug logs", value=False)
    st.caption("Tip: speed knobs via env vars RC_CONCURRENCY, RC_SCAN_LIMIT.")

# ============ Run button & status placeholder ============
run = st.button("Run scraping", type="primary")
status_box = st.empty()  # will show blue info then green success

# ============ Run action ============
if run:
    # Build final dates list
    if date_mode == "Every day for N months":
        dates = generate_every_day_for_months(start_date, months_to_generate)
    elif date_mode == "Weekday + Weekend per month":
        dates = generate_weekday_and_weekend_per_month(start_date, months_to_generate)
    else:
        normalize_custom_text()
        dates = []
        for line in st.session_state["custom_dates_text"].splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                d = parse_ddmmyyyy(s)
                dates.append(datetime.combine(d, datetime.min.time()))
            except Exception:
                pass
        dates = sorted(list({d: None for d in dates}.keys()))

    if not hotels_input:
        st.error("Please add at least one hotel.")
        st.stop()
    if not dates:
        st.error("Please provide dates.")
        st.stop()

    status_box.info(f"Scraping {len(hotels_input)} hotel(s) on {len(dates)} date(s) …")

    # scrape
    results = asyncio.run(
        scrape_hotels_for_dates(
            hotels=hotels_input,
            dates=dates,
            selected_currency=selected_currency,
            debug=debug_flag,
        )
    )

    # Build debug frame
    debug_rows = []
    for (name, ymd), r in results.items():
        row = {"hotel": name, "date": ymd}
        if r.get("status") == "OK":
            row.update({"status": "OK", "reason": None})
        else:
            row.update({"status": "No rate found", "reason": r.get("reason")})
        debug_rows.append(row)
    debug_df = pd.DataFrame(debug_rows).sort_values(["hotel","date"])

    # Breakfast row (one-time detection per hotel on the first scraped date)
    breakfast_by_hotel = {h["name"]: None for h in hotels_input}
    for (name, ymd), r in results.items():
        if r.get("status") == "OK" and r.get("breakfast_included") is not None and breakfast_by_hotel.get(name) is None:
            breakfast_by_hotel[name] = "Yes" if r["breakfast_included"] else "No"
    for h in hotels_input:
        if breakfast_by_hotel[h["name"]] is None:
            breakfast_by_hotel[h["name"]] = "No"

    # Build wide results table
    breakfast_row = {"Date": "Breakfast"}
    for h in hotels_input:
        breakfast_row[h["name"]] = breakfast_by_hotel[h["name"]]

    out_rows = [breakfast_row]
    for d in dates:
        row = {"Date": ddmmyyyy(d)}
        for h in hotels_input:
            key = (h["name"], d.strftime("%Y-%m-%d"))
            r = results.get(key)
            row[h["name"]] = "" if not r or r.get("status") != "OK" or r.get("value") is None else f"{r['value']:.2f}"
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)

    # Persist to session so they remain after downloads/reruns
    st.session_state.last_results = results
    st.session_state.last_out_df = out_df
    st.session_state.last_debug_df = debug_df
    st.session_state.last_currency = selected_currency
    st.session_state.last_hotels = [h["name"] for h in hotels_input]
    st.session_state.last_dates = [d.strftime("%Y-%m-%d") for d in dates]
    st.session_state.show_debug = False  # default hidden each run

    # Replace blue with green
    status_box.success("Done.")

# ============ Render persisted results (if any) ============
if st.session_state.last_out_df is not None:
    # Header with Debug toggle at top-right
    hcol1, hcol2 = st.columns([1, 1])
    with hcol1:
        st.subheader("Results")
    with hcol2:
        # right align: add some filler then the button
        r1, r2, r3 = st.columns([5, 1, 1])
        with r2:
            if st.button("Debug", help="Show/hide debug table"):
                st.session_state.show_debug = not st.session_state.show_debug
                st.rerun()
        # r3 left for symmetry / future

    # Optional debug table (above results)
    if st.session_state.show_debug and st.session_state.last_debug_df is not None:
        st.dataframe(st.session_state.last_debug_df, use_container_width="stretch")

    # Results table
    st.dataframe(st.session_state.last_out_df, use_container_width="stretch")

    # Bottom row: Download (left) and Reset (right)
    dcol_l, dcol_r = st.columns([1, 1])
    with dcol_l:
        st.download_button(
            "Download CSV",
            st.session_state.last_out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"booking_rates_{st.session_state.last_currency or 'EUR'}.csv",
            mime="text/csv",
        )
    with dcol_r:
        # Align to right by adding filler columns
        f1, f2, f3 = st.columns([6, 2, 1])
        with f3:
            if st.button("Reset", type="secondary", help="Clear hotels, dates and results"):
                # clear everything
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

