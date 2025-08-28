
# scraper.py — fast, Booking.com-only, classic logic + one-time breakfast check
import os, sys, re, json, asyncio, random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from rapidfuzz import fuzz
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeoutError

# ---------- Windows Playwright event loop fix ----------
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ---------- Tunables (can override via environment) ----------
NUM_CONCURRENCY = int(os.getenv("RC_CONCURRENCY", "4"))
NAV_TIMEOUT_MS  = int(os.getenv("RC_NAV_TIMEOUT_MS", "15000"))
SEL_TIMEOUT_MS  = int(os.getenv("RC_SEL_TIMEOUT_MS", "9000"))
SCAN_LIMIT      = int(os.getenv("RC_SCAN_LIMIT", "20"))  # how many price cells to scan per page

COMPANY = os.getenv("COMPANY_NAME", "CBRE")
CONTACT = os.getenv("RC_CONTACT", "rates@example.com")

# ---------- Utilities ----------
def canonicalize_booking_url(u: Optional[str]) -> Optional[str]:
    if not u: return None
    u = u.strip()
    u = re.sub(r"^https?://m\.booking\.com", "https://www.booking.com", u, flags=re.I)
    u = re.sub(r"^https?://[^/]*booking\.com", "https://www.booking.com", u, flags=re.I)
    return u.split("#")[0].split("?")[0]

def ddmmyyyy(d: datetime) -> str: return d.strftime("%d.%m.%Y")
def iso(d: datetime) -> str:      return d.strftime("%Y-%m-%d")

def _must_be_booking(url: str) -> None:
    if "booking.com" not in urlparse(url).netloc.lower():
        raise RuntimeError("Only booking.com domains are supported.")

_MONEY_RE = re.compile(
    r'(?<![A-Za-z0-9])(\d{1,3}(?:[.\s\u00A0]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2})?)(?![\dA-Za-z])'
)

def parse_money_max(text: str) -> Optional[float]:
    """Pick the largest monetary number in a string; tolerate 1.234,56 / 1,234.56 / '€ 217'."""
    if not text: return None
    t = text.replace("\u00A0", " ").strip()
    best = None
    for m in _MONEY_RE.findall(t):
        s = m.replace(" ", "").replace("\u00A0", "")
        if "," in s and "." in s:
            s = s.replace(".", "") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
        elif "," in s:
            s = s.replace(".", "").replace(",", ".")
        try:
            v = float(s)
            best = v if best is None else max(best, v)
        except: pass
    return best

def detect_breakfast_included(text: str) -> Optional[bool]:
    if not text:
        return None
    t = text.lower()
    yes = ["frühstück inbegriffen", "frühstück inklusive", "inklusive frühstück",
           "breakfast included", "with breakfast", "mit frühstück"]
    no  = ["ohne frühstück", "frühstück nicht inbegriffen", "frühstück nicht inklusive",
           "breakfast not included", "no breakfast"]
    if any(k in t for k in yes): return True
    if any(k in t for k in no):  return False
    # Heuristic: mentions breakfast with a € price usually means extra (not included)
    if ("frühstück" in t or "breakfast" in t) and ("inkl" not in t and "included" not in t):
        return False
    return None

async def accept_cookies_if_present(page: Page):
    for sel in [
        'button:has-text("Accept")',
        'button:has-text("Alle akzeptieren")',
        'button:has-text("Ich stimme zu")',
        '[id*="onetrust-accept"]',
        '[data-testid="cookie-notice-accept"]',
    ]:
        try:
            if await page.locator(sel).count():
                await page.locator(sel).first.click(timeout=2000)
                break
        except: pass

async def page_settle(page: Page):
    await page.wait_for_timeout(900)
    try:
        for _ in range(2):
            await page.mouse.wheel(0, 1200)
            await page.wait_for_timeout(300)
    except: pass

async def _wait_for_any(page: Page, selectors: List[str], timeout: int) -> bool:
    end = asyncio.get_event_loop().time() + timeout/1000
    remaining = timeout
    for sel in selectors:
        try:
            await page.wait_for_selector(sel, timeout=remaining, state="visible")
            return True
        except:
            remaining = int((end - asyncio.get_event_loop().time())*1000)
            if remaining <= 0: break
    return False

# ---------- Resolve a search result to property URL ----------
def score_candidate(hotel_query: str, city_hint: Optional[str], name_text: str, area_text: str) -> float:
    base = fuzz.token_sort_ratio(hotel_query, name_text)
    city_bonus = 15 if city_hint and city_hint.lower() in (name_text + " " + area_text).lower() else 0
    return base + city_bonus

async def resolve_property_url(page: Page, hotel_name: str, city: Optional[str], debug: bool=False) -> Optional[str]:
    query = f"{hotel_name} {city}" if city else hotel_name
    search_url = ("https://www.booking.com/searchresults.html"
                  f"?ss={quote_plus(query)}&group_adults=2&no_rooms=1&group_children=0&lang=de-de")
    try:
        await page.goto(search_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    except: 
        return None

    await accept_cookies_if_present(page)
    await page_settle(page)

    # sometimes redirected directly to property
    if "/hotel/" in page.url:
        return page.url.split("?")[0]

    card_selectors = [
        '[data-testid="property-card"]',
        '[data-testid="property-card-container"]',
        'div[data-testid^="property-card"]',
        'div[data-testid="sr_list"] article',
    ]
    ok = await _wait_for_any(page, card_selectors, timeout=SEL_TIMEOUT_MS)
    if not ok:
        return None

    cards = page.locator(", ".join(card_selectors))
    n = min(await cards.count(), 30)
    best = None
    for i in range(n):
        card = cards.nth(i)
        name_loc = card.locator('[data-testid="title"], a[data-testid="title-link"], h3')
        addr_loc = card.locator('[data-testid="address"], [data-testid="location"]')
        link_loc = card.locator('a[data-testid="title-link"], a[href*="/hotel/"]')
        if not await name_loc.count() or not await link_loc.count():
            continue
        title = (await name_loc.first.inner_text()).strip()
        addr  = (await addr_loc.first.inner_text()).strip() if await addr_loc.count() else ""
        href  = await link_loc.first.get_attribute("href")
        url   = "https://www.booking.com"+href if href and href.startswith("/") else href
        score = score_candidate(hotel_name, city, title, addr)
        best = (score, url) if (best is None or score > best[0]) else best

    return best[1].split("?")[0] if best else None

# ---------- Quick DOM scan: cheapest visible price ----------
async def cheapest_price_quick(page: Page, nights: int, debug: bool = False, want_breakfast: bool = False):
    """
    Find the cheapest *bookable* rate for the current search context.
    Bookable == the row's quantity <select> has an enabled option value="1".
    This respects group_adults=2&no_rooms=1&group_children=0 that we put in the URL.
    """
    await _wait_for_any(page, ["#hp_availability", "[data-component='hotel/new-rooms-table']"], SEL_TIMEOUT_MS)

    # 1) Focus on rows that are actually selectable (rooms available for our occupancy)
    qty_selects = page.locator("select").filter(
        has=page.locator("option[value='1']:not([disabled])")
    )

    best_val = None
    best_bf = None

    count = await qty_selects.count()
    n = min(count, SCAN_LIMIT)

    for i in range(n):
        # Get the row container for this selectable quantity control
        sel = qty_selects.nth(i)
        row = sel.locator(
            "xpath=ancestor::*[self::tr or self::div]"
            "[descendant::select]"
            "[descendant::*["
            "  @data-testid='price-and-discounted-price' or "
            "  contains(@data-testid,'price-for') or "
            "  contains(@class,'bui-price-display__value') or "
            "  contains(@class,'prco-ltr-right-align-helper') or "
            "  contains(@class,'prco-valign-middle-helper')"
            "]]"
        ).first

        try:
            if not await row.is_visible(timeout=1500):
                continue
            # Skip calendar/dialog overlays
            if await row.locator("xpath=ancestor-or-self::*[@role='dialog' or contains(@data-testid,'calendar')]").count():
                continue
        except Exception:
            continue

        # 2) Extract the current price from the row's price cell
        price_el = row.locator(
            ":is([data-testid='price-and-discounted-price'], "
            "[data-testid*='price-for'], "
            ".bui-price-display__value, "
            ".prco-ltr-right-align-helper, "
            ".prco-valign-middle-helper)"
        ).first

        try:
            txt = (await price_el.text_content(timeout=800)) or ""
        except Exception:
            continue

        val = parse_money_max(txt)
        if val is None:
            continue

        # 3) (optional) detect breakfast on the cheapest row only
        if best_val is None or val < best_val:
            if want_breakfast:
                try:
                    row_text = (await row.text_content(timeout=800)) or ""
                except Exception:
                    row_text = ""
                best_bf = detect_breakfast_included(row_text)
            best_val = val

    if best_val is None:
        return None

    total = float(best_val)
    per = round(total / nights, 2) if nights else None
    if want_breakfast:
        return total, per, (best_bf if best_bf is not None else False)
    return total, per

# ---------- GraphQL fallback (fast) ----------
def _pagename_from_url(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path
        if "/hotel/" in path and path.endswith(".html"):
            return path.split("/")[-1].replace(".html","")
    except: pass
    return None

def _extract_property_tokens_from_html(html: str) -> dict:
    toks: Dict[str,str] = {}
    for pat in [r"b_csrf_token:\s*'([^']+)'", r'b_csrf_token:\s*"([^"]+)"', r'"b_csrf_token"\s*:\s*"([^"]+)"']:
        m = re.search(pat, html);  toks["csrf"]=m.group(1) if m else toks.get("csrf")
    for pat in [r'hotelName:\s*"([^"]+)"', r"hotelName:\s*'([^']+)'", r'"hotelName"\s*:\s*"([^"]+)"']:
        m = re.search(pat, html);  toks["pagename"]=m.group(1) if m else toks.get("pagename")
    for pat in [r'hotelCountry:\s*"([^"]+)"', r"hotelCountry:\s*'([^']+)'", r'"hotelCountry"\s*:\s*"([^"]+)"']:
        m = re.search(pat, html);  toks["country"]=m.group(1) if m else toks.get("country")
    return toks

async def graphql_availability_price(page: Page, checkin: datetime, days: int = 31, debug: bool=False) -> Optional[dict]:
    html = await page.content()
    toks = _extract_property_tokens_from_html(html)
    if "pagename" not in toks:
        p = _pagename_from_url(page.url)
        if p: toks["pagename"] = p
    if not {"pagename","csrf"}.issubset(toks.keys()):
        return {"error":"tokens_not_found"}

    body = {
        "operationName":"AvailabilityCalendar",
        "variables":{"input":{
            "travelPurpose":2,
            "pagenameDetails":{"countryCode": toks.get("country",""), "pagename": toks["pagename"]},
            "searchConfig":{"searchConfigDate":{"startDate": checkin.strftime("%Y-%m-%d"), "amountOfDays": max(31,days)},
                            "nbAdults":2, "nbRooms":1},
        }},
        "extensions":{},
        "query":"query AvailabilityCalendar($input: AvailabilityCalendarQueryInput!) { availabilityCalendar(input: $input) { ... on AvailabilityCalendarQueryResult { days { available avgPriceFormatted checkin minLengthOfStay __typename } __typename } ... on AvailabilityCalendarQueryError { message __typename } __typename } }"
    }
    resp = await page.context.request.post(
        "https://www.booking.com/dml/graphql?lang=de-de",
        data=json.dumps(body, separators=(",",":")),
        headers={"content-type":"application/json","x-booking-csrf-token":toks["csrf"],
                 "origin":"https://www.booking.com","referer": page.url.split("?")[0]}
    )
    if not resp.ok: return {"error": f"http_{resp.status}"}
    try:
        data = await resp.json()
    except:
        return {"error":"bad_json"}

    days_data = data.get("data",{}).get("availabilityCalendar",{}).get("days",[]) or []
    target = next((d for d in days_data if d.get("checkin")==checkin.strftime("%Y-%m-%d")), None)
    if not target:                return {"error":"date_not_in_calendar"}
    if not target.get("available",0): return {"error":"sold_out"}
    per = parse_money_max(target.get("avgPriceFormatted","") or "")
    if per is None:              return {"error":"price_not_found"}
    minlos = int(target.get("minLengthOfStay") or 1)
    total  = round(per * minlos, 2)
    return {"nights_queried": minlos, "minstay_applied": (minlos>1),
            "total_incl_taxes": total, "per_night": round(total/minlos, 2)}

# ---------- Get price for a date ----------
async def get_price_for_dates(page: Page, property_url: str, checkin: datetime, nights: int, currency: str,
                              debug: bool=False, check_breakfast: bool=False) -> Dict:
    base_url = property_url.split("?")[0];  _must_be_booking(base_url)
    url = (f"{base_url}?checkin={iso(checkin)}&checkout={(checkin+timedelta(days=nights)).strftime('%Y-%m-%d')}"
           f"&group_adults=2&no_rooms=1&group_children=0&selected_currency={currency}&lang=de-de")
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    except PWTimeoutError:
        return {"error":"timeout_opening_page"}

    await accept_cookies_if_present(page)
    await page_settle(page)

    # quick DOM cheapest
    dom = await cheapest_price_quick(page, nights=nights, debug=debug, want_breakfast=check_breakfast)
    if dom:
        if check_breakfast:
            total, per, bf = dom
            return {"nights_queried": nights, "minstay_applied": False,
                    "total_incl_taxes": total, "per_night": per, "breakfast_included": bf}
        else:
            total, per = dom
            return {"nights_queried": nights, "minstay_applied": False,
                    "total_incl_taxes": total, "per_night": per}

    # fallback: GraphQL
    gql = await graphql_availability_price(page, checkin, days=max(31, nights+3), debug=debug)
    if gql and "error" not in gql:
        if check_breakfast:
            gql["breakfast_included"] = None  # unknown from API
        return gql
    return {"error": (gql or {}).get("error","no_rate")}

# ---------- Task on a shared context ----------
async def _scrape_one_on_context(context, hotel: Dict, checkin: datetime, selected_currency: str,
                                 debug: bool=False, check_breakfast: bool=False) -> Dict:
    hotel_name = hotel.get("name") or hotel.get("hotel") or ""
    provided_url = canonicalize_booking_url(hotel.get("url"))
    page = await context.new_page()
    page.set_default_timeout(max(NAV_TIMEOUT_MS, SEL_TIMEOUT_MS))
    page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

    try:
        url = provided_url or await resolve_property_url(page, hotel_name, city=None, debug=debug)
        if not url:
            await page.close()
            return {"hotel": hotel_name, "date": iso(checkin), "status":"No rate found", "reason":"no_url"}

        result = await get_price_for_dates(page, url, checkin, nights=1, currency=selected_currency,
                                           debug=debug, check_breakfast=check_breakfast)
    except Exception as e:
        await page.close()
        return {"hotel": hotel_name, "date": iso(checkin), "status":"No rate found", "reason": f"exception:{e.__class__.__name__}"}

    await page.close()
    if "error" in result:
        return {"hotel": hotel_name, "date": iso(checkin), "status":"No rate found", "reason": result["error"]}
    return {
        "hotel": hotel_name, "date": iso(checkin), "status": "OK",
        "value": result["per_night"], "total_for_queried_nights": result["total_incl_taxes"],
        "nights_queried": result["nights_queried"], "minstay_applied": result["minstay_applied"],
        "currency": selected_currency,
        "breakfast_included": result.get("breakfast_included")  # True/False/None (only first date per hotel is checked)
    }

# ---------- Orchestrator (single browser reused) ----------
async def scrape_hotels_for_dates(hotels: List[Dict], dates: List[datetime], selected_currency: str="EUR", debug: bool=False) -> Dict:
    results: Dict[Tuple[str,str], Dict] = {}
    exe = os.getenv("RC_CHROME_PATH")

    # Determine the FIRST date (once) to compute breakfast per hotel
    first_date_for: Dict[str, datetime] = {}
    if dates:
        min_date = min(dates)
        for h in hotels:
            first_date_for[h["name"]] = min_date

    async with async_playwright() as p:
        launch_args = dict(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        browser = None
        if exe and os.path.exists(exe):
            try: browser = await p.chromium.launch(executable_path=exe, **launch_args)
            except: browser = None
        if browser is None:
            for ch in ("chrome","msedge"):
                try:
                    browser = await p.chromium.launch(channel=ch, **launch_args)
                    break
                except: pass
        if browser is None:
            browser = await p.chromium.launch(**launch_args)

        context = await browser.new_context(
            locale="de-DE",
            user_agent=(f"{COMPANY}-RateChecker/1.0 (public rates; Playwright; contact: {CONTACT})")
        )

        # Speed-up: block heavy resources
        async def _route(route):
            rt = route.request.resource_type
            if rt in ("image", "media", "font"):
                await route.abort()
            else:
                await route.continue_()
        await context.route("**/*", _route)

        sem = asyncio.Semaphore(NUM_CONCURRENCY)

        async def _task(h, d):
            async with sem:
                await asyncio.sleep(random.uniform(0.08, 0.25))
                want_bf = (first_date_for.get(h["name"]) == d)
                r = await _scrape_one_on_context(context, h, d, selected_currency, debug, check_breakfast=want_bf)
                results[(h["name"], iso(d))] = r

        await asyncio.gather(*[_task(h, d) for h in hotels for d in dates])
        await context.close()
        await browser.close()

    return results
