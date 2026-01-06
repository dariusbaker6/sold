#!/usr/bin/env python3
# Streamlit app: TrenchFeed - robust Top Coins link generation (Dexscreener, Solscan, Birdeye)
# Key fixes:
# 1) normalize token column names from views (token_address, base_token, token, token_addr, mint, mint_address)
# 2) backfill token_address from pairs table when only pair_address exists
# 3) dexscreener prefers pair_address if present, else falls back to token_address
# 4) LinkColumn config applied only if available
# 5) sanity caption shows row counts and valid token/pair counts
# 6) PAYWALL: Requires valid API key from api_customers table

import os
from typing import Dict, List, Optional, Iterable, Set, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
import json
import time

# Optional auto-refresh component.  Attempt import from the third‑party package; if unavailable, leave as None.
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # fallback: no automatic refresh

# ============================= Config =============================
st.set_page_config(page_title="TrenchFeed - Early Leaders", page_icon="🚀", layout="wide")
# Inject custom CSS for a modern, beautiful dashboard.  This styling enhances the
# overall look and feel without altering the underlying functionality.  The
# gradient background, dark sidebar, styled headings, and improved table colors
# provide a cohesive visual experience.
custom_css = """
/* Remove default Streamlit main menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* General body styling with gradient background and subtle text coloring */
[data-testid="stAppViewContainer"] {
    background-color: #0F2027;
    background-image: linear-gradient(120deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    color: #E0E0E0;
}

/* Style the sidebar with a darker shade and consistent text color */
[data-testid="stSidebar"] {
    background-color: #1A2B3A;
    color: #E0E0E0;
}

/* Enhance appearance of sliders and toggles in sidebar */
[data-testid="stSidebar"] .stSlider > div, [data-testid="stSidebar"] .stToggle > label {
    color: #E0E0E0;
}
[data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 4px;
    padding: 4px;
}

/* Headings styling with gradient text effect */
h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2, h3, h4 {
    color: #E0E0E0;
}

/* Style hyperlinks to stand out on dark background */
a {
    color: #71C7EC;
}

/* Style for Streamlit dataframes (Ag-Grid) */
.ag-theme-streamlit {
    background-color: rgba(255,255,255,0.02) !important;
    color: #E0E0E0 !important;
    font-size: 14px;
}
.ag-theme-streamlit .ag-header, .ag-theme-streamlit .ag-header-cell {
    background-color: rgba(255,255,255,0.06) !important;
    color: #E0E0E0 !important;
    font-weight: bold;
}
.ag-theme-streamlit .ag-row-hover {
    background-color: rgba(255,255,255,0.05) !important;
}

/* Paywall styling */
.paywall-container {
    max-width: 500px;
    margin: 100px auto;
    padding: 40px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
}
.paywall-title {
    font-size: 2rem;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.paywall-subtitle {
    color: #A0A0A0;
    margin-bottom: 2rem;
}
.subscribe-btn {
    display: inline-block;
    padding: 12px 32px;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: #0F2027 !important;
    text-decoration: none;
    border-radius: 8px;
    font-weight: bold;
    margin-top: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.subscribe-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 201, 255, 0.3);
}

/* Stream status indicator styling */
.stream-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.stream-status.live {
    background: linear-gradient(135deg, rgba(0, 201, 255, 0.15) 0%, rgba(146, 254, 157, 0.15) 100%);
    border: 1px solid rgba(0, 201, 255, 0.4);
    color: #92FE9D;
}
.stream-status.paused {
    background: rgba(255, 185, 0, 0.12);
    border: 1px solid rgba(255, 185, 0, 0.4);
    color: #FFB900;
}
.stream-status .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #92FE9D;
    animation: pulse-glow 1.5s ease-in-out infinite;
}
.stream-status.paused .pulse {
    background: #FFB900;
    animation: none;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px rgba(146, 254, 157, 0.6); }
    50% { opacity: 0.5; box-shadow: 0 0 12px rgba(146, 254, 157, 0.9); }
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

def cfg(key: str, default: str = "") -> str:
    v = os.environ.get(key, default)
    try:
        v = st.secrets.get(key, v)  # type: ignore[attr-defined]
    except Exception:
        pass
    return str(v).strip()

SB_URL    = cfg("SUPABASE_URL").rstrip("/")
SB_KEY    = cfg("SUPABASE_SERVICE_ROLE")
SB_SCHEMA = cfg("SUPABASE_SCHEMA", "public")

if not SB_URL or not SB_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE")
    st.stop()

SESSION = requests.Session()
SESSION.headers.update({
    "apikey": SB_KEY,
    "Authorization": f"Bearer {SB_KEY}",
    "Content-Type": "application/json",
    "Accept-Profile": SB_SCHEMA,
    "Content-Profile": SB_SCHEMA,
})

# ============================= PAYWALL AUTHENTICATION =============================
STRIPE_PURCHASE_URL = "https://buy.stripe.com/dRm7sLdUF0DR2m7ga07IY08"

def validate_api_key(api_key: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate an API key against the api_customers table.
    
    Returns:
        Tuple of (is_valid, message, customer_data)
    """
    if not api_key or not api_key.strip():
        return False, "Please enter your API key.", None
    
    api_key = api_key.strip()
    
    try:
        url = f"{SB_URL}/rest/v1/api_customers"
        params = {
            "select": "id,email,api_key,tier,status,expires_at",
            "api_key": f"eq.{api_key}",
            "limit": "1"
        }
        
        response = SESSION.get(url, params=params, timeout=10)
        
        if response.status_code not in (200, 206):
            return False, f"Error validating API key. Please try again.", None
        
        data = response.json()
        
        if not data or len(data) == 0:
            return False, "Invalid API key. Please check your key or subscribe to get access.", None
        
        customer = data[0]
        
        # Check status
        if customer.get("status") != "active":
            status = customer.get("status", "unknown")
            return False, f"Your subscription is {status}. Please renew to continue.", None
        
        # Check expiration for trial tier
        if customer.get("tier") == "trial" and customer.get("expires_at"):
            expires_at = pd.to_datetime(customer["expires_at"], utc=True)
            now = pd.Timestamp.now(tz="UTC")
            if expires_at < now:
                return False, "Your trial has expired. Please subscribe to continue.", None
        
        return True, f"Welcome! Tier: {customer.get('tier', 'unknown').upper()}", customer
        
    except requests.exceptions.Timeout:
        return False, "Connection timeout. Please try again.", None
    except requests.exceptions.RequestException as e:
        return False, f"Network error. Please check your connection.", None
    except Exception as e:
        return False, f"Validation error. Please try again.", None

def show_paywall():
    """Display the paywall page with API key input and purchase link."""
    st.markdown("""
        <div class="paywall-container">
            <div class="paywall-title">🔒 TrenchFeed Dashboard</div>
            <div class="paywall-subtitle">Enter your API key to access the live dashboard</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the input form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔑 Enter API Key")
        
        # API Key input
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="tk_live_xxxxxxxxxxxxx",
            help="Enter your TrenchFeed API key"
        )
        
        # Login button
        if st.button("🚀 Access Dashboard", type="primary", use_container_width=True):
            if api_key_input:
                is_valid, message, customer = validate_api_key(api_key_input)
                if is_valid:
                    st.session_state["authenticated"] = True
                    st.session_state["api_key"] = api_key_input
                    st.session_state["customer"] = customer
                    st.success(message)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter your API key.")
        
        st.markdown("---")
        
        # Subscribe section
        st.markdown("### 💳 Don't have an API key?")
        st.markdown("""
            Get instant access to:
            - 📊 Real-time market data streaming
            - 🚀 Launch Radar for new tokens
            - 🏆 Early Leader identification
            - 🔎 Deep token analytics
        """)
        
        st.markdown(f"""
            <a href="{STRIPE_PURCHASE_URL}" target="_blank" class="subscribe-btn">
                Subscribe Now →
            </a>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("After subscribing, you'll receive your API key via email.")

def check_authentication() -> bool:
    """Check if user is authenticated. Returns True if authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    return st.session_state.get("authenticated", False)

def logout():
    """Clear authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["api_key"] = None
    st.session_state["customer"] = None

# ============================= CHECK AUTHENTICATION =============================
if not check_authentication():
    show_paywall()
    st.stop()

# ============================= AUTHENTICATED USER CONTENT BELOW =============================

# ============================= Helpers =============================
def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def iso(ts: pd.Timestamp) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        utc_ts = pd.Timestamp(ts).tz_convert("UTC")
    except Exception:
        utc_ts = pd.Timestamp(ts).tz_localize("UTC")
    utc_ts = utc_ts.floor("s")
    return utc_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def iso_hours_ago(hours: int) -> str:
    ts = now_utc() - pd.Timedelta(hours=hours)
    ts = ts.floor("s")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def to_dt(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =====================================================================
# Streaming helpers
#
# The enterprise API streams new‑line delimited JSON (NDJSON) events via
# Server‑Sent Events (SSE).  Each event is framed by a blank line and
# contains one or more lines beginning with "data: ".  The following
# helper decodes these frames and yields a parsed JSON object per event.
#
ENTERPRISE_URL: str = "https://api.trenchfeed.cc/stream/enterprise"
# NOTE: The provided token is a read‑only demonstration token suitable
# only for experimentation.  In production you should inject the token
# via environment variables or Streamlit secrets.
ENTERPRISE_TOKEN: str = "tk_live_henCCxnVFK4E1CaGlNRO7aH8PA11cxtf"

def stream_enterprise(api_key: str = ENTERPRISE_TOKEN, *, max_events: int = 200) -> Iterable[Dict]:
    """Yield parsed JSON objects from the enterprise streaming endpoint.

    Parameters
    ----------
    api_key: str
        The bearer token used for authentication.
    max_events: int
        Maximum number of events to yield before returning.  Limiting
        prevents runaway loops during interactive sessions.

    Yields
    ------
    dict
        Parsed JSON payload for each SSE `data:` event.  If a line
        cannot be decoded as JSON it is skipped.
    """
    url = ENTERPRISE_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream"
    }
    # Open a streaming HTTP connection.  The timeout is set to 60 seconds
    # per read to allow for network delays without freezing the UI.
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            buffer = ""
            event_count = 0
            for chunk in resp.iter_content(chunk_size=1024):
                # Append decoded text to our buffer
                try:
                    buffer += chunk.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                # Process complete frames separated by two newlines
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for line in frame.split("\n"):
                        if line.startswith("data: "):
                            json_text = line[6:]
                            try:
                                payload = json.loads(json_text)
                                yield payload
                                event_count += 1
                                if event_count >= max_events:
                                    return
                            except Exception:
                                # Skip malformed JSON lines
                                continue
    except Exception:
        # On any failure (network or authentication), simply return
        return

# ============================= UI helpers =============================
DISCLAIMER_TEXT: str = (
    "Token names and symbols displayed here are user‑generated by token creators in the market "
    "and do not reflect the views of TrenchFeed."
    " Investments made using information displayed here are purely at your own risk. TrenchFeed does not provide financial investment advice."
)

def display_disclaimer() -> None:
    """Render a consistent disclaimer across all tabs.

    Uses HTML to draw attention to the message without clashing with
    existing dark styling.  The box uses a semi‑transparent background
    so as not to distract from the content but to remain visible.
    """
    st.markdown(
        f"<div style='background: rgba(255, 255, 255, 0.1); padding: 0.5rem 0.75rem; "
        f"border-left: 4px solid #FFB900; margin-bottom: 1rem;'>"
        f"<strong>Disclaimer:</strong> {DISCLAIMER_TEXT}</div>",
        unsafe_allow_html=True,
    )

# REST base
def rest_get(table: str, params: Dict[str, str], start: int = 0, stop: int = 9999, timeout: int = 30) -> List[Dict]:
    url = f"{SB_URL}/rest/v1/{table}"
    headers = SESSION.headers.copy()
    headers["Range-Unit"] = "items"
    headers["Range"] = f"{start}-{stop}"
    r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
    # Successful responses include 200 and 206 (partial).  Return parsed JSON.
    if r.status_code in (200, 206):
        try:
            return r.json()
        except Exception:
            return []
    # Handle database timeouts gracefully: Supabase returns HTTP 500 with a cancellation code.
    if r.status_code == 500:
        # Display a user-friendly message rather than a raw exception.  Note: st may not be available in
        # all contexts, but this ensures that the UI remains clean for top_coins views.
        try:
            st.info(f"The '{table}' view timed out. Please reduce the date range or lower the row limit and try again.")
        except Exception:
            pass
        return []
    # For other errors, suppress noisy 404 messages for optional tables.  Some
    # views (e.g. 'listings' and 'risk_flags') may not exist in all schemas and
    # return a 404 with a descriptive body.  We silently return an empty list
    # for those cases so the UI remains clean.
    if r.status_code == 404:
        body_lower = (r.text or "").lower()
        if any(tok in body_lower for tok in ["public.listings", "public.risk_flags", "could not find the table"]):
            return []
    # For other errors, log a warning with truncated details
    st.warning(f"Fetch error for {table}: HTTP {r.status_code}: {r.text[:240]}")
    return []

def fetch_table(
    table: str,
    select: str = "*",
    where: Optional[Dict[str, str]] = None,
    order: Optional[str] = None,
    limit: int = 1000,
    start: int = 0,
) -> pd.DataFrame:
    params: Dict[str, str] = {"select": select, "limit": str(limit)}
    if order:
        params["order"] = order
    if where:
        params.update(where)
    rows = rest_get(table, params=params, start=start, stop=start + limit - 1)
    return pd.DataFrame(rows or [])

def fetch_view(view_name: str, limit: int) -> pd.DataFrame:
    df = fetch_table(view_name, select="*", limit=limit)
    if df.empty:
        return df
    for col in ["snapshot_ts","pair_created_at","last_seen","last_window","created_at","start_ts","end_ts","ts"]:
        if col in df.columns:
            df[col] = to_dt(df[col])
    return df

# Chunk helper
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# Token metadata join
def fetch_tokens_for_addresses(addresses: Iterable[str]) -> pd.DataFrame:
    addrs = sorted({a for a in addresses if isinstance(a, str) and a})
    if not addrs:
        return pd.DataFrame(columns=["token_address","name","symbol"])
    CHUNK = 150
    parts: List[pd.DataFrame] = []
    for batch in _chunk(addrs, CHUNK):
        where = {"token_address": "in.(" + ",".join(batch) + ")"}
        parts.append(fetch_table("tokens", select="token_address,name,symbol", where=where, limit=len(batch)))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["token_address","name","symbol"])
    return out.drop_duplicates("token_address", keep="last")

# Pair mapping from base_token
def latest_pair_map_for_tokens(token_addrs: Iterable[str]) -> Dict[str, str]:
    tlist = sorted({a for a in token_addrs if isinstance(a, str) and a})
    if not tlist:
        return {}
    CHUNK = 120
    maps: Dict[str, str] = {}
    for batch in _chunk(tlist, CHUNK):
        where = {"base_token": "in.(" + ",".join(batch) + ")"}
        cols = "pair_address,base_token,snapshot_ts,pair_created_at"
        pairs = fetch_table("pairs", select=cols, where=where, order="snapshot_ts.desc.nullslast", limit=len(batch)*6)
        if pairs.empty:
            continue
        for c in ["snapshot_ts","pair_created_at"]:
            pairs[c] = to_dt(pairs[c])
        pairs = pairs.sort_values(["base_token","snapshot_ts","pair_created_at"], ascending=[True, False, False])
        latest = pairs.drop_duplicates("base_token", keep="first")
        for _, row in latest.iterrows():
            b = row.get("base_token")
            p = row.get("pair_address")
            if isinstance(b, str) and isinstance(p, str) and b and p:
                maps[b] = p
    return maps

# Backfill token from pairs when only pair is present
def base_token_map_for_pairs(pair_addrs: Iterable[str]) -> Dict[str, str]:
    plist = sorted({p for p in pair_addrs if isinstance(p, str) and p})
    if not plist:
        return {}
    CHUNK = 150
    maps: Dict[str, str] = {}
    for batch in _chunk(plist, CHUNK):
        where = {"pair_address": "in.(" + ",".join(batch) + ")"}
        cols = "pair_address,base_token"
        pairs = fetch_table("pairs", select=cols, where=where, limit=len(batch))
        if pairs.empty:
            continue
        for _, row in pairs.iterrows():
            pa = row.get("pair_address")
            bt = row.get("base_token")
            if isinstance(pa, str) and isinstance(bt, str) and pa and bt:
                maps[pa] = bt
    return maps

# Normalize token column name from a variety of possibilities
def normalize_token_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}
    candidates = ["token_address","base_token","token","token_addr","mint","mint_address"]
    found_src = None
    for cand in candidates:
        if cand in cols:
            found_src = cols[cand]
            break
    if found_src and found_src != "token_address":
        out = out.rename(columns={found_src: "token_address"})
    if "token_address" not in out.columns:
        out["token_address"] = ""
    return out, "token_address"

# Ensure pair links exist, by mapping from token_address when needed
def ensure_pair_links(df: pd.DataFrame, token_col: str = "token_address") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "pair_address" not in out.columns or out["pair_address"].fillna("").eq("").all():
        tokens = [t for t in out.get(token_col, pd.Series(dtype=str)).astype(str).tolist() if t]
        pmap = latest_pair_map_for_tokens(tokens)
        out["pair_address"] = out.get("pair_address", pd.Series([""]*len(out)))
        out.loc[:, "pair_address"] = out[token_col].map(lambda t: pmap.get(str(t), ""))
    return out

# If token is missing but pair exists, backfill token from pairs.base_token
def ensure_token_from_pairs(df: pd.DataFrame, token_col: str = "token_address") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    need = (out.get(token_col, pd.Series(dtype=str)).fillna("") == "")
    if "pair_address" in out.columns and need.any():
        plist = out.loc[need, "pair_address"].dropna().astype(str).tolist()
        if plist:
            bmap = base_token_map_for_pairs(plist)
            out.loc[need, token_col] = out.loc[need, "pair_address"].map(lambda p: bmap.get(str(p), ""))
    return out

# Join token names and symbols
def attach_token_names(df: pd.DataFrame, token_col: str = "token_address") -> pd.DataFrame:
    out = df.copy()
    if out.empty or token_col not in out.columns:
        for c in ["name","symbol"]:
            if c not in out.columns:
                out[c] = ""
        return out
    toks = out[token_col].dropna().astype(str).unique().tolist()
    meta = fetch_tokens_for_addresses(toks)
    if meta.empty:
        for c in ["name","symbol"]:
            if c not in out.columns:
                out[c] = ""
        return out
    out = out.drop(columns=[c for c in ["name","symbol"] if c in out.columns])
    out = out.merge(meta.rename(columns={"token_address": token_col}), on=token_col, how="left")
    out[["name","symbol"]] = out[["name","symbol"]].fillna("")
    return out

# Link builder prefers pair link first, then token link
def add_links(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Dexscreener
    if "pair_address" in out.columns:
        out["dexscreener"] = out["pair_address"].apply(
            lambda p: f"https://dexscreener.com/solana/{p}" if isinstance(p, str) and p else ""
        )
    else:
        out["dexscreener"] = ""
    # Fallback to token link when pair link is missing
    if "token_address" in out.columns:
        out.loc[out["dexscreener"] == "", "dexscreener"] = out.loc[out["dexscreener"] == "", "token_address"].apply(
            lambda t: f"https://dexscreener.com/solana/{t}" if isinstance(t, str) and t else ""
        )
        out["solscan"] = out["token_address"].apply(
            lambda m: f"https://solscan.io/token/{m}" if isinstance(m, str) and m else ""
        )
        out["birdeye"] = out["token_address"].apply(
            lambda m: f"https://birdeye.so/token/{m}?chain=solana" if isinstance(m, str) and m else ""
        )
    else:
        out["solscan"] = ""
        out["birdeye"] = ""
    return out

def link_config(cols: List[str]):
    cfg_map = {}
    colmod = getattr(st, "column_config", None)
    if not colmod or not hasattr(colmod, "LinkColumn"):
        return cfg_map
    if "dexscreener" in cols:
        cfg_map["dexscreener"] = colmod.LinkColumn("Dexscreener", display_text="Open")
    if "solscan" in cols:
        cfg_map["solscan"] = colmod.LinkColumn("Solscan", display_text="Scan")
    if "birdeye" in cols:
        cfg_map["birdeye"] = colmod.LinkColumn("Birdeye", display_text="Bird")
    return cfg_map

# ============================= Minimal other tabs infra =============================
def fetch_recent_pairs(max_pairs: int, recency_hours: int, max_age_minutes: int, *, use_snapshot_fallback: bool = True) -> pd.DataFrame:
    since_iso = iso_hours_ago(recency_hours)
    pairs = fetch_table(
        "pairs",
        select=("pair_address,base_token,quote_token,price_usd,fdv_usd,market_cap_usd,"
                "pair_created_at,snapshot_ts,base_token_name,base_token_symbol,quote_token_name,quote_token_symbol"),
        where={"snapshot_ts": f"gte.{since_iso}"},
        order="snapshot_ts.desc.nullslast",
        limit=max_pairs,
    )
    if pairs.empty:
        return pairs
    pairs = numeric(pairs, ["price_usd","fdv_usd","market_cap_usd"])
    for tcol in ["pair_created_at","snapshot_ts"]:
        if tcol in pairs.columns:
            pairs[tcol] = to_dt(pairs[tcol])
    eff = pairs["pair_created_at"].copy()
    if use_snapshot_fallback:
        eff = eff.where(eff.notna(), pairs["snapshot_ts"])
    min_ts = now_utc() - pd.Timedelta(minutes=max_age_minutes)
    pairs = pairs.loc[eff >= min_ts].copy()
    pairs["effective_created_at"] = eff
    pairs = pairs.rename(columns={"base_token": "token_address"})
    return pairs

def fetch_swaps_for_pairs(pair_addrs: List[str], since_iso: str, limit_per_batch: int = 10000) -> pd.DataFrame:
    if not pair_addrs:
        return pd.DataFrame(columns=["pair_address","ts","trader_wallet","side","amount_in","amount_out","amount_usd","price_usd"])
    CHUNK = 120
    parts: List[pd.DataFrame] = []
    for batch in _chunk(pair_addrs, CHUNK):
        where = {"pair_address": "in.(" + ",".join(batch) + ")", "ts": f"gte.{since_iso}"}
        cols = "pair_address,ts,trader_wallet,side,amount_in,amount_out,amount_usd,price_usd"
        sw = fetch_table("swaps", select=cols, where=where, order="ts.asc.nullslast", limit=limit_per_batch)
        if not sw.empty:
            sw["ts"] = to_dt(sw["ts"])
            parts.append(sw)
    if parts:
        out = pd.concat(parts, ignore_index=True)
        return numeric(out, ["amount_in","amount_out","amount_usd","price_usd"])
    return pd.DataFrame(columns=["pair_address","ts","trader_wallet","side","amount_in","amount_out","amount_usd","price_usd"])

def fetch_pwm_for_pairs(pair_addrs: List[str], since_iso: str) -> pd.DataFrame:
    if not pair_addrs:
        return pd.DataFrame(columns=["pair_address","window_code","price_change_pct","buys","sells","volume_usd","snapshot_ts"])
    CHUNK = 200
    parts: List[pd.DataFrame] = []
    for batch in _chunk(pair_addrs, CHUNK):
        where = {"pair_address": "in.(" + ",".join(batch) + ")", "snapshot_ts": f"gte.{since_iso}", "window_code": "eq.m5"}
        cols = "pair_address,window_code,price_change_pct,buys,sells,volume_usd,snapshot_ts"
        pm = fetch_table("pair_window_metrics", select=cols, where=where, order="snapshot_ts.asc.nullslast", limit=5000)
        if not pm.empty:
            pm["snapshot_ts"] = to_dt(pm["snapshot_ts"])
            parts.append(pm)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pair_address","window_code","price_change_pct","buys","sells","volume_usd","snapshot_ts"])
    return numeric(out, ["price_change_pct","buys","sells","volume_usd"])

def fetch_lp_events_for_pairs(pair_addrs: List[str], since_iso: str) -> pd.DataFrame:
    if not pair_addrs:
        return pd.DataFrame(columns=["pair_address","ts","action","value_usd"])
    CHUNK = 200
    parts: List[pd.DataFrame] = []
    for batch in _chunk(pair_addrs, CHUNK):
        where = {"pair_address": "in.(" + ",".join(batch) + ")", "ts": f"gte.{since_iso}"}
        cols = "pair_address,ts,action,value_usd"
        le = fetch_table("liquidity_events", select=cols, where=where, order="ts.asc.nullslast", limit=20000)
        if not le.empty:
            le["ts"] = to_dt(le["ts"])
            parts.append(le)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pair_address","ts","action","value_usd"])
    return numeric(out, ["value_usd"])

def _first_trade_ts(sw: pd.DataFrame) -> Optional[pd.Timestamp]:
    if sw.empty:
        return None
    ts = sw["ts"].dropna()
    return ts.min() if not ts.empty else None

def _buy_mask(sw: pd.DataFrame) -> pd.Series:
    if "side" in sw.columns and not sw["side"].isna().all():
        return sw["side"].astype(str).str.lower().eq("buy")
    return pd.Series([i % 2 == 0 for i in range(len(sw))], index=sw.index)

def _amount_for_concentration(sw_buys: pd.DataFrame) -> pd.Series:
    if "amount_usd" in sw_buys.columns and not sw_buys["amount_usd"].isna().all():
        return pd.to_numeric(sw_buys["amount_usd"], errors="coerce").fillna(0.0)
    if "amount_in" in sw_buys.columns and not sw_buys["amount_in"].isna().all():
        return pd.to_numeric(sw_buys["amount_in"], errors="coerce").fillna(0.0)
    if "amount_out" in sw_buys.columns and not sw_buys["amount_out"].isna().all():
        return pd.to_numeric(sw_buys["amount_out"], errors="coerce").fillna(0.0)
    return pd.Series(1.0, index=sw_buys.index)

def compute_early_metrics(
    pairs: pd.DataFrame,
    swaps: pd.DataFrame,
    pwm: pd.DataFrame,
    lp_events: pd.DataFrame,
    burst_window_s: int = 120,
    uniques_window_m: int = 10,
    pwm_window_total_m: int = 15,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()

    out = pairs.copy()
    out["pair_created_at"] = to_dt(out.get("pair_created_at"))
    out["snapshot_ts"]     = to_dt(out.get("snapshot_ts"))
    if "effective_created_at" not in out.columns:
        eff = out["pair_created_at"].where(out["pair_created_at"].notna(), out["snapshot_ts"])
        out["effective_created_at"] = eff

    out["first_trade_ts"] = pd.NaT
    out["time_to_first_trade_s"] = np.nan
    out["swaps_in_burst"] = 0.0
    out["swaps_per_min_burst"] = 0.0
    out["uniq_traders_10m"] = 0.0
    out["buy_ratio_15m"] = np.nan
    out["top5_concentration"] = np.nan
    out["lp_add_usd_15"] = 0.0
    out["lp_remove_usd_15"] = 0.0

    sw_by_pair = {k: v for k, v in (swaps.groupby("pair_address") if not swaps.empty else [])}
    pwm_by_pair = {k: v for k, v in (pwm.groupby("pair_address") if not pwm.empty else [])}
    le_by_pair  = {k: v for k, v in (lp_events.groupby("pair_address") if not lp_events.empty else [])}

    for i, row in out.iterrows():
        p = row.get("pair_address")
        created = row.get("effective_created_at")
        if not isinstance(p, str) or pd.isna(created):
            continue

        sw = sw_by_pair.get(p, pd.DataFrame())
        if not sw.empty:
            sw = sw[sw["ts"].fillna(pd.Timestamp.max.tz_localize("UTC")) >= created]

        first_ts = _first_trade_ts(sw)
        # Compute time to first trade using the original timezone-aware timestamp.  Do not
        # cast to naive before subtracting, as that would raise an exception when mixed with
        # timezone-aware 'created'.
        if pd.notna(first_ts):
            try:
                out.at[i, "time_to_first_trade_s"] = float(max(0.0, (first_ts - created).total_seconds()))
            except Exception:
                out.at[i, "time_to_first_trade_s"] = np.inf
        else:
            out.at[i, "time_to_first_trade_s"] = np.inf
        # Cast the stored first trade timestamp to a timezone‑naive datetime to avoid future
        # dtype incompatibility warnings.  We preserve the instant by removing the timezone
        # after conversion.
        if pd.notna(first_ts):
            try:
                cast_ts = pd.to_datetime(first_ts)
                if cast_ts.tzinfo is not None:
                    cast_ts = cast_ts.tz_convert(None)
            except Exception:
                cast_ts = pd.NaT
        else:
            cast_ts = pd.NaT
        out.at[i, "first_trade_ts"] = cast_ts

        if pd.notna(first_ts):
            burst_end = first_ts + pd.Timedelta(seconds=burst_window_s)
            burst = sw[(sw["ts"] >= first_ts) & (sw["ts"] <= burst_end)]
            swaps_in_burst = float(len(burst))
            swaps_per_min = swaps_in_burst / max(burst_window_s / 60.0, 1e-9)
            out.at[i, "swaps_in_burst"] = swaps_in_burst
            out.at[i, "swaps_per_min_burst"] = float(swaps_per_min)

            uniq_end = first_ts + pd.Timedelta(minutes=uniques_window_m)
            w = sw[(sw["ts"] >= first_ts) & (sw["ts"] <= uniq_end)] if not sw.empty else sw
            uniq_cnt = float(w["trader_wallet"].dropna().astype(str).nunique()) if ("trader_wallet" in w.columns and not w.empty) else 0.0
            out.at[i, "uniq_traders_10m"] = uniq_cnt

            sw_15 = sw[(sw["ts"] >= first_ts) & (sw["ts"] <= first_ts + pd.Timedelta(minutes=pwm_window_total_m))]
            if not sw_15.empty and "side" in sw_15.columns:
                is_buy = _buy_mask(sw_15)
                buys = float(is_buy.sum())
                sells = float((~is_buy).sum())
                out.at[i, "buy_ratio_15m"] = buys / max(buys + sells, 1.0)
            else:
                pwm_df = pwm_by_pair.get(p, pd.DataFrame())
                if not pwm_df.empty:
                    pwm_early = pwm_df[(pwm_df["snapshot_ts"] >= created) & (pwm_df["snapshot_ts"] <= created + pd.Timedelta(minutes=pwm_window_total_m))]
                    if not pwm_early.empty and {"buys","sells"}.issubset(pwm_early.columns):
                        buys = float(pd.to_numeric(pwm_early["buys"], errors="coerce").fillna(0).sum())
                        sells = float(pd.to_numeric(pwm_early["sells"], errors="coerce").fillna(0).sum())
                        out.at[i, "buy_ratio_15m"] = buys / max(buys + sells, 1.0)

            conc = np.nan
            if not sw_15.empty and "trader_wallet" in sw_15.columns:
                buys_df = sw_15[_buy_mask(sw_15)].copy()
                if not buys_df.empty:
                    amt = _amount_for_concentration(buys_df)
                    g = buys_df.assign(_amt=amt).groupby("trader_wallet")["_amt"].sum().sort_values(ascending=False)
                    total = float(g.sum())
                    top5 = float(g.head(5).sum())
                    conc = (top5 / total) if total > 0 else np.nan
            out.at[i, "top5_concentration"] = conc

        le_df = le_by_pair.get(p, pd.DataFrame())
        if not le_df.empty and pd.notna(created):
            span_end = created + pd.Timedelta(minutes=15)
            win = le_df[(le_df["ts"] >= created) & (le_df["ts"] <= span_end)]
            if not win.empty and {"action","value_usd"}.issubset(win.columns):
                add_usd = pd.to_numeric(win.loc[win["action"].astype(str).str.lower().eq("add"), "value_usd"], errors="coerce").fillna(0).sum()
                rem_usd = pd.to_numeric(win.loc[win["action"].astype(str).str.lower().eq("remove"), "value_usd"], errors="coerce").fillna(0).sum()
                out.at[i, "lp_add_usd_15"] = float(add_usd)
                out.at[i, "lp_remove_usd_15"] = float(rem_usd)

    return out

def score_and_classify(
    df: pd.DataFrame,
    *,
    ttf_ceil_s: int = 600,
    min_swaps_per_min: float = 20,
    min_uniques_10m: int = 50,
    buy_ratio_center: float = 0.55,
    buy_ratio_tol: float = 0.25,
    max_concentration: float = 0.70,
    leader_score_min: float = 60.0,
) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    vel   = out["swaps_per_min_burst"].astype(float).fillna(0.0)
    uniq  = out["uniq_traders_10m"].astype(float).fillna(0.0)
    br    = out["buy_ratio_15m"].astype(float)
    conc  = out["top5_concentration"].astype(float)
    lprem = out["lp_remove_usd_15"].astype(float).fillna(0.0)

    s_vel  = (vel / max(min_swaps_per_min, 1e-9)).clip(0, 1)
    s_uniq = (uniq / max(min_uniques_10m, 1e-9)).clip(0, 1)
    s_br   = 1.0 - (br.sub(buy_ratio_center).abs() / max(buy_ratio_tol, 1e-9))
    s_br   = s_br.clip(0, 1).fillna(0.5)
    s_conc = 1.0 - ((conc - 0.50) / 0.50)
    s_conc = s_conc.clip(0, 1).fillna(0.5)
    lp_pen = (lprem > 0.0).astype(float)

    gate_tradeable = out["time_to_first_trade_s"].astype(float).fillna(np.inf) <= float(ttf_ceil_s)
    gate_velocity  = vel  >= float(min_swaps_per_min)
    gate_uniques   = uniq >= float(min_uniques_10m)
    gate_br        = br.between(buy_ratio_center - buy_ratio_tol, buy_ratio_center + buy_ratio_tol, inclusive="both").fillna(True)
    gate_conc      = conc.fillna(0.50) <= float(max_concentration)
    gate_lp_ok     = ~(lprem > 0.0)

    score_01 = 0.35*s_vel + 0.35*s_uniq + 0.10*s_br + 0.20*s_conc
    score_01 = score_01 * (1.0 - 0.9 * lp_pen)
    out["early_score"] = (score_01.clip(0, 1) * 100.0).round(1)

    # Classification order is critical: strong tokens should not fall through to "Loser" simply because
    # the tradeability or LP removal gates are checked first.  We first check for Early Leader and Hype
    # classifications before assigning any Loser labels.  Only after evaluating those categories do we
    # fallback to Loser statuses based on trade activity and LP events.
    labels, reasons = [], []
    # Precompute float early scores once
    early_scores = out["early_score"].astype(float).fillna(0.0)
    for trd, vel_ok, unq_ok, br_ok, conc_ok, lp_ok, sc in zip(
        gate_tradeable, gate_velocity, gate_uniques, gate_br, gate_conc, gate_lp_ok, early_scores
    ):
        # Prioritize Early Leader classification
        if (vel_ok and unq_ok and br_ok and conc_ok and sc >= leader_score_min):
            labels.append("Early Leader")
            reasons.append("Velocity+Uniques+Balance+Dispersion")
            continue
        # Next consider borderline / hype tokens.  We treat any token with score >=35 or with high velocity
        # alongside either sufficient uniques or a balanced buy ratio as "Hype / Risky".  Reasons encode
        # missing gates to help explain borderline cases.
        if sc >= 35.0 or (vel_ok and (unq_ok or br_ok)):
            labels.append("Hype / Risky")
            missing = []
            if not conc_ok:
                missing.append("Concentration")
            if not unq_ok:
                missing.append("Uniques")
            if not br_ok:
                missing.append("BuyRatio")
            reasons.append(" & ".join(missing) if missing else "Borderline")
            continue
        # Only now consider outright losers based on trading and LP removal
        if not trd:
            labels.append("Loser (no early trades)")
            reasons.append("No trade <=10m")
            continue
        if not lp_ok:
            labels.append("Loser (early LP remove)")
            reasons.append("LP removed <=15m")
            continue
        # Default loser classification when thresholds are not met
        labels.append("Loser")
        why = []
        if not vel_ok:
            why.append("Low velocity")
        if not unq_ok:
            why.append("Low uniques")
        if not br_ok:
            why.append("Unbalanced flow")
        reasons.append("Weak" if not why else " & ".join(why))
    out["classification"] = labels
    out["reason"] = reasons
    return out

# ============================= UI =============================
st.title("TrenchFeed")
st.caption("Robust Top Coins link generation (Dexscreener, Solscan, Birdeye)")

# Show authenticated user info and logout button in sidebar header
with st.sidebar:
    customer = st.session_state.get("customer", {})
    if customer:
        tier = customer.get("tier", "unknown").upper()
        email = customer.get("email", "")
        st.markdown(f"""
            <div style='background: rgba(0, 201, 255, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 1rem;'>
                <strong>🔓 Logged In</strong><br>
                <small>Tier: {tier}</small><br>
                <small style='color: #888;'>{email[:20]}...</small> 
            </div>
        """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
    
    st.markdown("---")

with st.sidebar:
    # Encapsulate all controls in an expander for a cleaner, more organized look
    with st.expander("🛠️ Controls & Settings", expanded=True):
        # Manual refresh button with icon
        if st.button("🔄 Manual Refresh"):
            st.rerun()

        # Settings for scanning tokens and windows
        max_pairs       = st.slider("Max pairs to scan", 200, 10000, 2000, 100)
        recency_hours   = st.slider("Only tokens newer than (hours)", 1, 72, 2, 1)
        max_age_minutes = st.slider("Max token age for candidates (minutes)", 1, 240, 30, 1)
        use_snapshot_fallback = st.toggle("Use snapshot_ts when pair_created_at is NULL", value=True)

        st.markdown("---")
        st.markdown("**Early Leader thresholds**")
        min_swaps_per_min = st.slider("Min swaps/min in 2m burst", 1, 80, 20, 1)
        min_uniques_10m   = st.slider("Min unique traders in first 10m", 5, 200, 50, 5)
        buy_center        = st.slider("Buy ratio center", 0.40, 0.70, 0.55, 0.01)
        buy_tol           = st.slider("Buy ratio tolerance (±)", 0.05, 0.35, 0.25, 0.01)
        max_conc          = st.slider("Max Top-5 concentration", 0.50, 0.95, 0.70, 0.01)
        leader_score_min  = st.slider("Min Early Leader score", 0, 100, 60, 1)

        st.markdown("---")
        radar_window_m  = st.slider("Launch Radar: lookback minutes", 30, 240, 120, 15)
        radar_max       = st.slider("Launch Radar: max rows", 20, 400, 120, 20)
        detail_limit    = st.slider("Token Detail: rows per table", 50, 5000, 500, 50)

        st.markdown("---")
        st.markdown("**Auto Refresh**")
        auto_refresh_enabled = st.toggle("Enable Auto Refresh", value=False)
        # Allow the user to specify a refresh interval in seconds when auto refresh is enabled.
        refresh_interval = None
        if auto_refresh_enabled:
            refresh_interval = st.slider("Refresh interval (seconds)", 15, 600, 60, 15)
        # Schedule an automatic rerun of the script if auto refresh is on and the autorefresh
        # component is available.  Convert seconds to milliseconds as required by st_autorefresh.
        if auto_refresh_enabled and refresh_interval and callable(st_autorefresh):
            st_autorefresh(interval=int(refresh_interval * 1000), key="auto_refresh")

# Arrange tabs so that "All Candidates" appears first (leftmost).  The ordering
# is All Candidates, Early Leaders, Token Detail, Top Coins, and Launch Radar.
tab_all, tab_leaders, tab_detail, tab_top, tab_radar = st.tabs([
    "📋 All Candidates",
    "🏆 Early Leaders",
    "🔎 Token Detail",
    "🔴 LIVE",
    "🚀 Launch Radar"
])

# ============================= Early Leaders =============================
with tab_leaders:
    st.subheader("Early Leaders")
    # Disclaimer message across all tabs
    display_disclaimer()
    pairs = fetch_recent_pairs(max_pairs, recency_hours, max_age_minutes, use_snapshot_fallback=use_snapshot_fallback)
    if pairs.empty:
        st.info("No recent pairs in the selected window.")
    else:
        earliest_eff = pairs["effective_created_at"].min()
        since_iso = iso(earliest_eff - pd.Timedelta(minutes=1)) if pd.notna(earliest_eff) else iso_hours_ago(recency_hours)

        pair_ids = pairs["pair_address"].dropna().astype(str).unique().tolist()
        swaps = fetch_swaps_for_pairs(pair_ids, since_iso)
        pwm   = fetch_pwm_for_pairs(pair_ids, since_iso)
        lpev  = fetch_lp_events_for_pairs(pair_ids, since_iso)

        metrics = compute_early_metrics(pairs, swaps, pwm, lpev)
        ranked  = score_and_classify(metrics,
                                     ttf_ceil_s=600,
                                     min_swaps_per_min=float(min_swaps_per_min),
                                     min_uniques_10m=int(min_uniques_10m),
                                     buy_ratio_center=float(buy_center),
                                     buy_ratio_tol=float(buy_tol),
                                     max_concentration=float(max_conc),
                                     leader_score_min=float(leader_score_min))

        ranked = ensure_pair_links(ranked, token_col="token_address")
        ranked = attach_token_names(ranked, token_col="token_address")
        ranked = add_links(ranked)
        # Sort the Early Leaders view by descending early_score to surface the highest
        # scoring tokens at the top.  If early_score is missing the sort has no effect.
        if not ranked.empty and "early_score" in ranked.columns:
            ranked = ranked.sort_values(["early_score"], ascending=[False]).reset_index(drop=True)

        cols = [
            "early_score","classification","reason",
            "swaps_per_min_burst","uniq_traders_10m","buy_ratio_15m","top5_concentration",
            # LP columns are removed from the display in Early Leaders; metrics still include them
            "token_address","name","symbol",
            "dexscreener","solscan","birdeye",
            "pair_address","effective_created_at","first_trade_ts","time_to_first_trade_s",
        ]
        shown = [c for c in cols if c in ranked.columns]
        # Display summary metrics for Early Leaders.  These indicators give a quick glance at
        # how many tokens fall into each classification and how strong they are on average.
        if not ranked.empty:
            leader_count = int((ranked.get("classification") == "Early Leader").sum()) if "classification" in ranked.columns else 0
            hype_count   = int((ranked.get("classification") == "Hype / Risky").sum()) if "classification" in ranked.columns else 0
            loser_count  = int(ranked.get("classification").astype(str).str.contains("Loser").sum()) if "classification" in ranked.columns else 0
            avg_score    = float(ranked.get("early_score").mean()) if "early_score" in ranked.columns else float('nan')
            max_score    = float(ranked.get("early_score").max()) if "early_score" in ranked.columns else float('nan')
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Leaders", f"{leader_count}")
            c2.metric("Hype", f"{hype_count}")
            c3.metric("Losers", f"{loser_count}")
            c4.metric("Avg Score", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
            c5.metric("Top Score", f"{max_score:.1f}" if not pd.isna(max_score) else "N/A")
        st.dataframe(ranked[shown].reset_index(drop=True), use_container_width=True, height=520, column_config=link_config(shown))

# ============================= All Candidates =============================
with tab_all:
    st.subheader("All Candidates")
    display_disclaimer()
    since_iso = iso_hours_ago(recency_hours)
    pairs = fetch_table(
        "pairs",
        select=("pair_address,base_token,quote_token,price_usd,fdv_usd,market_cap_usd,"
                "pair_created_at,snapshot_ts,base_token_name,base_token_symbol,quote_token_name,quote_token_symbol"),
        where={"snapshot_ts": f"gte.{since_iso}"},
        order="snapshot_ts.desc.nullslast",
        limit=max_pairs,
    )
    if pairs.empty:
        st.info("No recent pairs.")
    else:
        pairs = numeric(pairs, ["price_usd","fdv_usd","market_cap_usd"])
        for tcol in ["pair_created_at","snapshot_ts"]:
            pairs[tcol] = to_dt(pairs[tcol])
        eff = pairs["pair_created_at"].where(pairs["pair_created_at"].notna(), pairs["snapshot_ts"])
        min_ts = now_utc() - pd.Timedelta(minutes=max_age_minutes)
        pairs = pairs.loc[eff >= min_ts].copy()
        pairs["effective_created_at"] = eff
        pairs = pairs.rename(columns={"base_token": "token_address"})
        pairs = ensure_pair_links(pairs, token_col="token_address")
        pairs = attach_token_names(pairs, token_col="token_address")
        pairs = add_links(pairs)

        cols = [
            "price_usd","fdv_usd","market_cap_usd","effective_created_at","snapshot_ts",
            "token_address","name","symbol",
            "dexscreener","solscan","birdeye",
            "pair_address","quote_token_symbol","quote_token_name",
        ]
        shown = [c for c in cols if c in pairs.columns]
        st.dataframe(pairs[shown].reset_index(drop=True), use_container_width=True, height=620, column_config=link_config(shown))

# ============================= Token Detail =============================
with tab_detail:
    st.subheader("Token Detail")
    display_disclaimer()
    q_token = st.text_input("Token address", help="Paste token address")
    if q_token:
        if st.button("Fetch latest Data"):
            import subprocess
            helius_path = "/opt/sol/etl/hel.py"
            try:
                subprocess.run(["python3", helius_path, q_token], check=False)
                st.success("Refreshed from Helius. You can now view updated tables.")
            except Exception as e:
                st.warning(f"Helius refresh failed: {e}")
        tok = fetch_table("tokens", select="token_address,chain_id,name,symbol,updated_at", where={"token_address": f"eq.{q_token}"}, limit=1)
        if not tok.empty: tok["updated_at"] = to_dt(tok.get("updated_at"))
        st.write("Tokens")
        st.dataframe(tok.reset_index(drop=True), use_container_width=True, height=120)

        tstate = fetch_table("token_state", select="*", where={"token_address": f"eq.{q_token}"}, limit=1)
        if not tstate.empty and "last_window" in tstate.columns:
            tstate["last_window"] = to_dt(tstate["last_window"])
        st.write("Token State")
        st.dataframe(tstate.reset_index(drop=True), use_container_width=True, height=150)

        creators = fetch_table("creators", select="*", where={"token_address": f"eq.{q_token}"}, limit=detail_limit)
        if not creators.empty and "created_at" in creators.columns:
            creators["created_at"] = to_dt(creators["created_at"])
        st.write("Creators")
        st.dataframe(creators.reset_index(drop=True), use_container_width=True, height=150)

        holders = fetch_table("holder_snapshots", select="*", where={"token_address": f"eq.{q_token}"}, order="snapshot_ts.desc", limit=detail_limit)
        if not holders.empty: holders["snapshot_ts"] = to_dt(holders.get("snapshot_ts"))
        st.write("Holder Snapshots")
        st.dataframe(holders.reset_index(drop=True), use_container_width=True, height=220)

        rflags = fetch_table("risk_flags", select="*", where={"token_address": f"eq.{q_token}"}, order="ts.desc", limit=detail_limit)
        if not rflags.empty: rflags["ts"] = to_dt(rflags.get("ts"))
        st.write("Risk Flags")
        st.dataframe(rflags.reset_index(drop=True), use_container_width=True, height=180)

        pairs_q = fetch_table(
            "pairs",
            select=("pair_address,chain_id,dex_id,base_token,quote_token,price_usd,fdv_usd,market_cap_usd,"
                    "pair_created_at,snapshot_ts,base_token_name,base_token_symbol,quote_token_name,quote_token_symbol"),
            where={"or": f"(base_token.eq.{q_token},quote_token.eq.{q_token})"},
            order="snapshot_ts.desc.nullslast",
            limit=detail_limit,
        )
        if not pairs_q.empty:
            pairs_q = numeric(pairs_q, ["price_usd","fdv_usd","market_cap_usd"])
            for tcol in ["pair_created_at","snapshot_ts"]:
                pairs_q[tcol] = to_dt(pairs_q[tcol])
            pairs_q = pairs_q.rename(columns={"base_token":"token_address"})
            pairs_q["effective_created_at"] = pairs_q["pair_created_at"].where(pairs_q["pair_created_at"].notna(), pairs_q["snapshot_ts"])
            pairs_q = ensure_pair_links(pairs_q, token_col="token_address")
            pairs_q = attach_token_names(pairs_q, token_col="token_address")
            pairs_q = add_links(pairs_q)
        st.write("Pairs")
        st.dataframe(pairs_q.reset_index(drop=True), use_container_width=True, height=250,
                     column_config=link_config(list(pairs_q.columns) if not pairs_q.empty else []))

        pair_ids: List[str] = list(pairs_q["pair_address"].dropna().unique()) if not pairs_q.empty else []
        pair_ids_str = ",".join(pair_ids[:500])

        for table, title in [
            ("pair_window_metrics","Pair Window Metrics"),
            ("pair_price_snapshots","Pair Price Snapshots"),
            ("features","Features"),
            ("liquidity_events","Liquidity Events"),
            ("listings","Listings"),
            ("swaps","Swaps"),
        ]:
            where = {"pair_address": f"in.({pair_ids_str})"} if pair_ids_str else None
            order = "snapshot_ts.desc" if table in ("pair_window_metrics","pair_price_snapshots") else "ts.desc"
            df = fetch_table(table, select="*", where=where, order=order, limit=detail_limit)
            tscol = "snapshot_ts" if (not df.empty and "snapshot_ts" in df.columns) else ("ts" if (not df.empty and "ts" in df.columns) else None)
            if not df.empty and tscol: df[tscol] = to_dt(df[tscol])
            st.write(title)
            st.dataframe(df.reset_index(drop=True), use_container_width=True, height=220)

        chains = fetch_table("chains", select="*", where={"chain_id": f"eq.{tok.iloc[0]['chain_id']}"} if not tok.empty else None, limit=1)
        st.write("Chains")
        st.dataframe(chains.reset_index(drop=True), use_container_width=True, height=100)

        wallet_ids: Set[str] = set()
        if not creators.empty and "wallet_id" in creators.columns:
            wallet_ids.update([w for w in creators["wallet_id"].dropna().astype(str).tolist()])
        if pair_ids:
            sw = fetch_table("swaps", select="trader_wallet", where={"pair_address": f"in.({','.join(pair_ids[:400])})"}, limit=5000)
            if not sw.empty and "trader_wallet" in sw.columns:
                wallet_ids.update(sw["trader_wallet"].dropna().astype(str).tolist())
        wl = list(sorted(wallet_ids))[:200]
        wlabels = fetch_table("wallet_labels", select="*", where={"wallet_id": "in.(" + ",".join(wl) + ")"} if wl else None, limit=len(wl) or 1)
        st.write("Wallet Labels")
        st.dataframe(wlabels.reset_index(drop=True), use_container_width=True, height=150)
        w = fetch_table("wallets", select="*", where={"wallet_id": "in.(" + ",".join(wl) + ")"} if wl else None, limit=len(wl) or 1)
        if not w.empty: w["first_seen"] = to_dt(w.get("first_seen"))
        st.write("Wallets")
        st.dataframe(w.reset_index(drop=True), use_container_width=True, height=200)

# ============================= Top Coins (LIVE) =============================
with tab_top:
    # Transform the Top Coins tab into a live streaming dashboard.  The live feed
    # connects to the enterprise streaming endpoint and renders incoming events
    # into a table on the fly.  Users can click "Refresh Feed" to restart
    # the connection.
    st.subheader("Live Market Data Stream")
    # Display the disclaimer prominently
    display_disclaimer()
    st.caption("Real‑time streaming of market events from TrenchFeed Enterprise API. \n"
               "Click the button below to re‑establish the feed at any time.")
    # Call‑to‑action inviting users to subscribe for uninterrupted access
    st.markdown(
        "<div style='margin: 0.5rem 0;'>"
        "For continuous, high‑fidelity data streams, head over to "
        "<a href='https://trenchfeed.cc' target='_blank' style='font-weight:bold; color:#FFB900;'>"
        "trenchfeed.cc</a> and grab a subscription.</div>",
        unsafe_allow_html=True,
    )
    
    # Initialize stream control state
    if "stream_paused" not in st.session_state:
        st.session_state.stream_paused = False
    if "stream_data" not in st.session_state:
        st.session_state.stream_data = []
    if "stream_initialized" not in st.session_state:
        st.session_state.stream_initialized = False
    
    # Create button row with Refresh and Pause/Resume side by side
    btn_col1, btn_col2, status_col = st.columns([1, 1, 2])
    
    with btn_col1:
        refresh_feed = st.button("🔄 Refresh Feed", key="refresh_feed", use_container_width=True)
    
    with btn_col2:
        # Toggle pause/resume state
        is_paused = st.session_state.stream_paused
        pause_label = "▶️ Resume" if is_paused else "⏸️ Pause"
        if st.button(pause_label, key="pause_resume", use_container_width=True):
            st.session_state.stream_paused = not st.session_state.stream_paused
            st.rerun()
    
    with status_col:
        # Display current stream status with visual indicator
        if st.session_state.stream_paused:
            st.markdown(
                '<div class="stream-status paused">'
                '<span class="pulse"></span>'
                'PAUSED'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="stream-status live">'
                '<span class="pulse"></span>'
                'STREAMING'
                '</div>',
                unsafe_allow_html=True
            )
    
    # Handle refresh - reset data and re-initialize
    if refresh_feed:
        st.session_state.stream_data = []
        st.session_state.stream_initialized = False
        st.session_state.stream_paused = False
        st.rerun()
    
    # Placeholder for the live table
    live_placeholder = st.empty()
    
    # Define a mapping from raw column names to degen‑friendly jargon.  These
    # names resonate with crypto traders while remaining self‑explanatory.
    _live_rename_map = {
        "payload.token.name": "Token Name",
        "payload.token.symbol": "Ticker",
        "payload.pair.base_token": "Base Token",
        "payload.pair.dex": "DEX",
        "payload.pair.price_usd": "Price USD",
        "payload.current_swap.side": "Side",
        "payload.current_swap.price_usd": "Swap Price USD",
        "payload.current_swap.usd_value": "Swap Value USD",
        "payload.current_swap.sol_amount": "SOL Amount",
        "payload.current_swap.token_amount": "Token Amount",
        "payload.current_swap.trader_wallet": "Trader Wallet",
        "payload.timestamp": "Event Timestamp",
    }
    
    def render_stream_table():
        """Render the current stream data as a table."""
        if not st.session_state.stream_data:
            live_placeholder.info("No data yet. Waiting for stream events...")
            return
        
        try:
            df_stream = pd.json_normalize(st.session_state.stream_data)
        except Exception:
            df_stream = pd.DataFrame(st.session_state.stream_data)
        
        # Drop the recent_swaps column if present, as it contains opaque objects
        if "payload.recent_swaps" in df_stream.columns:
            df_stream = df_stream.drop(columns=["payload.recent_swaps"])
        
        # Rename columns using the friendly mapping where available
        rename_map = {k: v for k, v in _live_rename_map.items() if k in df_stream.columns}
        df_stream = df_stream.rename(columns=rename_map)
        
        # Determine primary columns (after rename, if rename occurred)
        primary_raw = ["payload.token.name", "payload.token.symbol", "payload.pair.base_token"]
        primary_cols = [rename_map.get(col, _live_rename_map.get(col, col)) for col in primary_raw if (rename_map.get(col, _live_rename_map.get(col, col)) in df_stream.columns)]
        
        # Build an ordered list of columns: primary first, then the rest
        other_cols = [c for c in df_stream.columns if c not in primary_cols]
        ordered_cols = primary_cols + other_cols
        
        # Take the last 50 events and reverse order so newest events appear first
        df_show = df_stream[ordered_cols].tail(50).iloc[::-1].copy()
        
        # Assign descending row numbers (highest at top) as the index
        n_rows = len(df_show)
        df_show.index = range(n_rows, 0, -1)
        
        # Display the table
        live_placeholder.dataframe(
            df_show,
            use_container_width=True,
            height=620,
        )
    
    def run_live_stream():
        """Internal helper to consume streaming data and update the table."""
        # Collect a bounded number of events to avoid infinite loops
        for event in stream_enterprise(max_events=200):
            # Check if stream is paused - if so, stop consuming new events
            # Note: Due to Streamlit's execution model, this check happens
            # between events. The pause takes effect after the current event.
            if st.session_state.stream_paused:
                break
            
            # Append the raw event to our session state
            st.session_state.stream_data.append(event)
            
            # Render the updated table
            render_stream_table()
    
    # If paused, just render the existing data
    if st.session_state.stream_paused:
        render_stream_table()
    # Start the stream when the component first renders or upon refresh
    elif not st.session_state.get("stream_initialized"):
        st.session_state.stream_initialized = True
        run_live_stream()
    else:
        # Already initialized and not paused - continue streaming
        run_live_stream()

# ============================= Launch Radar =============================
with tab_radar:
    st.subheader("Launch Radar")
    display_disclaimer()
    lookback_iso = iso(now_utc() - pd.Timedelta(minutes=radar_window_m))
    recent_pairs = fetch_table(
        "pairs",
        select="pair_address,token_address:base_token,base_token_name,base_token_symbol,pair_created_at,snapshot_ts,price_usd,fdv_usd,market_cap_usd",
        where={"pair_created_at": f"gte.{lookback_iso}"},
        order="pair_created_at.desc.nullslast",
        limit=2000
    )
    if recent_pairs.empty or recent_pairs["pair_created_at"].isna().all():
        extra = fetch_table(
            "pairs",
            select="pair_address,token_address:base_token,base_token_name,base_token_symbol,pair_created_at,snapshot_ts,price_usd,fdv_usd,market_cap_usd",
            where={"snapshot_ts": f"gte.{lookback_iso}"},
            order="snapshot_ts.desc.nullslast",
            limit=2000
        )
        if not extra.empty:
            recent_pairs = extra

    if recent_pairs.empty:
        st.info("No very recent launches.")
    else:
        recent_pairs["pair_created_at"] = to_dt(recent_pairs["pair_created_at"])
        recent_pairs["snapshot_ts"]     = to_dt(recent_pairs["snapshot_ts"])
        eff = recent_pairs["pair_created_at"].where(recent_pairs["pair_created_at"].notna(), recent_pairs["snapshot_ts"])
        recent_pairs["effective_created_at"] = eff

        pair_ids = recent_pairs["pair_address"].dropna().astype(str).unique().tolist()
        swaps = fetch_swaps_for_pairs(pair_ids, lookback_iso)
        pwm   = fetch_pwm_for_pairs(pair_ids, lookback_iso)
        lpev  = fetch_lp_events_for_pairs(pair_ids, lookback_iso)

        metrics = compute_early_metrics(recent_pairs, swaps, pwm, lpev)
        ranked  = score_and_classify(metrics,
                                     ttf_ceil_s=600,
                                     min_swaps_per_min=float(min_swaps_per_min),
                                     min_uniques_10m=int(min_uniques_10m),
                                     buy_ratio_center=float(buy_center),
                                     buy_ratio_tol=float(buy_tol),
                                     max_concentration=float(max_conc),
                                     leader_score_min=float(leader_score_min))

        ranked = ensure_pair_links(ranked, token_col="token_address")
        ranked = attach_token_names(ranked, token_col="token_address")
        ranked = add_links(ranked)

        if "market_cap_usd" in ranked.columns:
            ranked = numeric(ranked, ["market_cap_usd"])
            ranked = ranked[ranked["market_cap_usd"].fillna(0) >= 30000]

        cols = [
            "effective_created_at",
            "token_address","name","symbol",
            "dexscreener","solscan","birdeye",
            "early_score","classification","reason",
            "swaps_per_min_burst","uniq_traders_10m","buy_ratio_15m","top5_concentration",
            # LP columns are intentionally excluded from Launch Radar display
            "pair_address",
        ]
        shown = [c for c in cols if c in ranked.columns]
        # Sort Launch Radar strictly by highest early_score first.  This ensures that top-scoring
        # tokens (actual winners) appear first instead of being grouped by classification or
        # alphabetical order.  We drop secondary sorting by classification/effective_created_at.
        ranked = ranked.sort_values(["early_score"], ascending=[False]).head(radar_max).reset_index(drop=True)
        # Display summary metrics for Launch Radar.  These metrics provide a quick overview
        # of the newly launched pairs and how many qualify as leaders or hype tokens.
        total_pairs    = len(ranked)
        lr_leaders     = int((ranked.get("classification") == "Early Leader").sum()) if "classification" in ranked.columns else 0
        lr_hype        = int((ranked.get("classification") == "Hype / Risky").sum()) if "classification" in ranked.columns else 0
        lr_avg_score   = float(ranked.get("early_score").mean()) if "early_score" in ranked.columns else float('nan')
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("New Pairs", f"{total_pairs}")
        col_b.metric("Leaders", f"{lr_leaders}")
        col_c.metric("Hype", f"{lr_hype}")
        col_d.metric("Avg Score", f"{lr_avg_score:.1f}" if not pd.isna(lr_avg_score) else "N/A")
        st.dataframe(ranked[shown], use_container_width=True, height=640, column_config=link_config(shown))
