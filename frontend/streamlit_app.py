
import streamlit as st
import os
import requests
import pandas as pd
import jwt

def cfg(name, default=""):
    val = os.environ.get(name, default)
    try:
        val = st.secrets.get(name, val)
    except:
        pass
    return str(val).strip()

SB_URL = cfg("SUPABASE_URL").rstrip("/")
SB_KEY = cfg("SUPABASE_SERVICE_ROLE")
SCHEMA = cfg("SUPABASE_SCHEMA", "public")
JWT_SECRET = cfg("JWT_SECRET")

session = requests.Session()
session.headers.update({
    "apikey": SB_KEY,
    "Authorization": f"Bearer {SB_KEY}",
    "Content-Type": "application/json",
    "Accept-Profile": SCHEMA,
    "Content-Profile": SCHEMA,
})

def fetch_table(table, select="*", where=None, order=None, limit=1000):
    params = {"select": select, "limit": str(limit)}
    if where: params.update(where)
    if order: params["order"] = order
    try:
        r = session.get(f"{SB_URL}/rest/v1/{table}", params=params, timeout=15)
        if r.status_code in (200, 206):
            return pd.DataFrame(r.json())
    except Exception as e:
        st.warning(f"Fetch error: {e}")
    return pd.DataFrame()

def verify_token():
    token = st.query_params.get("token")
    if not token:
        st.warning("Missing access token - Check Subscription.")
        st.stop()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["plan"]
    except Exception as e:
        st.error(f"Invalid token: {e}")
        st.stop()

st.set_page_config(page_title="TrenchFeed Dashboard", layout="wide")
st.title("📊 TrenchFeed: Price action - Straight From the Front Lines.")

plan = verify_token()
st.subheader(f"Access Level: `{plan}`")

view_map = {
    "basic": "basic_view",
    "pro": "pro_view",
    "enterprise": "enterprise_view"
}
view = view_map.get(plan, "basic_view")

df = fetch_table(view, order="snapshot_ts.desc", limit=200)

if df.empty:
    st.info("No data returned.")
else:
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{plan}_data.csv")
