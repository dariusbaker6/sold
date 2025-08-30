
from fastapi import FastAPI, Request, HTTPException
import jwt, stripe, os
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI()

JWT_SECRET = os.getenv("JWT_SECRET", "pumpdottest")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://your-app.streamlit.app")
STRIPE_SECRET = os.getenv("STRIPE_SECRET", "sk_test_...")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "whsec_...")
stripe.api_key = STRIPE_SECRET

@app.get("/get-token")
def manual_token(plan: str = "pro", days: int = 30):
    token = jwt.encode({
        "plan": plan,
        "exp": datetime.utcnow() + timedelta(days=days)
    }, JWT_SECRET, algorithm="HS256")
    return {"token": token, "url": f"{DASHBOARD_URL}/?token={token}"}

@app.get("/")
def root_ui():
    return HTMLResponse("""
        <html><body>
        <h2>Generate Token</h2>
        <form action="/get-token">
            <label>Plan:</label>
            <select name="plan">
                <option value="basic">Basic</option>
                <option value="pro">Pro</option>
                <option value="enterprise">Enterprise</option>
            </select>
            <label>Days valid:</label>
            <input type="number" name="days" value="30"/>
            <button type="submit">Generate</button>
        </form>
        </body></html>
    """)

@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session.get("customer_email", "unknown")
        metadata = session.get("metadata", {}) or {}
        plan = metadata.get("plan", "basic")
        token = jwt.encode({
            "plan": plan,
            "exp": datetime.utcnow() + timedelta(days=30)
        }, JWT_SECRET, algorithm="HS256")
        dashboard_link = f"{DASHBOARD_URL}/?token={token}"
        print(f"✅ Payment from {customer_email} → {plan} plan")
        print(f"🔑 Access: {dashboard_link}")
    return JSONResponse({"status": "ok"})
