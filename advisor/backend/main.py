"""
advisor/backend/main.py
==========================
Voice-QL Advisor Edition v2.0
Author: Mohamed Attia — Voice-QI / DataOverflow Team
Model:  gemini-2.5-flash-native-audio-preview-12-2025

Phase 2: Voice-Activated Insurance Advisor
- Gemini Live duplex audio WebSocket (/ws/advisor)
- On TURN_COMPLETE: preprocess → predict → SHAP → Qdrant → JSON packet
- REST endpoints: POST /predict, GET /health
"""

import os
import sys
import re
import json
import asyncio
import logging
import traceback
from typing import Any, Optional
from time import perf_counter

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    from google import genai
    from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv(): pass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

# ── Path Setup ────────────────────────────────────────────────────────────────
try:
    from .solution import preprocess, _load_artifact
except ImportError:
    from solution import preprocess, _load_artifact

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ID       = "gemini-2.5-flash-native-audio-preview-12-2025"
QDRANT_PATH    = os.path.join(os.path.dirname(__file__), "..", "qdrant_data")
COLLECTION     = "insurance_clients"
PORT           = int(os.getenv("PORT", 8002))

CAT_COLS = [
    'Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
    'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
    'Policy_Start_Month'
]

# ── Global State ──────────────────────────────────────────────────────────────
model_artifact: Optional[dict] = None
explainer:      Optional[shap.TreeExplainer] = None
qdrant:         Optional[QdrantClient] = None
feat_cols:      list[str] = []

# ── Insurance Bundle Definitions (injected into system prompt) ────────────────
BUNDLE_DEFINITIONS = """
INSURANCE BUNDLE CATALOG (Purchased_Coverage_Bundle classes 0–9):
- Bundle 0 — Basic Essential: Minimal coverage, single individual, no riders, high deductible.
- Bundle 1 — Solo Starter: Individual plan with basic outpatient + emergency coverage.
- Bundle 2 — Young Family: Covers couple + 1–2 children, mid deductible, Monthly payment.
- Bundle 3 — Established Family: Full family coverage, inpatient + outpatient, medium riders.
- Bundle 4 — Senior Shield: Targeted at older policyholders, chronic illness riders, low deductible.
- Bundle 5 — Professional Premium: High-income individual, dental/vision riders, Annual payment.
- Bundle 6 — Corporate Group: Employer-sponsored, multi-vehicle, low deductible, Annual schedule.
- Bundle 7 — Premium Family Plus: Top-tier family plan, all riders, grace periods, low deductible.
- Bundle 8 — High-Risk Managed: Clients with previous claims, structured payment schedule, capped riders.
- Bundle 9 — Elite Comprehensive: Maximum coverage, all categories, Custom Riders + Grace extensions, Annual.
"""

# ── Data Field Definitions (from DataQuest Brief) ─────────────────────────────
DATA_DEFINITIONS = """
INSURANCE DATA FIELD REFERENCE:
Demographics:
- Adult_Dependents / Child_Dependents / Infant_Dependents: People covered under the policy.
- Estimated_Annual_Income: Household yearly income (numeric).
- Employment_Status: Working arrangement (Full-Time, Part-Time, Self-Employed, Retired, etc.).
- Region_Code: Anonymized geographic region (R_01 to R_20).
Customer History:
- Existing_Policyholder: 1 if already active policyholder, 0 otherwise.
- Previous_Claims_Filed: Number of prior insurance claims filed.
- Years_Without_Claims: Consecutive clean years with no claim.
- Previous_Policy_Duration_Months: Duration of last policy in months.
- Policy_Cancelled_Post_Purchase: 1 if user cancelled policy shortly after buying.
Policy Preferences:
- Deductible_Tier: Out-of-pocket deductible level (Low, Medium, High).
- Payment_Schedule: Premium frequency (Monthly, Quarterly, Annual).
- Vehicles_on_Policy: Number of vehicles covered.
- Custom_Riders_Requested: Add-on coverage count.
- Grace_Period_Extensions: Times the payment deadline was extended.
Engineered Features (computed by preprocess()):
- Total_Dependents: Adult + Child + Infant dependents.
- Income_per_Dependent: Income divided by (Total_Dependents + 1).
- Risk_Ratio: Claims / (Years_Without_Claims + 1).
"""

# ── System Prompts ───────────────────────────────────────────────────────

VOICE_PROMPT = """You are the "InsureGuard AI Spokesman". You are calling a client to gather missing insurance data (e.g. income, dependents, claim history).
CRITICAL RULES:
1. Keep all responses under 2 concise sentences. Be punchy, fast, and professional.
2. Only speak in English.
3. If this is the start of the conversation, introduce yourself and ask a single quick question to begin profiling them.
4. Do NOT output any JSON, markup tags, or special characters. Speak naturally.
"""

EXTRACTION_PROMPT = f"""
You are the "Intelligent Insurance Strategist" extraction engine.
Listen to the transcribed conversation and extract insurance features into a JSON block.

FEATURE EXTRACTION FORMAT — always emit this tag when you have any data:
<FEATURES>
{{
  "Estimated_Annual_Income": <number or null>,
  "Employment_Status": "<string or null>",
  "Region_Code": "<string or null>",
  "Adult_Dependents": <int or null>,
  "Child_Dependents": <int or null>,
  "Infant_Dependents": <int or null>,
  "Previous_Claims_Filed": <int or null>,
  "Years_Without_Claims": <int or null>,
  "Deductible_Tier": "<Low|Medium|High or null>",
  "Acquisition_Channel": "<string or null>",
  "Payment_Schedule": "<Monthly|Quarterly|Annual or null>",
  "Policy_Start_Month": "<string or null>",
  "Existing_Policyholder": <0|1|null>,
  "Previous_Policy_Duration_Months": <int or null>,
  "Vehicles_on_Policy": <int or null>,
  "Custom_Riders_Requested": <int or null>,
  "Grace_Period_Extensions": <int or null>,
  "Policy_Cancelled_Post_Purchase": <0|1|null>
}}
</FEATURES>

{BUNDLE_DEFINITIONS}
{DATA_DEFINITIONS}
CRITICAL RULES:
- Emit <FEATURES> even with partial data (use null for unknown fields).
"""

# Configure session - MUST use dictionary for this model and ["AUDIO"] ONLY
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": VOICE_PROMPT
}

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifact, explainer, qdrant, feat_cols

    logger.info("🚀 Starting Voice-QL Advisor backend …")

    # 1. Load Phase 1 model
    logger.info("   Loading model.pkl …")
    model_artifact = _load_artifact()
    feat_cols      = model_artifact["feature_cols"]
    lgbm_model     = model_artifact["model"]       # sklearn wrapper
    # Note: SHAP TreeExplainer removed in favor of native booster.predict(pred_contrib=True)
    # This is faster and avoids categorical mismatch issues.
    explainer = None 

    # 3. Connect to file-based Qdrant
    if os.path.isdir(QDRANT_PATH):
        logger.info(f"   Connecting to Qdrant at {QDRANT_PATH} …")
        try:
            qdrant = QdrantClient(path=QDRANT_PATH)
            info   = qdrant.get_collection(COLLECTION)
            logger.info(f"   ✅ Qdrant: {info.points_count:,} vectors in '{COLLECTION}'.")
        except Exception as e:
            logger.warning(f"   ⚠️ Qdrant failed: {e}. Run ingest_qdrant.py first.")
            qdrant = None
    else:
        logger.warning(f"   ⚠️ Qdrant data not found at {QDRANT_PATH}. Run ingest_qdrant.py first.")

    yield

    logger.info("🛑 Backend shutting down.")


app = FastAPI(title="Voice-QL Advisor", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"❌ 422 Unprocessable Entity\nURL: {request.url}\nBody: {body.decode()}\nErrors: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": body.decode()},
    )

# ── Gemini Client ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Helpers ────────────────────────────────────────────────────────────────────
FEATURE_DEFAULTS = {
    "Estimated_Annual_Income": 50000.0,
    "Employment_Status": "Full-Time",
    "Region_Code": "Unknown",
    "Adult_Dependents": 0,
    "Child_Dependents": 0,
    "Infant_Dependents": 0,
    "Previous_Claims_Filed": 0,
    "Years_Without_Claims": 1,
    "Deductible_Tier": "Medium",
    "Acquisition_Channel": "Online",
    "Payment_Schedule": "Monthly",
    "Policy_Start_Month": "January",
    "Existing_Policyholder": 0,
    "Previous_Policy_Duration_Months": 12,
    "Vehicles_on_Policy": 1,
    "Custom_Riders_Requested": 0,
    "Grace_Period_Extensions": 0,
    "Policy_Cancelled_Post_Purchase": 0,
    # cols that get dropped in preprocess but needed for input shape
    "Policy_Start_Year": 2024,
    "Policy_Start_Week": 1,
    "Policy_Start_Day": 1,
    "Days_Since_Quote": 30,
    "Underwriting_Processing_Days": 5,
    "Policy_Amendments_Count": 0,
    "Broker_Agency_Type": "Independent",
    "Broker_ID": -1,
    "Employer_ID": -1,
    "Risk_Score_Proxy": 0,
    "User_ID": "USR_000000",
}


def build_dataframe(raw: dict) -> pd.DataFrame:
    """Merge raw extracted fields with defaults and return a single-row DataFrame."""
    row = {**FEATURE_DEFAULTS}
    for k, v in raw.items():
        if v is not None and k in row:
            row[k] = v
    return pd.DataFrame([row])


def run_pipeline(raw: dict) -> dict:
    """
    Full advisory pipeline:
      1. Build DataFrame from extracted features
      2. preprocess() → all-numeric
      3. predict() using optimised NumPy path
      4. SHAP top-3
      5. Qdrant top-3 lookalikes
    Returns a dict ready to send over WebSocket.
    """
    t0 = perf_counter()

    # ── 1 + 2: Build & preprocess ─────────────────────────────────────
    df_raw  = build_dataframe(raw)
    df_proc = preprocess(df_raw)

    # ── 3: FAST predict (NumPy path, mirrors solution.py predict()) ───
    booster = model_artifact["model"].booster_
    X       = df_proc[feat_cols].to_numpy(dtype="float32")
    proba   = booster.predict(X, num_threads=1)           # raw proba matrix
    pred    = int(np.argmax(proba, axis=1)[0])
    confidence = float(proba[0][pred])

    predict_ms = (perf_counter() - t0) * 1000
    logger.info(f"   predict(): {predict_ms:.1f}ms → Bundle {pred} ({confidence:.1%})")

    # ── 4: SHAP (Native LightGBM path) ────────────────────────────────
    shap_data = []
    try:
        # pred_contrib=True returns [n_samples, n_classes * (n_features + 1)]
        # This is the fastest way to get local SHAP values for LightGBM.
        contribs = booster.predict(X, pred_contrib=True)
        
        # Slicing: for multiclass, columns are grouped by class.
        # Each class block has (n_features + 1) columns (features + bias).
        n_feats = len(feat_cols)
        start_col = pred * (n_feats + 1)
        end_col   = start_col + n_feats
        class_sv  = contribs[0, start_col:end_col]
        
        # Top 3 features by absolute magnitude
        top_idx = np.argsort(np.abs(class_sv))[::-1][:3]
        shap_data = [
            {
                "feature": feat_cols[i],
                "value":   float(class_sv[i]),
                "data":    float(X[0][i]),
            }
            for i in top_idx
        ]
    except Exception as e:
        logger.error(f"   Native SHAP failed: {e}")
        # Global importance fallback
        try:
            importances = model_artifact["model"].feature_importances_
            top_idx = np.argsort(importances)[::-1][:3]
            shap_data = [
                {"feature": feat_cols[i], "value": float(importances[i]), "data": float(X[0][i])}
                for i in top_idx
            ]
        except: pass

    # ── 5: Qdrant lookalikes ───────────────────────────────────────────
    lookalikes = []
    if qdrant is not None:
        try:
            # Using query_points for newer qdrant-client versions
            query_res = qdrant.query_points(
                collection_name=COLLECTION,
                query=X[0].tolist(),
                limit=3,
                with_payload=True,
            )
            for hit in query_res.points:
                payload = hit.payload
                lookalikes.append({
                    "similarity":  round(float(hit.score) * 100, 1),
                    "bundle":      payload.get("Purchased_Coverage_Bundle"),
                    "income":      payload.get("Estimated_Annual_Income"),
                    "employment":  payload.get("Employment_Status"),
                    "region":      payload.get("Region_Code"),
                    "dependents":  int(
                        (payload.get("Adult_Dependents") or 0)
                        + (payload.get("Child_Dependents") or 0)
                        + (payload.get("Infant_Dependents") or 0)
                    ),
                    "claims":      payload.get("Previous_Claims_Filed"),
                })
        except Exception as e:
            logger.warning(f"   Qdrant search failed: {e}")

    return {
        "type":          "ANALYSIS",
        "prediction":    pred,
        "confidence":    round(confidence * 100, 1),
        "shap":          shap_data,
        "lookalikes":    lookalikes,
        "extracted_form": raw,
        "predict_ms":    round(predict_ms, 1),
    }


def parse_features(text: str) -> Optional[dict]:
    """Extract JSON from <FEATURES>...</FEATURES> tags in Gemini text output."""
    match = re.search(r"<FEATURES>(.*?)</FEATURES>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        logger.warning("   Could not parse <FEATURES> JSON.")
        return None


# ── WebSocket: /ws/advisor ─────────────────────────────────────────────────────
@app.websocket("/ws/advisor")
async def ws_advisor(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ WebSocket /ws/advisor connected.")

    try:
        while True:
            # Reconnect loop for Gemini Session
            try:
                async with client.aio.live.connect(model=MODEL_ID, config=CONFIG) as session:
                    logger.info(f"✅ Gemini Live connected: {MODEL_ID}")
                    
                    # 🚀 AI Speaks First Initiative
                    logger.info("   Triggering AI greeting...")
                    try:
                        await session.send(input="Introduce yourself briefly as the InsureGuard AI Spokesman. Say you are calling to gather their details. Keep it very short and conversational.", end_of_turn=True)
                    except Exception as e:
                        logger.warning(f"   Could not send initial text prompt: {e}")

                    async def handle_analysis(transcript: str):
                        """Runs the Heavy REST generation asynchronously so it doesn't block audio"""
                        logger.info(f"   Received user transcript: {transcript[:50]}...")
                        try:
                            # Use AIO to prevent blocking the event loop! This is a massive source of lag if synchronous.
                            res = await client.aio.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=f"{EXTRACTION_PROMPT}\n\nTRANSCRIBED CLIENT DESCRIPTION:\n{transcript}"
                            )
                            extracted = parse_features(res.text)
                            if extracted:
                                result = run_pipeline(extracted)
                                await websocket.send_text(json.dumps(result))
                            else:
                                await websocket.send_text(json.dumps({"type": "TURN_COMPLETE"}))
                        except Exception as e:
                            logger.error(f"   Hybrid Analysis failed: {e}")

                    async def receive_from_client():
                        """PCM bytes or USER_TRANSCRIPT from React."""
                        try:
                            while True:
                                msg_raw = await websocket.receive()
                                if "bytes" in msg_raw:
                                    # Use exact format from live_server.py recipe
                                    await session.send_realtime_input(
                                        audio={"data": msg_raw["bytes"], "mime_type": "audio/pcm"}
                                    )
                                elif "text" in msg_raw:
                                    text = msg_raw["text"]
                                    if text == "STOP": return
                                    try:
                                        data = json.loads(text)
                                        if data.get("type") == "USER_TRANSCRIPT":
                                            transcript = data.get("content", "")
                                            # Fire and forget the analysis so audio keeps flowing
                                            asyncio.create_task(handle_analysis(transcript))
                                    except: pass
                        except WebSocketDisconnect:
                            logger.info("Frontend disconnected (receive loop).")
                            return
                        except RuntimeError as e:
                            if "Cannot call" in str(e):
                                logger.info("Receive called on closed socket (normal shutdown).")
                                return
                            else:
                                logger.error(f"RuntimeError in receive_from_client: {e}")
                                raise
                        except Exception as e:
                            logger.error(f"Error in receive_from_client: {e}")
                            raise

                    async def send_to_client():
                        """Gemini audio → React."""
                        try:
                            logger.info("Starting send_to_client loop...")
                            while True:
                                try:
                                    async for part in session.receive():
                                        if part.server_content is None: continue
                                        
                                        if part.server_content.turn_complete:
                                            logger.info("✅ Gemini Turn Complete. Sending signal to client.")
                                            await websocket.send_text(json.dumps({"type": "TURN_COMPLETE"}))

                                        model_turn = part.server_content.model_turn
                                        if model_turn:
                                            for p in model_turn.parts:
                                                if p.inline_data:
                                                    await websocket.send_bytes(p.inline_data.data)
                                    
                                    logger.info("session.receive() finished. Restarting loop...")
                                    await asyncio.sleep(0.1)
                                except Exception as e:
                                    logger.error(f"Error in inner send_to_client loop: {e}")
                                    break
                        except Exception as e:
                            logger.error(f"Error in send_to_client: {e}")
                            raise

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(receive_from_client())
                        tg.create_task(send_to_client())
                        
                    logger.info("Session context exited. Reconnecting...")

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected (Outer Loop).")
                break
            except asyncio.CancelledError:
                logger.info("Task Cancelled (Outer Loop).")
                break
            except Exception as e:
                if "WebSocketDisconnect" in str(e) or "Cannot call" in str(e):
                    logger.info("WebSocket disconnected (Exception Group).")
                    break
                logger.error(f"Session Error (Outer Loop): {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"WS fatal: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket /ws/advisor closed.")

# ── REST: POST /predict ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Estimated_Annual_Income:        Any = 50000.0
    Employment_Status:              Any = "Full-Time"
    Region_Code:                    Any = "Unknown"
    Adult_Dependents:               Any = 0
    Child_Dependents:               Any = 0
    Infant_Dependents:              Any = 0
    Previous_Claims_Filed:          Any = 0
    Years_Without_Claims:           Any = 1
    Deductible_Tier:                Any = "Medium"
    Acquisition_Channel:            Any = "Online"
    Payment_Schedule:               Any = "Monthly"
    Policy_Start_Month:             Any = "January"
    Existing_Policyholder:          Any = 0
    Previous_Policy_Duration_Months: Any = 12
    Vehicles_on_Policy:             Any = 1
    Custom_Riders_Requested:        Any = 0
    Grace_Period_Extensions:        Any = 0
    Policy_Cancelled_Post_Purchase: Any = 0


@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    result = run_pipeline(req.model_dump())
    return result


# ── REST: GET /health ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "online",
        "version":      "2.0",
        "branding":     "Voice-QL Advisor Edition",
        "author":       "Mohamed Attia — DataOverflow Team",
        "model":        MODEL_ID,
        "qdrant":       qdrant is not None,
        "shap":         explainer is not None,
        "features":     len(feat_cols),
    }


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advisor.backend.main:app", host="0.0.0.0", port=PORT, reload=True)
