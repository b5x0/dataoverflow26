import os
import asyncio
import json
import logging
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS - Allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY not found in environment variables!")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025" 

# --- KNOWLEDGE INJECTION ---
def load_knowledge():
    """Loads and merges knowledge from JSON files."""
    knowledge_text = "\n\n### KNOWLEDGE BASE ###\n"
    
    # 1. General Knowledge Base (Local)
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
                knowledge_text += f"\n[GENERAL Q&A]:\n{json.dumps(kb_data, indent=2, ensure_ascii=False)}\n"
        except Exception as e:
            logger.error(f"Error reading knowledge_base.json: {e}")
    else:
        logger.warning(f"knowledge_base.json not found at {kb_path}")

    # 2. Museum Archives (Relative Path)
    # Path: ../kiosk-ui/src/data/museum_archives.json
    museum_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../kiosk-ui/src/data/museum_archives.json"))
    if os.path.exists(museum_path):
        try:
            with open(museum_path, "r", encoding="utf-8") as f:
                museum_data = json.load(f)
                knowledge_text += f"\n[MUSEUM ARCHIVES]:\n{json.dumps(museum_data, indent=2, ensure_ascii=False)}\n"
        except Exception as e:
            logger.error(f"Error reading museum_archives.json: {e}")
    else:
        logger.warning(f"museum_archives.json not found at {museum_path}")
        
    return knowledge_text

# Define Persona
MASTER_PROMPT = (
    "You are the SHIELDS UP AI Core, a tactical cybersecurity intelligence defending GDGC ISSATSo data.\n"
    "The Shields Up event (Feb 14-15) promotes cyber awareness. Treat participants as modern digital protectors.\n"
    "CRITICAL: Keep responses under 2 concise sentences. Be punchy, fast, and tactical.\n"
    "Language Rule: Auto-detect language. Reply in Tunisian Arabic (Derja) if spoken to in Tounsi, French for French, English for English."
)

# Append Knowledge
MASTER_PROMPT += load_knowledge()

# Configure session
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": MASTER_PROMPT
}

@app.websocket("/ws/gemini")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ WebSocket connected.")

    try:
        while True:
            # Reconnect loop for Gemini Session
            try:
                async with client.aio.live.connect(model=MODEL_ID, config=CONFIG) as session:
                    logger.info(f"✅ Connected to Gemini Live API: {MODEL_ID}")

                    async def receive_from_client():
                        """Receives audio from React frontend and sends to Gemini."""
                        try:
                            while True:
                                message = await websocket.receive()
                                
                                if "bytes" in message:
                                    data = message["bytes"]
                                    # Wrap raw PCM bytes for Gemini
                                    # Use exact format from local_test.py to ensure session persistence
                                    await session.send_realtime_input(audio={"data": data, "mime_type": "audio/pcm"})
                                    
                                elif "text" in message:
                                    # Ignore text messages or handle graceful stop if needed.
                                    pass

                        except WebSocketDisconnect:
                            logger.info("Frontend disconnected (receive loop).")
                            return # Exit correctly, TaskGroup will cancel other tasks
                        except RuntimeError as e:
                            if "Cannot call" in str(e):
                                logger.info("Receive called on closed socket (normal shutdown).")
                                return # Exit correctly
                            else:
                                logger.error(f"RuntimeError in receive_from_client: {e}")
                                raise
                        except Exception as e:
                            logger.error(f"Error in receive_from_client: {e}")
                            # Don't raise, just let task group finish if possible or break?
                            # If receive fails, we probably should reconnect session or close.
                            raise

                    async def send_to_client():
                        """Receives audio from Gemini and sends to React frontend."""
                        try:
                            logger.info("Starting send_to_client loop...")
                            while True:
                                try:
                                    async for part in session.receive():
                                        if part.server_content is None:
                                            continue
                                        
                                        if part.server_content.turn_complete:
                                            logger.info("✅ Gemini Turn Complete. Sending signal to client.")
                                            await websocket.send_text("TURN_COMPLETE")

                                        model_turn = part.server_content.model_turn
                                        if model_turn:
                                            for p in model_turn.parts:
                                                if p.inline_data:
                                                    # Send raw PCM bytes back to frontend
                                                    await websocket.send_bytes(p.inline_data.data)
                                    
                                    logger.info("session.receive() finished. Restarting loop...")
                                    await asyncio.sleep(0.1)

                                except Exception as e:
                                    logger.error(f"Error in inner send_to_client loop: {e}")
                                    break # If error, break inner loop to trigger outer reconnect

                        except Exception as e:
                            logger.error(f"Error in send_to_client: {e}")
                            raise

                    # Run both loops concurrently
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
                # Check for ExceptionGroup wrapping disconnects
                if "WebSocketDisconnect" in str(e) or "Cannot call" in str(e):
                    logger.info("WebSocket disconnected (Exception Group).")
                    break
                    
                logger.error(f"Session Error (Outer Loop): {e}")
                traceback.print_exc()
                # Basic backoff or check if we should break
                await asyncio.sleep(1)

    except Exception as e:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket closed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
