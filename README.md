# Voice-QL Advisor: AI Spokesman Edition (v2.2)

An intelligent, voice-activated insurance advisor HUD. The AI Agent acts as a "Spokesman," calling potential clients to gather demographic and risk data, providing real-time bundle predictions and lookalike searches.

## 🚀 Quick Start (Local)

1. **Environment**:
   - Create a `.env` file in the root:
     ```env
     GEMINI_API_KEY=your_key_here
     ```
2. **Backend**:
   - `pip install -r requirements.txt`
   - `python -m uvicorn advisor.backend.main:app --port 8002 --reload`
3. **Frontend**:
   - `cd advisor/frontend`
   - `npm install`
   - `npm run dev`

## ☁️ Deployment (Render)

### Backend (Web Service)
1. **Repository**: Connect your GitHub repo.
2. **Runtime**: Python.
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `python -m uvicorn advisor.backend.main:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables**:
   - Add `GEMINI_API_KEY`.

### Frontend (Static Site)
1. **Build Command**: `npm run build`
2. **Publish Directory**: `advisor/frontend/dist`

## 🛠️ Architecture
- **Voice**: Browser Web Speech API (Transcription) + Gemini Live (Native Audio Interaction).
- **Analytics**: LightGBM (Bundle Prediction) + SHAP (Impact Drivers) + Qdrant (Lookalike Search).
- **Frontend**: React + TypeScript + Vitest + Lucide Icons.
- **Backend*: FastAPI + Pydantic + Uvicorn.
