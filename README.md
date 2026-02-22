# Voice-QL Advisor: AI Spokesman Edition (v2.2)

An intelligent, voice-activated insurance advisor HUD and "Spokesman" simulator, developed as part of the **Data Overflow 2026 Hackathon**.

## 📖 Project Context: Data Overflow 2026

This project was built in two distinct phases to solve a complex insurance sales optimization challenge:

### Phase 1: Model Engineering & Training
In the first phase, we focused on deep data science:
- **Dataset**: Analyzed a large insurance client dataset with diverse demographic and risk features.
- **Model Training**: Developed and fine-tuned a **LightGBM** multiclass classifier to predict optimal coverage bundles (0–9).
- **Optimization**: Achieved a critical balance between high accuracy (Macro F1 benchmark) and ultra-fast inference latency (sub-10ms) using a native NumPy prediction path.

### Phase 2: Creative HUD & Live AI Deployment
The second phase transformed a dry predictive model into a "WOW" user experience:
- **AI Spokesman**: Integrated **Gemini 2.5 Live (Native Audio)** to create a proactive salesperson persona. The AI "calls" clients, gathers data via voice, and delivers tactical pitches.
- **Creative UI**: Designed a premium, dark-themed "Strategist HUD" using React and TypeScript, providing real-time visual feedback on audio levels, transcripts, and model insights.
- **Qdrant Integration**: Implemented a vector search layer using **Qdrant** to instantly find "Lookalike" client profiles from a database of over **60,000** records, providing agents with historical context.
- **Hybrid Voice Flow**: Combined the **Browser Web Speech API** (for high-fidelity transcription) with **Gemini Live** (for low-latency voice interaction) to bypass hardware and API limitations.

---

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

For a step-by-step guide to cloud deployment, see the [Render Deployment Guide](.system_generated/Render-Deployment-Guide.md).

## 🛠️ Technical Stack
- **AI**: Gemini 2.5 Flash (Live & REST).
- **Database**: Qdrant (Vector Engine).
- **ML**: LightGBM, SHAP, NumPy, Pandas.
- **Backend**: FastAPI, Pydantic, Uvicorn.
- **Frontend**: React, TypeScript, Vite, Lucide Icons.
