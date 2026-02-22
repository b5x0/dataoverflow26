# Technical Specification: Voice-QL Advisor (v2.2)

This document details the architectural and implementation specifics of the Voice-QL project.

---

## 1. Phase 1: Predictive Engine (`solution.py`)

The core of the project is a high-performance insurance bundle classifier.

### 1.1 Preprocessing Pipeline
- **Categorical Encoding**: Converts string-based features (Region, Employment, etc.) into stable integer codes matching the training-time dictionary. This eliminates the overhead of Pandas' `category` dtype during live inference.
- **Feature Engineering**:
    - `Total_Dependents`: Aggregate of Adult, Child, and Infant dependents.
    - `Income_per_Dependent`: Normalized financial capability score.
    - `Risk_Ratio`: Claims frequency over policy duration.
- **Performance**: Pre-processes a single sample in <1ms by using vectorised NumPy operations.

### 1.2 Model Architecture
- **Type**: LightGBM Multi-class Classifier.
- **Classes**: 10 distinct insurance bundles (0–9).
- **Optimization**: Optimized for Macro F1-Score while maintaining a sub-200MB memory footprint.
- **Inference**: Uses the native LightGBM Booster `predict()` path directly to bypass Scikit-Learn wrapper overhead.

---

## 2. Phase 2: Creative AI & HUD (`advisor/`)

The creative layer integrates real-time voice intelligence and vector searching.

### 2.1 Gemini Live AI Integration
- **Persona**: Proactive "AI Spokesman" implemented via `MASTER_PROMPT`.
- **Modality**: Native `AUDIO` duplex streaming using the Gemini 2.5 Live API.
- **Hybrid Speech Flow**:
    - **Transcription**: Uses the Browser **Web Speech API** for local transcription of the user. This ensures higher reliability than extracting text from the Live API stream.
    - **Extraction**: User transcripts are sent via WebSocket to the backend, where the standard **Gemini REST API** extracts 18+ insurance features in real-time.

### 2.2 Explainability & Analytics
- **Live SHAP**: Uses LightGBM's native `pred_contrib` path to calculate local feature impact drivers in just a few milliseconds.
- **Fallback**: Automatically falls back to global feature importances if local SHAP calculation fails, ensuring HUD stability.

### 2.3 Vector Search (Qdrant)
- **Engine**: Local Qdrant instance storing **60,868** client vectors.
- **Similarity**: Uses the engineered feature array as a query vector to find the top 3 "lookalike" profiles based on cosine similarity.
- **Payload**: Returns rich demographic data to give the agent instant historical context.

### 2.4 Frontend HUD
- **Stack**: React (hook-based state), TypeScript (type-safe messaging), Vite (build system).
- **Audio Control**: `AudioContext` at 16kHz for recording and 24kHz for playback, with a minimized recording buffer (1024 samples) to optimize latency.
- **Responsive Branding**: Premium "Strategist" aesthetic with animated `VoicePulse` and scrolling `LiveTranscription` feed.
