---
phase: 6
plan: 1
wave: 1
---

# Plan 6.1: Minimal Inference API and UI

## Objective
Build a lightweight API with at least two endpoints and a simple UI for testing model inferences to fulfill Phase II Productization & Deployment deliverables. 

## Context
- .gsd/SPEC.md
- .gsd/ARCHITECTURE.md
- solution.py
- read_pdf.py (Hackathon Brief Sections 8.2 & 9)
- train.csv / test.csv (for input schema reference)

## Tasks

<task type="auto">
  <name>Build FastAPI Application</name>
  <files>api.py, requirements.txt</files>
  <action>
    - Create a FastAPI application running on Uvicorn.
    - Implement a `/health` endpoint to return server status.
    - Implement a `/predict` endpoint that takes JSON mimicking the test.csv schema, uses `load_model()` and `preprocess()` from `solution.py`, and returns the exact integer class prediction (0-9).
    - Add `fastapi` and `uvicorn` to `requirements.txt`.
  </action>
  <verify>python -m uvicorn api:app --help</verify>
  <done>API is built with at least 2 endpoints and can mock-predict a JSON row.</done>
</task>

<task type="auto">
  <name>Create Simple UI</name>
  <files>templates/index.html, api.py</files>
  <action>
    - Ensure `api.py` serves an HTML template at `/`.
    - Build `index.html` with a basic, cleanly-styled form where a user can enter at least 5-10 core features (e.g., Dependents, Income, Tiers) and submit for a prediction.
    - Render the prediction result immediately below the form.
  </action>
  <verify>python -c "import os; print(os.path.exists('templates/index.html'))"</verify>
  <done>UI HTML file exists and API is configured to serve it.</done>
</task>

## Success Criteria
- [ ] API has two working endpoints (`/health` and `/predict`)
- [ ] UI is accessible and serves a form
- [ ] Predictions return an integer from 0-9 using the trained `model.pkl`
