# ROADMAP.md

> **Current Phase**: Not started
> **Milestone**: v1.0

## Must-Haves (from SPEC)
- [ ] Phase A: Data Profiling & EDA (`train.csv`, `test.csv`)
- [ ] Phase B: Feature Engineering & Preprocessing encapsulatable in `preprocess(df)`
- [ ] Phase C: Model Training & Validation (CatBoost/LightGBM, <200MB, <10s)
- [ ] Phase D: Submission Package (`solution.py`, `model.<ext>`, `requirements.txt` into `submission.zip`)

## Phases

### Phase 1: Project Setup & Research
**Status**: ✅ Done
**Objective**: Parse PDF constraints and prepare the development environment

### Phase 2: Data Profiling & Exploratory Data Analysis
**Status**: ✅ Done
**Objective**: Understand the distribution of datasets and characteristics

### Phase 3: Preprocessing and Feature Engineering
**Status**: ✅ Done
**Objective**: Create robust feature engineering pipelines encapsulated into functions

### Phase 4: Model Training and Validation
**Status**: ✅ Done
**Objective**: Train model with LightGBM/CatBoost, validate metrics

### Phase 5: Submission Packaging
**Status**: ✅ Done
**Objective**: Zip the necessary artifacts properly

### Phase 6: API Development & Deployment (Phase II Hackathon)
**Status**: ✅ Done
**Objective**: Build a minimal API (at least 2 endpoints) and a simple UI for inference

### Phase 7: Technical Report & MLOps
**Status**: ✅ Done
**Objective**: Write the technical report with an architecture diagram, feature explanations, and MLOps strategies

### Phase 8: Final Pitch & Repo Formatting
**Status**: ✅ Done
**Objective**: Prepare Git repository, documentation, and a final presentation/demo

### Phase 9: Optimization Sprint
**Status**: ⬜ Not Started
**Objective**: Maximize Macro F1 via deep feature engineering (interaction terms, Target Encoding) and Optuna hyperparameter tuning
