# Technical Report - DataQuest Hackathon

This report fulfills the Phase II deliverables for the DataQuest Insurance Recommender System hackathon, detailing the system architecture, feature engineering methodology, and justification for the final modeling approach.

## 1. System Architecture

The following diagram illustrates the inference and deployment architecture constructed for Phase II.

```mermaid
flowchart TD
    subgraph Client Application
        A[User Browser]
    end

    subgraph FastAPI Service [API & Inference Server]
        B[FastAPI Router]
        C[Endpoint: /predict]
        D[Endpoint: /health]
        E[Endpoint: /]
    end

    subgraph Preprocessing Pipeline
        F[solution.preprocess]
        G[Null Value Interpolation]
        H[Downcasting & Type Mapping]
        I[Feature Extraction: Dependents/Risk]
    end

    subgraph Machine Learning Model
        J[load_model]
        K[(LightGBM: model.pkl)]
        L[solution.predict]
    end

    A -->|1. HTTP GET /| E
    E -->|HTML Template| A
    A -->|2. HTTP POST JSON Payload| C
    C -->|Raw DataFrame| F
    F --> G
    G --> H
    H --> I
    I -->|Processed DataFrame| L
    J -->|Loads| K
    K -->|Weights| L
    L -->|Integer Class (0-9)| C
    C -->|JSON Response| A
```

## 2. Feature Engineering & Preprocessing

The dataset (`train.csv` / `test.csv`) contained extremely sparse rows for features like `Employer_ID` and `Broker_ID`, alongside categorical elements such as `Region_Code` and `Deductible_Tier`.

Our fully encapsulated `solution.preprocess(df)` pipeline performs the following robust transformations:

1. **Memory Optimization**: The strict 1GB RAM constraint enforced aggressive memory profiling. We downcasted all `float64` columns to `float32` and safely compressed all integer features (excluding `User_ID`) to minimal integer depths (`int8`/`int16` where valid) using `pd.to_numeric(downcast='integer')`.
2. **Missing Value Imputation**: Null features were natively handled where possible. `-1` was used as a discrete filler for numerical sparsities (e.g. integer representations), and `'Unknown'` was applied to categorical spaces (`Deductible_Tier`, `Acquisition_Channel`, etc.).
3. **Categorical Enforcement**: Object string features were converted strictly to the Pandas `category` dtype. This allowed LightGBM to split on these distributions natively without the memory overhead of One-Hot Encoding schemas.
4. **Derived Risk Metrics**: We condensed household parameters into a single continuous representation (`Total_Dependents = Adult_Dependents + Child_Dependents + Infant_Dependents`). Furthermore, we calculated a baseline `Risk_Score_Proxy` by subtracting `Previous_Claims_Filed` from `Years_Without_Claims`. 

## 3. Justification of Model Choice

We selected a lightweight **LightGBM Classifier** due to its incredible synergy with the Hackathon's specific restrictions.

**Why LightGBM?**
- **Hardware Footprint (1GB RAM & 1 CPU core)**: LightGBM is inherently fast and scalable. By bounding the number of estimators (150) and restraining tree depth (`max_depth=6`), the final serialized `model.pkl` size is mere **4.96 MB**. This perfectly satisfied the `< 200MB` model penalty criteria, yielding a **0% penalty** on serialization size.
- **Inference Latency Limit (< 10 seconds)**: Deep neural networks or bulky ensembles would risk massive timing penalties. LightGBM consistently rendered batch inference over 1,000 rows in `< 0.1` seconds, far underneath the 10-second penalty curve, resulting in another **0% penalty** on latency scores.
- **Class Imbalance**: The 10-class integer problem was immensely skewed towards class `2` and `4`. Using `class_weight='balanced'` implicitly reweighted sparse observations (like classes 8 and 9) seamlessly using LightGBMs optimal histogram aggregations, natively optimizing the target `Macro F1` judging metric.

## 4. Model Performance & Benchmarks

To ensure the solution adheres to the rigorous Hackathon limits and provides competitive accuracy, the following empirical benchmarks were recorded during Final Optimized evaluation:

*   **Validation Metric (OOF Macro F1)**: `0.8767` 
    * *Achieved via Optuna hyperparameter optimization (50 trials) and Deep Feature Engineering.*
*   **Consolidated Model Size**: `6.11 MB` 
    * *This single `.pkl` file encapsulates both the LightGBM Classifier and the Target Encoder state.*
*   **Inference Latency (1,000 requests)**: `0.099 seconds`
    * *Maintained high-speed rendering underneath the 10.0s penalty ceiling despite increased feature complexity.*
    
These benchmarks demonstrate an extremely lean, yet accurate classification engine perfectly tuned for the mandated 1 CPU / 1GB RAM constraint environment. 

## 5. Feature Engineering Summary (Optimized)
- **Deep Interactions**: Ratio-based features for `Income_per_Dependent` and household `Risk_Ratio`.
- **Target Encoding**: High-cardinality features (`Region_Code`, `Broker_ID`, `Employer_ID`) were mapped using 3-Fold Stratified K-Fold Target Encoding to prevent leakage while capturing local variance.
- **Combined Pipeline**: The entire preprocessing, encoding, and inference logic is encapsulated within `solution.py`, requiring zero external downloads at runtime.
