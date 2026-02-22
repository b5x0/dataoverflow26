# 🛡️ InsureGuard: High-Performance Bundle Prediction & Advisor

InsureGuard is an end-to-end insurance decision support system designed for high-concurrency, low-latency environments. It combines a surgically optimized LightGBM prediction engine with advanced Explainable AI (SHAP) and a Vectorized Lookalike Engine (Qdrant) to empower insurance agents with data-driven insights.

## 🚀 The "Surgical Strike" Methodology
Unlike standard "black-box" models, InsureGuard was built under strict 1-core CPU and 1GB RAM constraints. We prioritized **Efficiency-per-Split** to achieve a competitive **0.5413 Macro F1** while maintaining sub-2-second inference latency.

### Key Optimization Pillars:
* **Feature Pruning:** Systematically removed high-cardinality noise (e.g., `Region_Code`, `Acquisition_Channel`) to reduce leaf-traversal overhead.
* **Engineered Signal:** Injected high-intensity interactions like `Risk_Ratio`, `Loyalty_Index`, and `Premium_Capacity` to maximize accuracy with fewer trees.
* **Hardware-Aware Inference:** Hard-coded `n_jobs=1` and utilized `.values` NumPy conversion to bypass Pandas indexing bottlenecks in containerized environments.

## 🛠️ Tech Stack & Architecture


* **Prediction Engine:** LightGBM (Optimized for 1-core/150-tree inference).
* **Vector Search:** Qdrant (Customer Lookalike Engine for social proofing).
* **Explainability:** SHAP (Local feature contribution analysis).
* **Backend:** FastAPI (Asynchronous inference & reasoning).
* **Frontend:** React (Interactive Agent Dashboard).

## 💡 Phase 2 Features: The "Transparent Advisor"
1. **Automated Onboarding:** Using Gemini OCR to extract vehicle and user data directly from registration documents, reducing data-entry friction.
2. **Social Proofing:** Qdrant lookalike search identifies the 5 most similar historical clients to provide evidence-based recommendations.
3. **The "Why" Layer:** Real-time SHAP visualizations explain exactly which factors (e.g., household burden, claim history) influenced the bundle suggestion.
4. **Agent Mentor:** Gemini-powered natural language reasoning transforms raw scores into personalized sales scripts.

## 📦 Installation & Setup
```bash
# Clone the repository
git clone [https://github.com/b5x0/dataoverflow26.git](https://github.com/b5x0/dataoverflow26.git)

# Install dependencies
pip install -r requirements.txt

# Run the local Qdrant instance
docker run -p 6333:6333 qdrant/qdrant

```

## 📈 Performance Metrics

| Metric | Result |
| --- | --- |
| **Macro F1 Score** | 0.5513 |
| **Inference Latency** | 0.96s (1-core) |
| **Model Size** | 3.25 MB |
| **RAM Usage** | < 1GB |

---

*Developed for the 2026 Insurance Innovation Challenge.*
