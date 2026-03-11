# XGBoost: GPU vs CPU Training Benchmark

Benchmarking XGBoost classifier training speed and accuracy using GPU vs CPU on student performance data.

## Dataset
- **File**: `StudentPerformance.csv`
- **Objective**: Predicts whether a student passes (Performance Index >= 50)

## Notebooks

| File | Description |
| :--- | :--- |
| `gpu_training.ipynb` | XGBoost with `tree_method='gpu_hist'` |
| `cpu_training.ipynb` | XGBoost with `tree_method='hist'` (CPU only) |

## Results

| Mode | Training Time | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- |
| **GPU** | 0.23s ✅ | 96.75% | 97.23% |
| **CPU** | 0.39s | 96.85% | 97.31% |

**Analysis**:
- GPU is **~1.7× faster** than CPU
- Accuracy difference is negligible (< 0.1%) — both modes perform equally well
- GPU advantage scales up significantly on larger datasets

## Model Config

```python
import xgboost as xgb

# GPU
gpu_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=100, random_state=42)

# CPU
cpu_model = xgb.XGBClassifier(tree_method='hist', n_estimators=100, random_state=42)
```

## Setup & Execution

**Install Requirements**:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib
```

**Run Analysis**:
```bash
jupyter notebook
