# Smart Cow Analysis System

This module implements a unified system for farm analytics across four features:

- Milk Litre Analysis with trend prediction (XGBoost/LSTM with fallback)
- Cow Health Detection from audio using CNN on 128-band Mel-spectrograms
- Breed Identification from image using transfer learning (EfficientNetB0/MobileNetV3)
- Milk Profit Calculation with dynamic pricing and cost breakdown

File: `smart_cow_analysis.py`

## Installation

```bash
pip install -r requirements.txt
```

Some dependencies are optional at runtime. Each feature will report a clear error if a required package/model is missing.

## Quick Start

```python
from smart_cow_analysis import analyze_all

result = analyze_all(
    milk_input={
        "cow_id": "COW123",
        "milking_events": [6.2, 5.8],
        "timestamp": ["2025-12-11T06:30:00", "2025-12-11T18:45:00"],
        "history_daily_totals": [11.4, 11.2, 11.6, 11.0, 11.9, 12.1, 11.8],
        # model_type="auto" | "xgboost" | "lstm"
    },
    audio_input=None,  # requires a trained CNN model path
    image_input=None,  # requires a trained classifier model path
    profit_input={
        "total_litres_today": 12.0,
        "price_per_litre": 38.0,
        "feed_kg_today": 10.0,
        "feed_price_per_kg": 20.0,
        "labor_minutes": 90,
        "labor_rate": 120.0,
        "vet_cost": 50.0,
        "other_costs": 30.0,
    }
)
print(result)
```

## API Reference

### 1) milk_litre_analysis()
Inputs:
- `cow_id: str`
- `milking_events: List[float]`
- `timestamp: List[str]` (ISO)
- Optional: `lactation_days`, `feed_intake`, `cow_weight`
- Optional: `history_daily_totals: List[float]` (oldest -> newest)
- Optional: `model_type: str` ("auto" | "xgboost" | "lstm")

Outputs (dict):
- `total_litres_today`, `litres_morning`, `litres_evening`
- `trend` (increasing/decreasing)
- `expected_next_day_litres`
- `model_used`

### 2) cow_health_detection()
Inputs:
- `audio_path: str`
- `model_path: str` (tf.keras model with 5-class output)
- Optional: `class_names` (default order: Healthy, Coughing, Distress, Mastitis sound, Abnormal)
- Optional: `sr`, `n_mels=128`, `hop_length`, `duration_sec`

Outputs (dict):
- `health_status`, `disease_sign` (if any), `confidence_score` (0-100)
- `top5` probabilities

Note: You must provide a trained model file path.

### 3) breed_identification()
Inputs:
- `image_path: str`
- `model_path: str` (tf.keras model)
- Optional: `breeds` (default: Jersey, Holstein Friesian, Gir, Sahiwal, Red Sindhi, Kangeyam, Crossbreed)
- Optional: `use_efficientnet: bool` or set to False for MobileNetV3 preprocessing
- Optional: `use_yolo_detect: bool`, `yolo_model_path` to crop the cow ROI before classification
- Optional: `input_min_size=(720,720)`, `top_k=3`

Outputs (dict):
- `breed_name`, `confidence` (%), `top_3_predictions`

### 4) milk_profit_calculation()
Inputs:
- `total_litres_today`, `price_per_litre`
- `feed_kg_today`, `feed_price_per_kg`
- `labor_minutes`, `labor_rate`
- `vet_cost`, `other_costs`

Formula:
- Revenue = total_litres_today × price_per_litre
- Costs = feed_cost + labor_cost + healthcare_cost + misc_cost
- Profit = Revenue − Costs

Outputs (dict):
- `revenue_today`, `total_cost_today`, `profit_today`, `profit_per_litre`, `weekly_profit_estimate`
- `breakdown` of costs

## Models
- Health detection expects a 5-class CNN model trained on Mel-spectrogram inputs shaped like `(H, W, 1)`.
- Breed identification expects a classifier compatible with EfficientNetB0 or MobileNetV3 preprocessing.
- Provide model paths via `model_path`.

## Notes
- Functions validate inputs and return structured error messages when needed.
- All outputs are JSON-compatible dictionaries for easy integration.
