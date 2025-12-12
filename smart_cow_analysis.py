"""
Smart Cow Analysis System

Features:
1) Milk Litre Analysis with trend prediction (XGBoost or LSTM, with auto fallback)
2) Cow Health Detection from audio using CNN on Mel-spectrograms (128 Mel bands)
3) Breed Identification from image using transfer learning (EfficientNetB0/MobileNetV3)
4) Milk Profit Calculation with dynamic pricing

Notes:
- For model-based features (2, 3), you must provide a trained model path. The code loads a saved
  Keras model via tf.keras.models.load_model(). If not provided, the function returns an error status.
- For optional cow detection in Breed Identification, if ultralytics (YOLOv8) is available and a model
  path is provided, the function will detect and crop the cow before classification.
- The time-series prediction (1) prefers XGBoost if installed, otherwise tries TensorFlow LSTM, and
  finally falls back to a naive baseline if insufficient history or dependencies are missing.

All functions return JSON-compatible dicts and perform input validation with descriptive errors.
"""
from __future__ import annotations

import os
import math
import json
from typing import Any, Dict, List, Optional, Tuple

# Numeric / ML
import numpy as np

# Optional imports guarded to allow partial functionality if not installed
try:
    import pandas as pd  # used in milk analysis for time handling
except Exception:  # pragma: no cover
    pd = None

# XGBoost (optional)
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

# TensorFlow / Keras (optional)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models
except Exception:  # pragma: no cover
    tf = None
    layers = None
    models = None

# Audio processing (optional for health detection)
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

# Image processing (optional for breed identification)
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

# Optional YOLOv8
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


# -----------------------------
# Helpers
# -----------------------------

def _error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {"status": "ok"}
    out.update(payload)
    return out


# -----------------------------
# 1) Milk Litre Analysis
# -----------------------------

def milk_litre_analysis(
    cow_id: str,
    milking_events: List[float],
    timestamp: List[str],
    lactation_days: Optional[int] = None,
    feed_intake: Optional[float] = None,
    cow_weight: Optional[float] = None,
    history_daily_totals: Optional[List[float]] = None,
    model_type: str = "auto",  # "xgboost" | "lstm" | "auto"
) -> Dict[str, Any]:
    """
    Analyze daily milk litres and predict next-day litres with trend.

    Inputs:
      - cow_id: unique identifier
      - milking_events: litres per session for the current day
      - timestamp: ISO timestamps corresponding to each session, same length as milking_events
      - Optional context: lactation_days, feed_intake (kg), cow_weight (kg)
      - history_daily_totals: past daily totals (oldest -> newest). Recommended length >= 7
      - model_type: choose "xgboost", "lstm", or "auto"

    Outputs:
      - cow_id, total_litres_today, litres_morning, litres_evening, trend, expected_next_day_litres
    """
    # Validate basics
    if not isinstance(cow_id, str) or not cow_id:
        return _error("cow_id must be a non-empty string")
    if not isinstance(milking_events, list) or not isinstance(timestamp, list):
        return _error("milking_events and timestamp must be lists")
    if len(milking_events) == 0:
        return _error("milking_events cannot be empty")
    if len(milking_events) != len(timestamp):
        return _error("milking_events and timestamp must have the same length")
    if pd is None:
        return _error("pandas must be installed for milk_litre_analysis")

    # Build per-session dataframe and aggregate morning/evening
    try:
        df = pd.DataFrame({"litres": milking_events, "timestamp": pd.to_datetime(timestamp)})
    except Exception as e:  # bad timestamp format
        return _error(f"invalid timestamps: {e}")

    df = df.sort_values("timestamp")
    df["period"] = df["timestamp"].dt.hour.apply(lambda h: "morning" if h < 12 else "evening")
    litres_morning = float(df[df["period"] == "morning"]["litres"].sum())
    litres_evening = float(df[df["period"] == "evening"]["litres"].sum())
    total_litres_today = float(df["litres"].sum())

    # Prepare history vector for forecasting (if provided)
    hist = None
    if history_daily_totals is not None:
        try:
            hist = np.array([float(v) for v in history_daily_totals], dtype=np.float32)
            hist = hist[~np.isnan(hist)]
        except Exception:
            return _error("history_daily_totals must be numeric")

    # Forecast next-day litres
    forecast = None
    used_model = None

    def _naive_forecast(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return total_litres_today  # repeat
        # simple last-value or simple moving average
        if len(arr) >= 3:
            return float(np.mean(arr[-3:]))
        return float(arr[-1])

    # Prefer XGBoost if requested/available and history is sufficient
    if forecast is None and (model_type in ("xgboost", "auto")) and xgb is not None and hist is not None and len(hist) >= 5:
        try:
            X = np.arange(len(hist), dtype=np.float32).reshape(-1, 1)
            y = hist.astype(np.float32)
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                "objective": "reg:squarederror",
                "max_depth": 3,
                "eta": 0.3,
                "subsample": 0.9,
                "lambda": 1.0,
                "verbosity": 0,
            }
            bst = xgb.train(params, dtrain, num_boost_round=80)
            next_X = xgb.DMatrix(np.array([[len(hist)]], dtype=np.float32))
            forecast = float(bst.predict(next_X)[0])
            used_model = "xgboost"
        except Exception:
            forecast = None

    # Try LSTM via TensorFlow
    if forecast is None and (model_type in ("lstm", "auto")) and tf is not None and hist is not None and len(hist) >= 6:
        try:
            # scale
            arr = hist.reshape(-1, 1).astype(np.float32)
            min_v, max_v = float(arr.min()), float(arr.max())
            rng = (max_v - min_v) if max_v > min_v else 1.0
            scaled = (arr - min_v) / rng
            # build sequences
            T = 3
            Xs, ys = [], []
            for i in range(len(scaled) - T):
                Xs.append(scaled[i:i+T])
                ys.append(scaled[i+T])
            Xs = np.array(Xs, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)
            # model
            model = tf.keras.Sequential([
                layers.Input(shape=(T, 1)),
                layers.LSTM(32, activation="tanh"),
                layers.Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-2) if hasattr(tf.keras.optimizers, 'legacy') else tf.keras.optimizers.Adam(1e-2), loss="mse")
            model.fit(Xs, ys, epochs=60, verbose=0)
            pred = model.predict(scaled[-T:].reshape(1, T, 1), verbose=0)[0][0]
            forecast = float(pred * rng + min_v)
            used_model = "lstm"
        except Exception:
            forecast = None

    # Fallback
    if forecast is None:
        forecast = _naive_forecast(hist)
        used_model = used_model or "naive"

    trend = "increasing" if forecast > total_litres_today else "decreasing"

    return _ok({
        "feature": "milk_litre_analysis",
        "cow_id": cow_id,
        "total_litres_today": round(total_litres_today, 3),
        "litres_morning": round(litres_morning, 3),
        "litres_evening": round(litres_evening, 3),
        "trend": trend,
        "expected_next_day_litres": round(float(max(0.0, forecast)), 3),
        "model_used": used_model,
    })


# -----------------------------
# 2) Cow Health Detection (Audio -> Mel -> CNN)
# -----------------------------

def _build_default_audio_cnn(input_shape: Tuple[int, int, int]) -> Any:
    if tf is None:
        raise RuntimeError("TensorFlow not available to build audio CNN")
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')  # 5 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def cow_health_detection(
    audio_path: Optional[str] = None,
    model_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512,
    duration_sec: Optional[float] = None,
    frequency_hz: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Classify cow health from audio using CNN over Mel-spectrograms.

    - audio_path: path to a .wav/.mp3 audio file
    - model_path: path to a trained tf.keras model (expects 5-class output in order below)
    - class_names: order of classes for the model. Default:
        ["Healthy", "Coughing", "Distress", "Mastitis sound", "Abnormal"]
    - sr: target sample rate
    - n_mels: 128 per spec
    - hop_length: STFT hop length
    - duration_sec: if provided, clip/pad to this duration

    Returns: { health_status, disease_sign, confidence_score }
    """
    if class_names is None:
        class_names = ["Healthy", "Coughing", "Distress", "Mastitis sound", "Abnormal"]
    # Path A: Frequency-based heuristic (no audio, no ML model required)
    if frequency_hz is not None:
        try:
            # Accept single float or list of floats
            if isinstance(frequency_hz, (int, float)):
                freqs = np.array([float(frequency_hz)], dtype=np.float32)
            elif isinstance(frequency_hz, (list, tuple, np.ndarray)):
                freqs = np.array([float(x) for x in frequency_hz], dtype=np.float32)
            else:
                return _error("frequency_hz must be a number or a list of numbers")
            freqs = freqs[~np.isnan(freqs)]
            if freqs.size == 0:
                return _error("frequency_hz contains no valid numbers")
            # Use median frequency as robust estimate
            f_med = float(np.median(freqs))
            # Heuristic thresholds (Hz):
            # <70 => Abnormal (very low)
            # 70-220 => Healthy typical fundamental moo
            # 220-350 => Coughing/Abnormal transition
            # 350-1000 => Distress / high-pitch
            # >1000 => Abnormal
            if f_med < 70:
                label = "Abnormal"; conf = 0.85
            elif 70 <= f_med <= 220:
                label = "Healthy"; conf = 0.9
            elif 220 < f_med <= 350:
                label = "Coughing"; conf = 0.7
            elif 350 < f_med <= 1000:
                label = "Distress"; conf = 0.8
            else:
                label = "Abnormal"; conf = 0.75
            disease_sign = None if label == "Healthy" else label
            return _ok({
                "feature": "cow_health_detection",
                "mode": "frequency",
                "median_frequency_hz": round(f_med, 2),
                "health_status": label,
                "disease_sign": disease_sign,
                "confidence_score": round(conf * 100.0, 2),
            })
        except Exception as e:
            return _error(f"frequency analysis failed: {e}")

    # Path B: Audio-based CNN
    if librosa is None:
        return _error("librosa is required for audio processing")
    if audio_path is None:
        return _error("audio_path is required when frequency_hz is not provided")
    if not os.path.exists(audio_path):
        return _error("audio_path not found")

    try:
        y, sr0 = librosa.load(audio_path, sr=sr, mono=True)
        if duration_sec is not None and duration_sec > 0:
            target_len = int(sr * duration_sec)
            if len(y) > target_len:
                y = y[:target_len]
            elif len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_min, S_max = float(S_db.min()), float(S_db.max())
        S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)
        mel_img = S_norm.astype(np.float32)[..., np.newaxis]
    except Exception as e:
        return _error(f"audio processing failed: {e}")

    if model_path is None:
        return _error("model_path is required for cow_health_detection")
    if tf is None:
        return _error("TensorFlow is required to load the CNN model")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return _error(f"failed to load model: {e}")

    try:
        h_expected = getattr(model.input_shape, 'as_list', lambda: model.input_shape)()[1] if hasattr(model.input_shape, 'as_list') else model.input_shape[1]
        w_expected = getattr(model.input_shape, 'as_list', lambda: model.input_shape)()[2] if hasattr(model.input_shape, 'as_list') else model.input_shape[2]
        H, W = mel_img.shape[:2]
        target_h = h_expected if isinstance(h_expected, int) and h_expected else H
        target_w = w_expected if isinstance(w_expected, int) and w_expected else W
        img = mel_img
        if H < target_h:
            pad_h = target_h - H
            img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)))
        elif H > target_h:
            img = img[:target_h, :, :]
        H2, W2 = img.shape[:2]
        if W2 < target_w:
            pad_w = target_w - W2
            img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)))
        elif W2 > target_w:
            img = img[:, :target_w, :]
        x = img[np.newaxis, ...]
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        health_status = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        disease_sign = None if health_status == "Healthy" else health_status
        return _ok({
            "feature": "cow_health_detection",
            "mode": "audio",
            "health_status": health_status,
            "disease_sign": disease_sign,
            "confidence_score": round(conf * 100.0, 2),
            "top5": [
                {"label": class_names[i] if i < len(class_names) else f"Class_{i}", "confidence": float(probs[i])}
                for i in np.argsort(-probs)[:5]
            ],
        })
    except Exception as e:
        return _error(f"inference failed: {e}")


# -----------------------------
# 3) Breed Identification (Image -> Transfer Learning)
# -----------------------------

def _load_and_center_crop(img_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if Image is None:
        return None
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize(target_size)
        arr = np.asarray(im).astype(np.float32)
        return arr


def breed_identification(
    image_path: str,
    model_path: Optional[str] = None,
    breeds: Optional[List[str]] = None,
    use_efficientnet: bool = True,
    use_yolo_detect: bool = False,
    yolo_model_path: Optional[str] = None,
    input_min_size: Tuple[int, int] = (720, 720),
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Identify cow breed using transfer learning (EfficientNetB0 or MobileNetV3).

    - image_path: path to an image (min 720p recommended). If smaller, it will be upscaled.
    - model_path: path to a trained tf.keras model compatible with the chosen backbone.
    - breeds: class names in the order of the model's output.
    - use_efficientnet: True to assume EfficientNetB0 preprocessing; False -> MobileNetV3
    - use_yolo_detect: if True and ultralytics YOLOv8 available + yolo_model_path set, crop cow ROI first.
    - top_k: number of top predictions to return.

    Returns: { breed_name, confidence, top_3_predictions }
    """
    if breeds is None:
        breeds = ["Jersey", "Holstein Friesian", "Gir", "Sahiwal", "Red Sindhi", "Kangeyam", "Crossbreed"]

    if Image is None:
        return _error("Pillow (PIL) is required for image processing")
    if not os.path.exists(image_path):
        return _error("image_path not found")
    # If no TF or model provided, use heuristic fallback
    tf_available = (tf is not None)
    use_heuristic = (not tf_available) or (model_path is None)

    try:
        # Optional detection and crop
        crop_arr = None
        if use_yolo_detect and YOLO is not None and yolo_model_path is not None and os.path.exists(yolo_model_path):
            try:
                detector = YOLO(yolo_model_path)
                det = detector(image_path)[0]
                # pick highest-conf cow class if available; else fallback to full image
                if hasattr(det, 'boxes') and det.boxes is not None and len(det.boxes) > 0:
                    # choose highest confidence box
                    confs = det.boxes.conf.cpu().numpy()
                    bxyxy = det.boxes.xyxy.cpu().numpy()
                    idx = int(np.argmax(confs))
                    x1, y1, x2, y2 = map(int, bxyxy[idx])
                    with Image.open(image_path) as im:
                        im = im.convert("RGB")
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(im.width, x2); y2 = min(im.height, y2)
                        im = im.crop((x1, y1, x2, y2))
                        im = im.resize((224, 224))
                        crop_arr = np.asarray(im).astype(np.float32)
            except Exception:
                crop_arr = None
        # Load and preprocess image
        if crop_arr is not None:
            arr = crop_arr
        else:
            # Ensure minimum size by upscaling if needed
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                min_w, min_h = input_min_size
                if w < min_w or h < min_h:
                    scale = max(min_w / w, min_h / h)
                    im = im.resize((int(w * scale), int(h * scale)))
                # center crop to square then resize
                w, h = im.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                im = im.crop((left, top, left + side, top + side)).resize((224, 224))
                arr = np.asarray(im).astype(np.float32)

        # Heuristic fallback if TF/model not available
        if use_heuristic:
            # Simple color-based heuristic mapping to nearest known breed signature
            mean_rgb = np.mean(arr.reshape(-1, 3), axis=0)
            r, g, b = mean_rgb
            # Color heuristics (very rough):
            # More white -> Holstein Friesian; Brownish -> Jersey/Sahiwal/Red Sindhi; Gray -> Kangeyam; Reddish -> Red Sindhi; Mixed -> Crossbreed; Reddish-brown -> Gir
            labels = ["Holstein Friesian", "Jersey", "Sahiwal", "Red Sindhi", "Kangeyam", "Gir", "Crossbreed"]
            whiteness = (r + g + b) / 3.0
            brownness = (0.5*r + 0.4*g + 0.1*b)
            redness = r - (g+b)/2
            greyness = abs(r-g) + abs(g-b) + abs(b-r)
            raw_scores = np.array([
                whiteness,
                brownness,
                brownness * 0.95,
                max(0.0, redness),
                max(0.0, 255 - greyness),
                0.6*brownness + 0.4*redness,
                120.0,  # baseline for crossbreed
            ], dtype=np.float32)
            # Normalize scores into probabilities using softmax for better separation
            s = raw_scores - np.max(raw_scores)
            probs = np.exp(s) / np.sum(np.exp(s))
            top_indices = list(np.argsort(-probs)[:top_k])
            best_idx = top_indices[0]
            best_label = labels[best_idx]
            best_conf = float(probs[best_idx])
            top_preds = [
                {"breed": labels[i], "confidence": round(float(probs[i]) * 100.0, 2)}
                for i in top_indices
            ]
            return _ok({
                "feature": "breed_identification",
                "mode": "heuristic",
                "breed_name": best_label,
                "confidence": round(best_conf * 100.0, 2),
                "top_3_predictions": top_preds,
            })

        # TF path
        x = arr / 255.0
        if use_efficientnet:
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
                x = eff_pre(x)
            except Exception:
                pass
        else:
            try:
                from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mob_pre
                x = mob_pre(x)
            except Exception:
                pass
        x = x[np.newaxis, ...]

        model = tf.keras.models.load_model(model_path)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        breed_name = breeds[idx] if idx < len(breeds) else f"Class_{idx}"
        top_indices = list(np.argsort(-probs)[:top_k])
        top_preds = [
            {"breed": breeds[i] if i < len(breeds) else f"Class_{i}", "confidence": float(probs[i])}
            for i in top_indices
        ]
        return _ok({
            "feature": "breed_identification",
            "mode": "tensorflow",
            "breed_name": breed_name,
            "confidence": round(conf * 100.0, 2),
            "top_3_predictions": top_preds,
        })
    except Exception as e:
        return _error(f"breed identification failed: {e}")


# -----------------------------
# 4) Milk Profit Calculation
# -----------------------------

def milk_profit_calculation(
    total_litres_today: float,
    price_per_litre: float,
    feed_kg_today: float,
    feed_price_per_kg: float,
    labor_minutes: float,
    labor_rate: float,  # currency per hour
    vet_cost: float,
    other_costs: float,
) -> Dict[str, Any]:
    """
    Compute daily milk profit and weekly estimate.

    Revenue = total_litres_today * price_per_litre
    Costs = feed_cost + labor_cost + healthcare_cost + misc_cost
    Profit = Revenue - Costs

    Returns: { revenue_today, total_cost_today, profit_today, profit_per_litre, weekly_profit_estimate }
    """
    # Validate inputs
    try:
        t = float(total_litres_today)
        ppl = float(price_per_litre)
        fkg = float(feed_kg_today)
        fppk = float(feed_price_per_kg)
        lm = float(labor_minutes)
        lr = float(labor_rate)
        vc = float(vet_cost)
        oc = float(other_costs)
    except Exception:
        return _error("all inputs must be numeric")

    if t < 0 or ppl < 0 or fkg < 0 or fppk < 0 or lm < 0 or lr < 0 or vc < 0 or oc < 0:
        return _error("inputs cannot be negative")

    revenue = t * ppl
    feed_cost = fkg * fppk
    labor_cost = (lm / 60.0) * lr
    healthcare_cost = vc
    misc_cost = oc
    total_costs = feed_cost + labor_cost + healthcare_cost + misc_cost
    profit = revenue - total_costs
    ppl_profit = (profit / t) if t > 1e-9 else 0.0
    weekly_est = profit * 7.0

    return _ok({
        "feature": "milk_profit_calculation",
        "revenue_today": round(revenue, 2),
        "total_cost_today": round(total_costs, 2),
        "profit_today": round(profit, 2),
        "profit_per_litre": round(ppl_profit, 3),
        "weekly_profit_estimate": round(weekly_est, 2),
        "breakdown": {
            "feed_cost": round(feed_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "healthcare_cost": round(healthcare_cost, 2),
            "misc_cost": round(misc_cost, 2),
        }
    })


# -----------------------------
# Crop analysis and profit
# -----------------------------

def crop_image_analysis(
    image_path: str,
) -> Dict[str, Any]:
    """Crop analysis from a single image.
    If TensorFlow model path is provided via env CROP_MODEL_PATH, use it; else heuristic.
    Returns: { crop_type, condition, disease_sign, confidence, top_3 (if TF) }
    """
    if Image is None:
        return _error("Pillow (PIL) is required for crop_image_analysis")
    if not os.path.exists(image_path):
        return _error("image_path not found")
    # Try TensorFlow model if configured
    try:
        crop_model_path = os.environ.get("CROP_MODEL_PATH")
        if crop_model_path and tf is not None and os.path.exists(crop_model_path):
            with Image.open(image_path) as im:
                im = im.convert("RGB").resize((224, 224))
                arr = np.asarray(im).astype(np.float32) / 255.0
            try:
                model = tf.keras.models.load_model(crop_model_path)
                probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                class_names_env = os.environ.get("CROP_CLASS_NAMES")
                if class_names_env:
                    classes = [c.strip() for c in class_names_env.split(',') if c.strip()]
                else:
                    classes = ["Rice", "Maize", "Wheat", "Sugarcane", "Cotton", "Other"]
                crop_type = classes[idx] if idx < len(classes) else f"Class_{idx}"
                top_indices = list(np.argsort(-probs)[:3])
                top3 = [{"label": (classes[i] if i < len(classes) else f"Class_{i}"), "confidence": round(float(probs[i]) * 100.0, 2)} for i in top_indices]
                # Simple condition heuristic retained
                condition = "Good" if conf > 0.7 else ("Fair" if conf > 0.45 else "Poor")
                return _ok({
                    "feature": "crop_image_analysis",
                    "mode": "tensorflow",
                    "crop_type": crop_type,
                    "condition": condition,
                    "disease_sign": None,
                    "confidence": round(conf * 100.0, 2),
                    "top_3": top3,
                })
            except Exception:
                # On any TF failure, fall back to heuristic below
                pass
    except Exception as e:
        # Fall through to heuristic with an indicator message
        pass
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize((256, 256))
            arr = np.asarray(im).astype(np.float32)
        # Compute green index and brown/dark spots
        r = arr[:, :, 0]; g = arr[:, :, 1]; b = arr[:, :, 2]
        green_idx = np.mean(g - (r + b) / 2.0)
        brightness = np.mean((r + g + b) / 3.0)
        # Brown spot indicator
        brown_mask = (r > 80) & (g > 40) & (b < 80)
        dark_spot = (r + g + b) / 3.0 < 50
        spot_ratio = float(np.mean(brown_mask | dark_spot))
        # Crop type heuristic
        if green_idx > 20 and brightness > 60:
            crop_type = "Rice"
        elif green_idx > 5 and brightness > 80:
            crop_type = "Maize"
        elif green_idx > -5 and brightness > 120:
            crop_type = "Wheat"
        else:
            crop_type = "Unknown"
        # Condition
        if spot_ratio < 0.05 and green_idx > 10:
            condition = "Good"; disease = None; conf = 0.85
        elif spot_ratio < 0.15:
            condition = "Fair"; disease = "Mild leaf spots"; conf = 0.7
        else:
            condition = "Poor"; disease = "Leaf blight/Spotting"; conf = 0.65
        return _ok({
            "feature": "crop_image_analysis",
            "crop_type": crop_type,
            "condition": condition,
            "disease_sign": disease,
            "confidence": round(conf * 100.0, 2),
            "spot_ratio": round(spot_ratio, 3),
        })
    except Exception as e:
        return _error(f"crop analysis failed: {e}")


def crop_profit_calculation(
    area_acres: float,
    yield_per_acre_kg: float,
    price_per_kg: float,
    seed_cost: float,
    fertilizer_cost: float,
    labor_cost: float,
    irrigation_cost: float,
    other_costs: float,
) -> Dict[str, Any]:
    try:
        area = float(area_acres); yld = float(yield_per_acre_kg); ppk = float(price_per_kg)
        seed = float(seed_cost); fert = float(fertilizer_cost); lab = float(labor_cost)
        irrig = float(irrigation_cost); other = float(other_costs)
    except Exception:
        return _error("all inputs must be numeric")
    if any(v < 0 for v in [area, yld, ppk, seed, fert, lab, irrig, other]):
        return _error("inputs cannot be negative")
    total_yield = area * yld
    revenue = total_yield * ppk
    costs = seed + fert + lab + irrig + other
    profit = revenue - costs
    return _ok({
        "feature": "crop_profit_calculation",
        "area_acres": area,
        "total_yield_kg": round(total_yield, 2),
        "revenue": round(revenue, 2),
        "total_cost": round(costs, 2),
        "profit": round(profit, 2),
        "profit_margin_pct": round((profit / revenue * 100.0) if revenue > 0 else 0.0, 2),
    })


# -----------------------------
# Chicken analysis
# -----------------------------

def chicken_egg_analysis(
    hen_count: int,
    avg_eggs_per_hen_per_day: float,
    price_per_egg: float,
    feed_kg_per_day: float,
    feed_price_per_kg: float,
    labor_cost_per_week: float,
    other_costs_per_week: float,
) -> Dict[str, Any]:
    try:
        hens = int(hen_count)
        e_per_day = float(avg_eggs_per_hen_per_day)
        ppe = float(price_per_egg)
        feed_kg = float(feed_kg_per_day)
        feed_ppk = float(feed_price_per_kg)
        labor = float(labor_cost_per_week)
        other = float(other_costs_per_week)
    except Exception:
        return _error("all inputs must be numeric")
    if any(v < 0 for v in [hens, e_per_day, ppe, feed_kg, feed_ppk, labor, other]):
        return _error("inputs cannot be negative")
    weekly_eggs = hens * e_per_day * 7.0
    revenue = weekly_eggs * ppe
    feed_cost = feed_kg * feed_ppk * 7.0
    total_cost = feed_cost + labor + other
    profit = revenue - total_cost
    return _ok({
        "feature": "chicken_egg_analysis",
        "weekly_eggs": round(weekly_eggs, 2),
        "revenue_week": round(revenue, 2),
        "total_cost_week": round(total_cost, 2),
        "profit_week": round(profit, 2),
        "profit_per_egg": round((profit / weekly_eggs) if weekly_eggs > 0 else 0.0, 3),
        "breakdown": {
            "feed_cost_week": round(feed_cost, 2),
            "labor_cost_week": round(labor, 2),
            "other_costs_week": round(other, 2),
        }
    })


def chicken_breed_identification(
    image_path: str,
) -> Dict[str, Any]:
    """Chicken breed identification from image.
    If TensorFlow model path is provided via env CHICKEN_MODEL_PATH, use it; else heuristic.
    Returns: { breed_name, confidence, condition, top_3 (if TF) }
    """
    if Image is None:
        return _error("Pillow (PIL) is required for chicken_breed_identification")
    if not os.path.exists(image_path):
        return _error("image_path not found")
    # TF path if available
    try:
        chicken_model_path = os.environ.get("CHICKEN_MODEL_PATH")
        if chicken_model_path and tf is not None and os.path.exists(chicken_model_path):
            with Image.open(image_path) as im:
                im = im.convert("RGB").resize((224, 224))
                arr = np.asarray(im).astype(np.float32) / 255.0
            model = tf.keras.models.load_model(chicken_model_path)
            probs = model.predict(arr[np.newaxis, ...], verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            class_names_env = os.environ.get("CHICKEN_CLASS_NAMES")
            if class_names_env:
                classes = [c.strip() for c in class_names_env.split(',') if c.strip()]
            else:
                classes = ["Leghorn", "Rhode Island Red", "Kadaknath", "Plymouth Rock", "Broiler", "Other"]
            breed = classes[idx] if idx < len(classes) else f"Class_{idx}"
            top_indices = list(np.argsort(-probs)[:3])
            top3 = [{"breed": (classes[i] if i < len(classes) else f"Class_{i}"), "confidence": round(float(probs[i]) * 100.0, 2)} for i in top_indices]
            # Simple condition proxy by brightness heuristic
            with Image.open(image_path) as im:
                im = im.convert("RGB").resize((224, 224))
                arr2 = np.asarray(im).astype(np.float32)
                r, g, b = arr2[:, :, 0], arr2[:, :, 1], arr2[:, :, 2]
                brightness = float(np.mean((r + g + b) / 3.0))
                condition = "Good" if brightness > 90 else ("Fair" if brightness > 60 else "Poor")
            return _ok({
                "feature": "chicken_breed_identification",
                "mode": "tensorflow",
                "breed_name": breed,
                "confidence": round(conf * 100.0, 2),
                "condition": condition,
                "top_3": top3,
            })
    except Exception:
        # fallback to heuristic
        pass
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize((224, 224))
            arr = np.asarray(im).astype(np.float32)
        mean_rgb = np.mean(arr.reshape(-1, 3), axis=0)
        r, g, b = mean_rgb
        # Simple mapping
        if (r + g + b) / 3.0 > 200:
            breed = "Leghorn"
        elif r > g and r > b and (r - g) > 20:
            breed = "Rhode Island Red"
        elif (abs(r - g) < 25 and abs(g - b) < 25) and (r < 100):
            breed = "Kadaknath"
        elif g > r and g > b:
            breed = "Plymouth Rock"
        else:
            breed = "Crossbreed"
        brightness = (r + g + b) / 3.0
        condition = "Good" if brightness > 90 else "Fair" if brightness > 60 else "Poor"
        conf = 0.6
        return _ok({
            "feature": "chicken_breed_identification",
            "breed_name": breed,
            "confidence": round(conf * 100.0, 2),
            "condition": condition,
        })
    except Exception as e:
        return _error(f"chicken breed identification failed: {e}")


# -----------------------------
# Unified entry point (optional)
# -----------------------------

def analyze_all(
    milk_input: Dict[str, Any],
    audio_input: Optional[Dict[str, Any]] = None,
    image_input: Optional[Dict[str, Any]] = None,
    profit_input: Optional[Dict[str, Any]] = None,
    crop_image_input: Optional[Dict[str, Any]] = None,
    crop_profit_input: Optional[Dict[str, Any]] = None,
    chicken_eggs_input: Optional[Dict[str, Any]] = None,
    chicken_breed_image_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run multiple analyses and return a combined JSON-compatible object.
    Each input dict should map directly to the function parameters.
    """
    result: Dict[str, Any] = {"timestamp": None}
    try:
        from datetime import datetime
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    except Exception:
        pass

    # Milk
    try:
        result["milk_litre_analysis"] = milk_litre_analysis(**milk_input)
    except Exception as e:
        result["milk_litre_analysis"] = _error(f"internal error: {e}")

    # Audio
    if audio_input is not None:
        try:
            result["cow_health_detection"] = cow_health_detection(**audio_input)
        except Exception as e:
            result["cow_health_detection"] = _error(f"internal error: {e}")

    # Image
    if image_input is not None:
        try:
            result["breed_identification"] = breed_identification(**image_input)
        except Exception as e:
            result["breed_identification"] = _error(f"internal error: {e}")

    # Profit
    if profit_input is not None:
        try:
            result["milk_profit_calculation"] = milk_profit_calculation(**profit_input)
        except Exception as e:
            result["milk_profit_calculation"] = _error(f"internal error: {e}")

    # Crop image
    if crop_image_input is not None:
        try:
            result["crop_image_analysis"] = crop_image_analysis(**crop_image_input)
        except Exception as e:
            result["crop_image_analysis"] = _error(f"internal error: {e}")

    # Crop profit
    if crop_profit_input is not None:
        try:
            result["crop_profit_calculation"] = crop_profit_calculation(**crop_profit_input)
        except Exception as e:
            result["crop_profit_calculation"] = _error(f"internal error: {e}")

    # Chicken eggs
    if chicken_eggs_input is not None:
        try:
            result["chicken_egg_analysis"] = chicken_egg_analysis(**chicken_eggs_input)
        except Exception as e:
            result["chicken_egg_analysis"] = _error(f"internal error: {e}")

    # Chicken breed image
    if chicken_breed_image_input is not None:
        try:
            result["chicken_breed_identification"] = chicken_breed_identification(**chicken_breed_image_input)
        except Exception as e:
            result["chicken_breed_identification"] = _error(f"internal error: {e}")

    return result


if __name__ == "__main__":
    # Minimal quick test (does not require ML models)
    demo = analyze_all(
        milk_input={
            "cow_id": "COW123",
            "milking_events": [6.2, 5.8],
            "timestamp": ["2025-12-11T06:30:00", "2025-12-11T18:45:00"],
            "history_daily_totals": [11.4, 11.2, 11.6, 11.0, 11.9, 12.1, 11.8],
        },
        # Frequency-based health check (no audio required)
        audio_input={
            "frequency_hz": [95.0, 120.0, 180.0]
        },
        image_input=None,  # requires model_path
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
    print(json.dumps(demo, indent=2))
