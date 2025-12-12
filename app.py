from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Any
import uvicorn
import os
import shutil
import tempfile

from smart_cow_analysis import (
    milk_litre_analysis,
    cow_health_detection,
    breed_identification,
    milk_profit_calculation,
    crop_image_analysis,
    crop_profit_calculation,
    chicken_egg_analysis,
    chicken_breed_identification,
    analyze_all,
)

app = FastAPI(title="Smart Cow Analysis System")

# Serve static files (index.html, JS, CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root_page():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h3>Smart Cow Analysis API</h3><p>Static UI missing. Use the API endpoints.</p>")


@app.post("/api/milk")
async def api_milk(
    cow_id: str = Form(...),
    milking_events: str = Form(...),  # comma-separated floats
    timestamp: str = Form(...),       # comma-separated ISO times
    lactation_days: Optional[int] = Form(None),
    feed_intake: Optional[float] = Form(None),
    cow_weight: Optional[float] = Form(None),
    history_daily_totals: Optional[str] = Form(None),  # comma-separated floats
    model_type: str = Form("auto"),
):
    try:
        events = [float(x) for x in milking_events.split(',') if x.strip()]
        times = [x.strip() for x in timestamp.split(',') if x.strip()]
        hist = None
        if history_daily_totals:
            hist = [float(x) for x in history_daily_totals.split(',') if x.strip()]
        res = milk_litre_analysis(
            cow_id=cow_id,
            milking_events=events,
            timestamp=times,
            lactation_days=lactation_days,
            feed_intake=feed_intake,
            cow_weight=cow_weight,
            history_daily_totals=hist,
            model_type=model_type,
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/health")
async def api_health(
    frequency_hz: Optional[str] = Form(None),  # number or comma-separated numbers
    audio: Optional[UploadFile] = File(None),
    model_path: Optional[str] = Form(None),
):
    try:
        freq_value: Optional[Any] = None
        if frequency_hz:
            # Try parse float or list of floats
            parts = [p.strip() for p in frequency_hz.split(',') if p.strip()]
            if len(parts) == 1:
                try:
                    freq_value = float(parts[0])
                except Exception:
                    freq_value = [float(x) for x in parts]
            else:
                freq_value = [float(x) for x in parts]
        if freq_value is not None:
            return JSONResponse(cow_health_detection(frequency_hz=freq_value))
        # Audio path flow
        if audio is None:
            return JSONResponse({"status": "error", "message": "Provide frequency_hz or upload an audio file"})
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1] or ".wav") as tf:
            shutil.copyfileobj(audio.file, tf)
            temp_path = tf.name
        try:
            res = cow_health_detection(audio_path=temp_path, model_path=model_path)
            return JSONResponse(res)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/breed")
async def api_breed(
    image: UploadFile = File(...),
    model_path: Optional[str] = Form(None),
    # Deprecated/optional fields for backward compatibility
    use_efficientnet: Optional[bool] = Form(None),
    use_yolo_detect: Optional[bool] = Form(None),
    yolo_model_path: Optional[str] = Form(None),
):
    try:
        # Resolve model path if available; allow None for heuristic fallback
        model_path_resolved = model_path or os.environ.get("BREED_MODEL_PATH")
        # Resolve options (defaults if not sent)
        use_eff = True if use_efficientnet is None else use_efficientnet
        # Auto-enable YOLO if env path exists
        env_yolo_path = os.environ.get("YOLO_MODEL_PATH")
        if yolo_model_path is None and env_yolo_path and os.path.exists(env_yolo_path):
            yolo_model_path = env_yolo_path
            use_yolo_detect = True if use_yolo_detect is None else use_yolo_detect
        use_yolo = False if use_yolo_detect is None else use_yolo_detect
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1] or ".jpg") as tf:
            shutil.copyfileobj(image.file, tf)
            temp_path = tf.name
        try:
            res = breed_identification(
                image_path=temp_path,
                model_path=model_path_resolved,
                use_efficientnet=use_eff,
                use_yolo_detect=use_yolo,
                yolo_model_path=yolo_model_path,
            )
            return JSONResponse(res)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# ---------------- Crop endpoints -----------------
@app.post("/api/crop/analyze_image")
async def api_crop_analyze_image(image: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1] or ".jpg") as tf:
            shutil.copyfileobj(image.file, tf)
            temp_path = tf.name
        try:
            res = crop_image_analysis(image_path=temp_path)
            return JSONResponse(res)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/crop/profit")
async def api_crop_profit(
    area_acres: float = Form(...),
    yield_per_acre_kg: float = Form(...),
    price_per_kg: float = Form(...),
    seed_cost: float = Form(...),
    fertilizer_cost: float = Form(...),
    labor_cost: float = Form(...),
    irrigation_cost: float = Form(...),
    other_costs: float = Form(...),
):
    try:
        res = crop_profit_calculation(
            area_acres,
            yield_per_acre_kg,
            price_per_kg,
            seed_cost,
            fertilizer_cost,
            labor_cost,
            irrigation_cost,
            other_costs,
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


# ---------------- Chicken endpoints -----------------
@app.post("/api/chicken/eggs")
async def api_chicken_eggs(
    hen_count: int = Form(...),
    avg_eggs_per_hen_per_day: float = Form(...),
    price_per_egg: float = Form(...),
    feed_kg_per_day: float = Form(...),
    feed_price_per_kg: float = Form(...),
    labor_cost_per_week: float = Form(...),
    other_costs_per_week: float = Form(...),
):
    try:
        res = chicken_egg_analysis(
            hen_count,
            avg_eggs_per_hen_per_day,
            price_per_egg,
            feed_kg_per_day,
            feed_price_per_kg,
            labor_cost_per_week,
            other_costs_per_week,
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/chicken/breed")
async def api_chicken_breed(image: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1] or ".jpg") as tf:
            shutil.copyfileobj(image.file, tf)
            temp_path = tf.name
        try:
            res = chicken_breed_identification(image_path=temp_path)
            return JSONResponse(res)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/profit")
async def api_profit(
    total_litres_today: float = Form(...),
    price_per_litre: float = Form(...),
    feed_kg_today: float = Form(...),
    feed_price_per_kg: float = Form(...),
    labor_minutes: float = Form(...),
    labor_rate: float = Form(...),
    vet_cost: float = Form(...),
    other_costs: float = Form(...),
):
    try:
        res = milk_profit_calculation(
            total_litres_today,
            price_per_litre,
            feed_kg_today,
            feed_price_per_kg,
            labor_minutes,
            labor_rate,
            vet_cost,
            other_costs,
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/api/analyze_all")
async def api_analyze_all(
    # Milk
    cow_id: str = Form(...),
    milking_events: str = Form(...),
    timestamp: str = Form(...),
    history_daily_totals: Optional[str] = Form(None),
    model_type: str = Form("auto"),
    # Health
    frequency_hz: Optional[str] = Form(None),
    # Profit
    total_litres_today: float = Form(...),
    price_per_litre: float = Form(...),
    feed_kg_today: float = Form(...),
    feed_price_per_kg: float = Form(...),
    labor_minutes: float = Form(...),
    labor_rate: float = Form(...),
    vet_cost: float = Form(...),
    other_costs: float = Form(...),
):
    try:
        events = [float(x) for x in milking_events.split(',') if x.strip()]
        times = [x.strip() for x in timestamp.split(',') if x.strip()]
        hist = None
        if history_daily_totals:
            hist = [float(x) for x in history_daily_totals.split(',') if x.strip()]

        freq_value: Optional[Any] = None
        if frequency_hz:
            parts = [p.strip() for p in frequency_hz.split(',') if p.strip()]
            if len(parts) == 1:
                try:
                    freq_value = float(parts[0])
                except Exception:
                    freq_value = [float(x) for x in parts]
            else:
                freq_value = [float(x) for x in parts]

        res = analyze_all(
            milk_input={
                "cow_id": cow_id,
                "milking_events": events,
                "timestamp": times,
                "history_daily_totals": hist,
                "model_type": model_type,
            },
            audio_input={"frequency_hz": freq_value} if freq_value is not None else None,
            image_input=None,
            profit_input={
                "total_litres_today": total_litres_today,
                "price_per_litre": price_per_litre,
                "feed_kg_today": feed_kg_today,
                "feed_price_per_kg": feed_price_per_kg,
                "labor_minutes": labor_minutes,
                "labor_rate": labor_rate,
                "vet_cost": vet_cost,
                "other_costs": other_costs,
            }
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
