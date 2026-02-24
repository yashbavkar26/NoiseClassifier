from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import librosa
import tensorflow as tf
import smtplib
import ssl
import requests
from email.message import EmailMessage
import os

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔐 API KEY
# =========================
API_KEY = os.getenv("API_KEY","YASH123456")

# =========================
# 🎵 Audio Settings
# =========================
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40

# =========================
# 📍 Store Latest Location
# =========================
LATEST_LOCATION = {"latitude": "Unknown", "longitude": "Unknown"}

#Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# Reverse Geocoding

def get_location_name(latitude, longitude):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json"
        }

        headers = {
            "User-Agent": "NoiseClassifierApp/1.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()

        return data.get("display_name", "Unknown Location")

    except Exception as e:
        print(f"Location fetch error: {e}")
        return "Unknown Location"

# =========================
# 📧 Email Alert Function
# =========================
def send_email_alert(noise_class, confidence, location):
    latitude = location.get("latitude", "Unknown")
    longitude = location.get("longitude", "Unknown")

    location_name = "Unknown Location"

    if latitude != "Unknown" and longitude != "Unknown":
        location_name = get_location_name(latitude, longitude)

    maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"

    subject = f"⚠️ Harmful Noise Alert: {noise_class} Detected!"

    body = f"""
⚠️ Harmful Noise Detected!

🔊 Noise Type: {noise_class}
📊 Confidence: {confidence}%

📍 Approx Location:
{location_name}

📌 Coordinates:
Latitude: {latitude}
Longitude: {longitude}

🗺 View on Map:
{maps_link}

Please take necessary action.
"""

    em = EmailMessage()
    em["From"] = EMAIL_SENDER
    em["To"] = EMAIL_RECEIVER
    em["Subject"] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(em)
        print(f"📧 Email alert sent for {noise_class}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# =========================
# 🧠 Load Model
# =========================
model = tf.keras.models.load_model("noise_classifier_model.h5")

CLASS_LABELS = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

HARMFUL_CLASSES = {"car_horn", "drilling", "gun_shot", "jackhammer", "siren"}

# =========================
# 🎵 MFCC Extraction
# =========================
def extract_mfcc_from_audio(audio_path):
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    if len(signal) < SAMPLES_PER_TRACK:
        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_TRACK]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    return mfcc

# =========================
# 📍 Location Endpoint
# =========================
class LocationData(BaseModel):
    latitude: float
    longitude: float

@app.post("/location")
async def receive_location(data: LocationData):
    global LATEST_LOCATION
    LATEST_LOCATION = {
        "latitude": data.latitude,
        "longitude": data.longitude
    }
    print(f"📍 Location Updated -> {data.latitude}, {data.longitude}")
    return {"status": "success", "received": data}

# =========================
# 🔍 Prediction Endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    temp_filename = "temp_audio.wav"
    audio_bytes = await file.read()

    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)

    mfcc = extract_mfcc_from_audio(temp_filename)

    predictions = model.predict(mfcc)
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    label = CLASS_LABELS[predicted_class]
    harmful = label in HARMFUL_CLASSES

    confidence_percent = round(confidence * 100, 2)

    if harmful:
        print(f"🚨 Harmful noise detected: {label} ({confidence_percent}%)")
        send_email_alert(label, confidence_percent, LATEST_LOCATION)

    return {
        "predicted_class": label,
        "confidence": confidence_percent,
        "harmful": harmful,
        "location": LATEST_LOCATION
    }
