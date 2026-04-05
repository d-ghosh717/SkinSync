# SkinSync

SkinSync is an AI-powered beauty-tech system that analyzes facial skin and provides personalized foundation recommendations along with a real-time virtual try-on experience.

It offers two modes:

* Desktop (main.py) → Live webcam-based try-on
* Web App (web_server.py) → Image upload + analysis + recommendations

---

## Features 🚀

* 🎯 Accurate face detection using MediaPipe FaceMesh
* 🎨 Skin tone detection using LAB + ITA
* 🌡️ Undertone classification (Warm / Cool / Neutral)
* 🧴 Skin texture analysis (Smooth / Oily / Dry / Combination)
* 💡 Lighting correction using CLAHE
* 🪞 Real-time virtual foundation try-on
* 🛍️ Smart foundation recommendation system
* 🌐 Web API for image-based analysis

---

## Installation ⚙️

1. Clone the repository:

```
git clone https://github.com/your-username/skinsync.git
cd skinsync
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage 🧑‍💻

### 1. Real-Time Try-On (Terminal Mode)

Run:

```
python main.py
```

What it does:

* Uses webcam for live skin analysis
* Displays tone, undertone, and texture
* Applies foundation in real time

Controls:

* V → Toggle try-on
* * / - → Adjust intensity
* T → Toggle landmarks
* Q → Quit

---

### 2. Web App (Image Analysis + Recommendation)

Run:

```
python web_server.py
```

Open:

```
http://127.0.0.1:5000
```

What it does:

* Upload an image
* Detects skin tone, undertone, texture
* Suggests best matching foundation shades
* Displays results in a clean UI

---

## How It Works ⚡

* Face Detection → MediaPipe FaceMesh
* Skin Extraction → Forehead, cheeks, chin
* Preprocessing → LAB + CLAHE
* Analysis → Tone + Undertone + Texture
* Matching → Compare with foundation database
* Try-On → Apply foundation using alpha blending

---

## Requirements 📦

* Python 3.8+
* OpenCV
* MediaPipe
* Flask
* NumPy
* Pandas

---

## Future Scope 🔮

* 📱 Mobile app version
* 🧠 Advanced AI models for deeper skin analysis
* 🎯 More brands and shade database
* 🕶️ AR-based try-on improvements

---

## License 📄

This project is for academic and research purposes.

---

## Author 👨‍💻

Developed as part of an AI + Computer Vision project.
