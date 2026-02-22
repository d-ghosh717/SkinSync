"""
web_server.py – SkinSync Phase 4 (Rich API v2)
"""
import sys, os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_engine.detector  import FaceDetector
from face_engine.lighting  import LightingNormalizer
from face_engine.quality   import QualityChecker
from face_engine.skin_tone import SkinToneAnalyzer

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

# ── Engine ──────────────────────────────────────────────────────────────────
detector      = FaceDetector(max_num_faces=1)
normalizer    = LightingNormalizer(clip_limit=2.0, tile_grid_size=(8, 8))
checker       = QualityChecker()
tone_analyzer = SkinToneAnalyzer(window_size=1)

# ── Foundation Dataset ────────────────────────────────────────────────────────
def _hex_to_rgb(h: str):
    h = str(h).strip().lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _hex_to_lab_l(hex_str: str) -> float:
    r, g, b = _hex_to_rgb(hex_str)
    rgb = np.array([r, g, b], dtype=float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y   = float(0.2126*lin[0] + 0.7152*lin[1] + 0.0722*lin[2])
    fy  = Y**(1/3) if Y > 0.008856 else 7.787*Y + 16/116
    return max(0.0, 116*fy - 16)

FOUND_PATH = os.path.join(os.path.dirname(__file__), "data", "foundations.csv")
df_found   = None
if os.path.exists(FOUND_PATH):
    df_found = pd.read_csv(FOUND_PATH)
    df_found['L_value'] = df_found['hex'].apply(_hex_to_lab_l)
    df_found['R'] = df_found['hex'].apply(lambda h: _hex_to_rgb(h)[0])
    df_found['G'] = df_found['hex'].apply(lambda h: _hex_to_rgb(h)[1])
    df_found['B'] = df_found['hex'].apply(lambda h: _hex_to_rgb(h)[2])
    print(f"[INFO] Loaded {len(df_found)} foundation shades.")
else:
    print(f"[WARN] foundations.csv not found at {FOUND_PATH}")


# ── Match Score ───────────────────────────────────────────────────────────────
def _match_score(target_l: float, target_ut: str, row) -> float:
    """Return 0-100 match score. L close = 70pts, undertone match = 30pts."""
    l_diff   = abs(float(row['L_value']) - target_l)
    l_score  = max(0.0, 70.0 * (1 - min(l_diff, 40) / 40))
    ut_match = 30.0 if str(row.get('undertone','')).lower() == target_ut.lower() else 10.0
    return round(l_score + ut_match, 1)


def get_recommendations(target_l: float, undertone: str, skin_tone: str,
                        texture: str, skin_hex: str, top_n: int = 4) -> list[dict]:
    if df_found is None or df_found.empty:
        return []
    try:
        df = df_found.copy()
        df['score'] = df.apply(lambda r: _match_score(target_l, undertone, r), axis=1)
        df = df.sort_values('score', ascending=False).head(top_n)

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            R, G, B = int(row['R']), int(row['G']), int(row['B'])
            score   = float(row['score'])

            # Build why-it-matches bullets
            bullets = []
            l_diff = abs(float(row['L_value']) - target_l)
            bullets.append("Excellent color match" if l_diff < 5 else "Good color match")
            if str(row.get('undertone', '')).lower() == undertone.lower():
                bullets.append(f"Compatible {undertone.lower()} undertone")
            bullets.append(f"Perfect for {skin_tone} skin")
            bullets.append(f"{str(row.get('finish','')).capitalize()} {str(row.get('type',''))} ideal for {texture.lower()} skin")

            # Coverage-based description
            cov   = str(row.get('coverage', 'buildable'))
            desc  = str(row.get('description', 'Foundation with a natural finish'))
            sktyp = str(row.get('skin_type', 'all skin types'))

            results.append({
                "rank":       i + 1,
                "brand":      str(row['brand']),
                "shade":      str(row['shade']),
                "hex":        str(row['hex']),
                "skin_hex":   skin_hex,
                "undertone":  str(row.get('undertone', '')).capitalize(),
                "type":       str(row.get('type', '')).capitalize(),
                "finish":     str(row.get('finish', '')),
                "coverage":   cov,
                "price":      int(row['price']) if 'price' in row and str(row['price']).strip() else 0,
                "description": desc,
                "skin_type":  sktyp,
                "score":      int(score),
                "bullets":    bullets,
            })
        return results
    except Exception as e:
        print(f"[ERROR] foundation match: {e}")
        return []


# ── Quality warnings ─────────────────────────────────────────────────────────
def quality_warnings(frame: np.ndarray, quality) -> list[str]:
    warns = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mb = float(gray.mean())
    if mb < 60:
        warns.append("Image is too dark. Move to a well-lit area.")
    elif mb > 220:
        warns.append("Image is too bright. Avoid direct sunlight or flash.")
    if quality.blur_score < 80:
        warns.append("Image is blurry. Hold still and ensure good focus.")
    if quality.size_ratio < 0.08:
        warns.append("Move closer to the camera so your face fills more of the frame.")
    return warns


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    try:
        buf   = np.frombuffer(request.files['image'].read(), np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        norm   = normalizer.normalize(frame)
        lm_list = detector.process(norm)
        if not lm_list:
            return jsonify({"error": "No face detected. Ensure your face is clearly visible and well-lit."}), 400

        quality = checker.evaluate(norm, lm_list[0])
        warns   = quality_warnings(norm, quality)

        tone_analyzer.reset()
        result = tone_analyzer.analyze(norm, lm_list[0])
        if result is None:
            return jsonify({"error": "Could not extract skin tone."}), 400

        r, g, b = result.avg_rgb
        # Convert frame L to 0-100 scale for matching
        l_100 = result.l_value * 100 / 255

        recs = get_recommendations(
            target_l=l_100,
            undertone=result.undertone,
            skin_tone=result.tone,
            texture=result.texture_label,
            skin_hex=result.hex_color,
            top_n=4,
        )

        return jsonify({
            "success": True,
            "quality_warnings": warns,
            "profile": {
                "tone":           result.tone,
                "tone_label":     result.tone_label,
                "tone_type":      result.tone_type,
                "tone_desc":      result.tone_desc,
                "undertone":      result.undertone,
                "undertone_scores": result.undertone_scores,
                "texture":        result.texture,
                "texture_label":  result.texture_label,
                "texture_tags":   result.texture_tags,
                "finish_rec":     result.finish_rec,
                "foundation_types": result.foundation_types,
                "hex_color":      result.hex_color,
                "r": r, "g": g, "b": b,
                "luminance":      round(result.l_value, 1),
                "confidence":     round(result.confidence * 100),
            },
            "recommendations": recs,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("━" * 60)
    print("  SkinSync Web Server  →  http://127.0.0.1:5000")
    print("━" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
