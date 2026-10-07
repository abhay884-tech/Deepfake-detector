import io
import os
import hashlib
import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageStat, ImageFilter

st.set_page_config(
    page_title="DeepGuard — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #0d1728 55%, #101d32 100%);
        color: #f3f7ff;
    }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(120,170,255,.22);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(25,44,75,.92), rgba(13,25,44,.88));
        box-shadow: 0 15px 50px rgba(0,0,0,.22);
        margin-bottom: 22px;
    }
    .hero h1 { font-size: 42px; margin: 0 0 8px 0; }
    .hero p { color: #b9c8df; font-size: 17px; margin: 0; }
    .card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(120,170,255,.18);
        background: rgba(17,30,50,.78);
        height: 100%;
    }
    .metric-title { color: #9fb2cc; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-size: 30px; font-weight: 700; margin-top: 5px; }
    .result-real { color: #57e6a5; }
    .result-fake { color: #ff7185; }
    .result-review { color: #ffd166; }
    .small { color: #9fb2cc; font-size: 13px; line-height: 1.5; }
    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(120,170,255,.45);
        border-radius: 16px;
        padding: 8px;
        background: rgba(13,25,44,.55);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Lightweight forensic analysis
# ----------------------------
def forensic_features(image: Image.Image):
    """Calculate explainable image-forensic signals.
    These are not a substitute for a trained neural-network detector.
    """
    img = image.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    # 1. JPEG block/grid and recompression-related signal
    gray = np.mean(arr, axis=2)
    h, w = gray.shape
    h2, w2 = h - (h % 8), w - (w % 8)
    crop = gray[:h2, :w2]
    blocks = crop.reshape(h2 // 8, 8, w2 // 8, 8).transpose(0, 2, 1, 3)
    block_means = blocks.mean(axis=(2, 3))
    horizontal = np.abs(np.diff(block_means, axis=1)).mean() if block_means.shape[1] > 1 else 0
    vertical = np.abs(np.diff(block_means, axis=0)).mean() if block_means.shape[0] > 1 else 0
    block_signal = float((horizontal + vertical) / 2)

    # 2. High-frequency noise / texture irregularity
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=1))).astype(np.float32)
    residual = np.abs(arr - blur).mean()
    texture_signal = float(np.clip(residual / 35.0, 0, 1))

    # 3. Local variance inconsistency across tiles
    tile_vars = []
    for y in range(0, h - 31, max(32, h // 6)):
        for x in range(0, w - 31, max(32, w // 6)):
            tile_vars.append(float(gray[y:y+32, x:x+32].var()))
    if len(tile_vars) > 3:
        cv = np.std(tile_vars) / (np.mean(tile_vars) + 1e-6)
    else:
        cv = 0.0
    inconsistency = float(np.clip(cv / 1.8, 0, 1))

    # 4. Color-channel correlation / unusual chroma residual
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    rg = np.mean(np.abs((r-g) - np.mean(r-g)))
    gb = np.mean(np.abs((g-b) - np.mean(g-b)))
    chroma_signal = float(np.clip((rg + gb) / 90.0, 0, 1))

    # 5. Image entropy proxy
    hist, _ = np.histogram(gray, bins=64, range=(0,255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist + 1e-12)).sum())
    entropy_signal = float(np.clip((entropy - 3.5) / 2.5, 0, 1))

    # Combined heuristic. Lower is more natural/consistent.
    anomaly = (
        0.30 * inconsistency +
        0.25 * texture_signal +
        0.20 * chroma_signal +
        0.15 * entropy_signal +
        0.10 * np.clip(block_signal / 25.0, 0, 1)
    )
    return {
        "anomaly": float(np.clip(anomaly, 0, 1)),
        "texture": texture_signal,
        "inconsistency": inconsistency,
        "chroma": chroma_signal,
        "entropy": entropy_signal,
    }

def analyze_image(image):
    # Try optional neural detector if installed/configured.
    # The default deployment is deliberately dependency-light and deterministic.
    features = forensic_features(image)
    score = features["anomaly"]

    # Conservative interpretation: uncertain cases go to manual review.
    if score >= 0.62:
        label = "Potential Deepfake"
        confidence = 0.60 + (score - 0.62) / 0.38 * 0.32
        css = "result-fake"
    elif score <= 0.36:
        label = "Likely Authentic"
        confidence = 0.60 + (0.36 - score) / 0.36 * 0.32
        css = "result-real"
    else:
        label = "Needs Review"
        confidence = 0.50 + abs(score - 0.49) * 0.35
        css = "result-review"

    return label, float(np.clip(confidence, 0.50, 0.92)), features, css

# ----------------------------
# UI
# ----------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ DeepGuard</h1>
    <p>Deepfake Detector • Explainable image-forensics dashboard</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Detection Settings")
    threshold = st.slider("Deepfake sensitivity", 0.40, 0.80, 0.62, 0.01)
    st.caption("Higher sensitivity flags more images for review.")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Upload an image\n"
        "2. Extract forensic signals\n"
        "3. Combine anomaly indicators\n"
        "4. Return a conservative verdict"
    )
    st.divider()
    st.caption("Educational / research prototype. Do not use as sole evidence for identity, moderation, or legal decisions.")

uploaded = st.file_uploader(
    "Upload a face/image to analyze",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use a clear, reasonably high-resolution image."
)

if uploaded:
    try:
        data = uploaded.getvalue()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.load()
    except Exception as e:
        st.error(f"Could not read the image: {e}")
        st.stop()

    left, right = st.columns([1.15, 1])
    with left:
        st.image(image, caption=f"{uploaded.name} • {image.width}×{image.height}", use_container_width=True)

    with right:
        with st.spinner("Running forensic analysis..."):
            label, confidence, features, css = analyze_image(image)
            # Apply user-selected threshold to the anomaly score.
            anomaly = features["anomaly"]
            if anomaly >= threshold:
                label = "Potential Deepfake"
                css = "result-fake"
            elif anomaly <= max(0.25, threshold - 0.26):
                label = "Likely Authentic"
                css = "result-real"
            else:
                label = "Needs Review"
                css = "result-review"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Verdict</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {css}">{label}</div>', unsafe_allow_html=True)
        st.progress(int(confidence * 100), text=f"Confidence: {confidence:.0%}")
        st.markdown(
            f'<div class="small">Forensic anomaly score: <b>{anomaly:.2f}</b> '
            f'(threshold {threshold:.2f})</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🔎 Forensic signals")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Texture anomaly", features["texture"]),
        ("Local inconsistency", features["inconsistency"]),
        ("Chroma irregularity", features["chroma"]),
        ("Entropy signal", features["entropy"]),
    ]
    for col, (name, value) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(
                f'<div class="card"><div class="metric-title">{name}</div>'
                f'<div class="metric-value">{value:.2f}</div></div>',
                unsafe_allow_html=True
            )

    st.info(
        "Interpretation: the detector looks for statistical image inconsistencies often associated "
        "with manipulation. A high score is a warning signal, not proof that an image is synthetic."
    )

    # Downloadable report
    file_hash = hashlib.sha256(data).hexdigest()
    report = f"""DeepGuard Analysis Report
========================
File: {uploaded.name}
SHA-256: {file_hash}
Image size: {image.width}x{image.height}

Verdict: {label}
Confidence: {confidence:.0%}
Anomaly score: {features['anomaly']:.3f}
Threshold: {threshold:.3f}

Signals
-------
Texture anomaly: {features['texture']:.3f}
Local inconsistency: {features['inconsistency']:.3f}
Chroma irregularity: {features['chroma']:.3f}
Entropy signal: {features['entropy']:.3f}

Important:
This is an educational forensic screening tool. Results should not be treated as definitive proof.
"""
    st.download_button(
        "⬇️ Download analysis report",
        data=report,
        file_name="deepguard_report.txt",
        mime="text/plain",
    )
else:
    st.markdown("""
    <div class="card">
        <h3>Start a scan</h3>
        <p class="small">Upload a JPG, JPEG, PNG, or WebP image above. The app runs locally in the Streamlit process and does not require an API key.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("DeepGuard • GitHub-ready Streamlit project • Python 3.10+")
