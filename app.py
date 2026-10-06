"""Streamlit web app for image/video deepfake inference."""
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
from deepfake_detector.predict import DeepfakeDetector  # noqa: E402
from deepfake_detector.model import build_model  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "best.pt"
DEMO_CHECKPOINT = ROOT / "checkpoints" / "demo.pt"


def ensure_demo_checkpoint():
    """Create a small untrained checkpoint so the Streamlit UI can start."""
    if DEMO_CHECKPOINT.exists():
        return DEMO_CHECKPOINT
    DEMO_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    import torch

    model = build_model(pretrained=False)
    torch.save({"model_state": model.state_dict(), "img_size": 224, "demo_only": True}, DEMO_CHECKPOINT)
    return DEMO_CHECKPOINT


@st.cache_resource(show_spinner=False)
def load_detector(checkpoint_path: str):
    return DeepfakeDetector(checkpoint_path)


st.set_page_config(page_title="Deepfake Detector", page_icon="🔎", layout="centered")
st.title("🔎 Deepfake Image & Video Detector")
st.caption("Probabilistic estimate — not proof of manipulation.")

if CHECKPOINT.exists():
    checkpoint = CHECKPOINT
    st.success("Trained checkpoint found. Real inference mode is active.")
else:
    checkpoint = ensure_demo_checkpoint()
    st.warning(
        "No trained checkpoint (checkpoints/best.pt) was found. Demo Mode is active. "
        "Demo predictions are NOT meaningful for real deepfake detection."
    )

try:
    detector = load_detector(str(checkpoint))
except Exception as exc:
    st.error("Could not load the detector model.")
    st.exception(exc)
    st.stop()

mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze image", type="primary"):
            try:
                result = detector.predict_image(uploaded)
                p = result["fake_probability"]
                col1, col2 = st.columns(2)
                col1.metric("Verdict", result["label"])
                col2.metric("Fake probability", f"{p:.1%}")
                if not result["face_detected"]:
                    st.info("No face detected; the full image was analyzed and the result may be less reliable.")
            except Exception as exc:
                st.error("Image analysis failed.")
                st.exception(exc)

else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    frames = st.slider("Frames to sample", 4, 64, 16)
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            st.video(uploaded)
            if st.button("Analyze video", type="primary"):
                with st.spinner("Analyzing video frames..."):
                    result = detector.predict_video(tmp_path, num_frames=frames)
                col1, col2 = st.columns(2)
                col1.metric("Verdict", result["label"])
                col2.metric("Fake probability", f"{result['fake_probability']:.1%}")
                st.write(
                    f"**Frames flagged fake:** {result['frames_flagged_fake']}/"
                    f"{result['frames_analyzed']}  "
                    f"| **Faces found:** {result['faces_detected_in_frames']}"
                )
        except Exception as exc:
            st.error("Video analysis failed.")
            st.exception(exc)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

with st.expander("About / limitations"):
    st.write(
        "This application estimates whether an image or sampled video frames look fake. "
        "A trained checkpoint is required for meaningful predictions. Results can be wrong, "
        "especially with unusual lighting, compression, low resolution, or faces not detected."
    )
