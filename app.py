"""Gradio web demo.  python app.py --checkpoint checkpoints/best.pt"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from deepfake_detector.predict import DeepfakeDetector  # noqa: E402


def build_demo(detector, demo_mode=False):
    def on_image(img):
        if img is None:
            raise gr.Error("Please upload an image.")
        r = detector.predict_image(img)
        note = "" if r["face_detected"] else "No face detected - analysed the full image (less reliable)."
        p = r["fake_probability"]
        return {"Fake": p, "Real": 1 - p}, note

    def on_video(path, frames):
        if not path:
            raise gr.Error("Please upload a video.")
        r = detector.predict_video(path, num_frames=int(frames))
        p = r["fake_probability"]
        summary = (f"Verdict: {r['label']} | {r['frames_flagged_fake']}/{r['frames_analyzed']} "
                   f"frames flagged fake | faces found in {r['faces_detected_in_frames']} frames")
        return {"Fake": p, "Real": 1 - p}, summary

    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("# Deepfake Image & Video Detector\nProbabilistic estimate - not proof. See README for limitations.")
        if demo_mode:
            gr.Markdown("⚠️ **Demo mode:** no trained checkpoint was found. The generated checkpoint is only for testing the UI and pipeline; its predictions are **not valid deepfake detection results**. Train the model and use `checkpoints/best.pt` for real inference.")
        with gr.Tab("Image"):
            img = gr.Image(type="pil", label="Image")
            btn = gr.Button("Analyze")
            lbl, msg = gr.Label(label="Result"), gr.Textbox(label="Notes")
            btn.click(on_image, img, [lbl, msg])
        with gr.Tab("Video"):
            vid = gr.Video(label="Video")
            n = gr.Slider(4, 64, value=16, step=1, label="Frames to sample")
            btn2 = gr.Button("Analyze")
            lbl2, msg2 = gr.Label(label="Result"), gr.Textbox(label="Summary")
            btn2.click(on_video, [vid, n], [lbl2, msg2])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--demo", action="store_true", help="Use an untrained checkpoint for UI/pipeline testing.")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    demo_mode = args.demo
    if not checkpoint.exists():
        if not demo_mode:
            print(f"Checkpoint not found: {checkpoint}")
            print("No trained model is bundled with this project.")
            print("Starting in DEMO mode so the UI remains usable. Run training and restart with a real checkpoint for valid predictions.")
        demo_checkpoint = Path("checkpoints/demo.pt")
        if not demo_checkpoint.exists():
            from scripts.create_demo_checkpoint import main as create_demo_checkpoint
            old_argv = sys.argv
            try:
                sys.argv = ["create_demo_checkpoint", "--output", str(demo_checkpoint)]
                create_demo_checkpoint()
            finally:
                sys.argv = old_argv
        checkpoint = demo_checkpoint
        demo_mode = True

    build_demo(DeepfakeDetector(checkpoint), demo_mode=demo_mode).launch(share=args.share)
