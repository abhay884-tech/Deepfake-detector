# Deepfake Image & Video Detector

Deep-learning detector for manipulated faces in **images and videos**, built with PyTorch.
An ImageNet-pretrained **EfficientNet-B0** is fine-tuned as a binary classifier (real vs. fake) on face crops.
Video is handled by sampling frames, scoring each one, and aggregating the results.

Includes: training, evaluation, CLI inference, a Gradio web demo, data-prep tools, tests and CI.

> **Note:** this repo ships *code*, not trained weights. You train on a dataset of your choice (see below), then run inference.
> Accuracy depends entirely on your training data.

## Project layout
```
src/deepfake_detector/   model, dataset, transforms, face cropping, train, evaluate, predict
scripts/extract_frames.py  videos -> face-crop images (split by video, no leakage)
scripts/make_dummy_data.py synthetic data for smoke-testing the pipeline
app.py                   Gradio web UI
tests/                   pytest suite (run in CI)
```

## Setup
```bash
git clone https://github.com/<you>/deepfake-detector.git && cd deepfake-detector
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```
GPU is optional but strongly recommended for training (install the CUDA build of PyTorch from pytorch.org).

## 1. Get data
Public datasets (check each one's license/terms): FaceForensics++, Celeb-DF v2, DFDC (Kaggle), and
"140k Real and Fake Faces" (Kaggle, images).

**Images** - arrange as:
```
data/train/real  data/train/fake
data/val/real    data/val/fake
data/test/real   data/test/fake
```
**Videos** - put them in `raw_videos/real/` and `raw_videos/fake/`, then:
```bash
python scripts/extract_frames.py --input raw_videos --output data --frames-per-video 10
```
This detects faces, crops them, and splits **by video** so frames from one video never appear in both train and test.

## 2. Train
```bash
python -m deepfake_detector.train --data-dir data --epochs 10 --batch-size 32
```
Saves the best checkpoint (by validation AUC) to `checkpoints/best.pt` plus `history.json`.
Class imbalance is handled with a weighted loss; mixed precision is used automatically on GPU.

## 3. Evaluate
```bash
python -m deepfake_detector.evaluate --data-dir data/test --checkpoint checkpoints/best.pt
```
Prints precision/recall/F1, ROC-AUC and saves `checkpoints/confusion_matrix.png`.

## 4. Predict
```bash
python -m deepfake_detector.predict photo.jpg
python -m deepfake_detector.predict clip.mp4 --frames 32 --json
```
```python
from deepfake_detector import DeepfakeDetector
det = DeepfakeDetector("checkpoints/best.pt")
print(det.predict_image("photo.jpg"))
print(det.predict_video("clip.mp4", num_frames=16))
```
`fake_probability` is the model's score (label `FAKE` if >= threshold, default 0.5).

## 5. Web demo
```bash
python app.py                              # starts the UI; if no model exists, starts clearly-labeled DEMO mode
python app.py --checkpoint checkpoints/best.pt  # use a trained model
python app.py --demo                       # explicitly use the untrained demo checkpoint
```

If `checkpoints/best.pt` does not exist, the app automatically creates `checkpoints/demo.pt` so the image/video UI can be tested immediately. **Demo predictions are not meaningful deepfake predictions.** Train a real model before using the detector for inference.

## Smoke test (no real data needed)
```bash
python scripts/make_dummy_data.py --output data_dummy
python -m deepfake_detector.train --data-dir data_dummy --epochs 2 --no-pretrained --workers 0
pytest -q
```

## How it works
1. **Face crop** - OpenCV Haar cascade finds the largest face; crop with 30% margin (falls back to the full frame, flagged in output).
2. **Classifier** - EfficientNet-B0, single-logit head, BCE loss, AdamW + cosine LR.
3. **Video** - evenly spaced frames -> per-frame probabilities -> mean aggregate (per-frame scores returned too).

## Limitations (please read)
- Detectors **generalize poorly** to manipulation methods unseen in training; a model trained on one dataset can drop sharply on another.
- Compression, low resolution and unusual lighting hurt accuracy. Haar face detection misses non-frontal faces.
- Output is a probabilistic estimate, **not proof**. Don't use it as sole evidence for decisions affecting people.

## Ideas for improvement
Stronger face detector (MTCNN/RetinaFace), temporal models (LSTM/Transformer over frame features), frequency-domain features,
Grad-CAM explanations, cross-dataset evaluation, JPEG/compression augmentation.

## License
MIT

## Streamlit Cloud deployment

This repository is ready for Streamlit Community Cloud.

1. Push the **contents of this repository** to GitHub (including the `src/` directory).
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the main file to `app.py` and branch to `main`.
4. Deploy.

If `checkpoints/best.pt` is not present, the app automatically creates an untrained demo checkpoint so the UI can start. Demo predictions are not meaningful for real deepfake detection. Replace it with a trained `best.pt` for real inference.
