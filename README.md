# 🛡️ DeepGuard — Deepfake Detector
deepfake-detector-2v2qt6yzzxdxtve2pczn3v.streamlit.app
A lightweight, GitHub-ready **Streamlit deepfake screening application** for images.

## Features

- Modern dark dashboard UI
- Upload JPG, JPEG, PNG, and WebP images
- Explainable forensic signals:
  - texture anomaly
  - local inconsistency
  - chroma irregularity
  - entropy signal
- Adjustable sensitivity threshold
- Verdict: **Likely Authentic**, **Potential Deepfake**, or **Needs Review**
- Confidence indicator
- SHA-256 file hash in downloadable report
- No API key required
- Small dependency footprint, suitable for Streamlit Community Cloud

> **Important:** This is an educational/research screening prototype. Image-forensic heuristics can produce false positives and false negatives. Do not use the result as sole evidence for identity, moderation, employment, legal, or financial decisions.

## Project structure

```text
deepfake-detector/
├── app.py
├── requirements.txt
├── README.md
├── tests/
│   └── test_detector.py
├── .gitignore
└── LICENSE
```

## Run locally

### 1. Install Python

Python 3.10+ is recommended.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

The browser will open at the local Streamlit address.

## Test before deployment

Install dependencies and run:

```bash
python tests/test_detector.py
```

Expected output:

```text
DeepGuard detector tests: PASSED
```

Then launch the application:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the project files, including the `tests` folder.
3. Open Streamlit Community Cloud.
4. Choose **Deploy an app**.
5. Select your repository and `app.py`.
6. Deploy.

No secrets are required for this version.

## Push using Git

```bash
git init
git add .
git commit -m "Initial DeepGuard deepfake detector"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/deepfake-detector.git
git push -u origin main
```

## Detection approach

The current version intentionally uses deterministic image-forensics features rather than downloading a very large neural model during deployment. This makes it faster and more reliable for a student/demo deployment.

The anomaly score combines several measurable image characteristics and then uses a conservative threshold:

- **Low anomaly:** likely authentic
- **Middle range:** needs review
- **High anomaly:** potential deepfake

For production-grade detection, replace `analyze_image()` with a validated face-forensics neural model and benchmark it on a representative dataset.

## License

MIT
