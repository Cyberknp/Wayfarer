# Wayfarer

Wayfarer is a comprehensive, security-focused platform designed for personal and campus safety. Featuring a modern frontend experience built with React and Tailwind CSS, and a new intelligent backend powered by machine learning, Wayfarer emphasizes real‑time awareness, dynamic alerting, and community protection.

## Highlights
- **Unified Single-Page App**: Seamless navigation using `app.html` built with React.
- **AI-Powered Risk Detection**: Built-in machine learning models (`Anomaly_detector.py`) to analyze suspicious activity and automatically identify danger zones.
- **Interactive Security Tools**: Live Monitor, Danger Map, Incident Reporting, and Security Command Dashboard.
- **Dark Security Palette**: High-contrast, modern UI designed for rapid situational assessment.
- **No Complex Build Step**: The frontend uses CDN-based React and Tailwind for fast iteration.

## Project Structure
```
Wayfarer/
├─ app.html                 # Consolidated single-page React application
├─ Backend/
│  ├─ Anomaly_detector.py   # AI/ML-based anomaly and danger zone detector
│  └─ ai.js                 # Helper logic for risk level assessment
├─ Frontend/                # Individual standalone screen prototypes
│  ├─ welcome.html
│  ├─ login.html
│  ├─ index.html
│  ├─ securitydash.html
│  ├─ dangermap.html
│  ├─ livemonitor.html
│  ├─ reportincid.html
│  └─ profile.html
├─ .venv/                   # Python virtual environment
├─ requirements.txt         # Project dependencies list
└─ LICENSE
```

## Getting Started

### 1. Frontend Execution
The simplest way to use Wayfarer is to open the consolidated application in your browser:
- **`app.html`** — Launch the full single-page application experience.

Alternatively, you can test isolated UI screens by opening any HTML file from the `Frontend/` folder.

### 2. Backend Environment Setup
The backend utilizes Python-based Machine Learning models (Isolation Forest, KMeans) to power the incident detection logic.
To interact with the backend AI:
1. Ensure your terminal is in the root directory.
2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Run the ML pipeline (requires libraries like `scikit-learn`, `pandas`, `numpy` if fully deployed):
   ```bash
   python Backend/Anomaly_detector.py
   ```

## Tech Stack
- **Frontend**: React 18 (CDN), Tailwind CSS (CDN), Leaflet Maps
- **Backend / AI**: Python 3.x, `scikit-learn`, `pandas`, `numpy`
- **Typography & Design**: Google Fonts (Inter), custom dark UI palette