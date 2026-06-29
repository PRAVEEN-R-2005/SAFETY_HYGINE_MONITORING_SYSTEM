# AI-Based Safety & Hygiene Monitoring System using YOLOv11

A modular, production-ready computer vision monitoring system that uses **YOLOv11** and **ByteTrack tracking** to identify workplace safety infractions (missing helmets, vests, gloves, masks), restricted area intrusions, fire, and smoke in real time. It stores violation incidents in a database, triggers alerts (SMTP Email, Twilio WhatsApp, popups, and siren chime), and displays analytics on a dark-themed glassmorphic Flask web dashboard.

---

## Key Features

1. **PPE Detection**: Identifies Person, Helmet, Safety Vest, Gloves, and Face Mask.
2. **Object Tracking**: Employs an IoU-based centroid tracker to trace individual worker compliance, avoiding duplicate alert spam.
3. **Safety Rule Engine**:
   - `Person + No Helmet` $\rightarrow$ Helmet Violation
   - `Person + No Vest` $\rightarrow$ Vest Violation
   - `Person + No Gloves` $\rightarrow$ Gloves Violation
   - `Fire / Smoke` $\rightarrow$ Fire & Smoke Emergency Alarm
   - `Area Entry` $\rightarrow$ Restricted Zone Entry (using customized SVG polygon boundaries drawn by the operator in the live feed UI)
4. **Instant Alerts**: Supports email alerts with inline attached screenshots, Twilio WhatsApp SMS, browser alarm sirens, and SSE real-time toast popups.
5. **Interactive Controls**: Features a live monitoring SVG zone drawing coordinator, incident inspection modals, and settings config sliders.
6. **Reporting Portal**: Instant compilation of daily/weekly/monthly compliance logs into PDF reports (with embedded screenshots) and multi-sheet Excel workbooks.

---

## Directory Structure

```text
Safety_Hygiene_Monitoring_System/
├── dataset/                    # Dataset repository
│   ├── ppe_dataset/
│   ├── fire_dataset/
│   └── combined_dataset/       # Clean combined dataset
├── models/                     # Weights directory
│   ├── best.pt                 # Active trained weights
│   └── last.pt
├── src/                        # System source code
│   ├── alert_system.py         # Async email, Twilio and SSE pipeline
│   ├── camera.py               # Video capture streams & simulation feed
│   ├── database.py             # SQLAlchemy models & sessions
│   ├── detector.py             # YOLOv11 loading & fallback inferences
│   ├── tracker.py              # centroid tracking & worker indexing
│   ├── rule_engine.py          # PPE & boundary intersection checks
│   ├── report_generator.py     # PDF (ReportLab) & Excel (Pandas) compiler
│   ├── dataset_preparation.py  # VOC converters & synthetic mock generator
│   ├── train_model.py          # YOLOv11 PyTorch training pipeline
│   └── utils.py                # IoU, point-in-polygon math & loggers
├── templates/                  # Flask HTML UI views
├── static/                     # CSS, JS, and safety audio assets
│   ├── css/
│   │   └── style.css           # Premium Slate dark theme stylesheet
│   └── audio/
│       └── alarm.wav           # Programmatically compiled alarm sound
├── reports/                    # Compiled PDF & Excel exports
├── static/screenshots/         # Captured incident screenshots
├── app.py                      # Primary Web server entrypoint
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configurations
└── README.md
```

---

## Local Development Installation

### 1. Prerequisites
Ensure you have **Python 3.12+** and **pip** installed.

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Package Dependencies
```bash
pip install -r requirements.txt
```
*Note: The first time you load the YOLO detector, `ultralytics` will download the small `yolo11n.pt` base model. Please ensure you have an active internet connection on initialization.*

---

## YOLOv11 Training Pipeline

If you do not have an existing dataset, the preparation script can compile a **synthetic training set** (colored polygons with box annotations) to let you run the training script:

### 1. Generate Synthetic Dataset
```bash
python -m src.dataset_preparation
```
This structures `dataset/combined_dataset` with training/validation splits and generates `data.yaml`.

### 2. Run Model Training
```bash
python -m src.train_model
```
This loads YOLOv11, trains for 1 verification epoch (highly throttle-optimized for quick validation), copies weights to `models/best.pt`, and exports performance charts (precision, recall, loss). 
*(To train on a real dataset, configure the epochs parameter in `src/train_model.py` to 100 epochs).*

---

## Database Configuration

By default, the application runs on **SQLite** (`database/safety_monitor.db`).

### Production MySQL Connection
To switch the storage backend to a production **MySQL** server:
1. Open [config.py](file:///c:/Users/acer/Desktop/Safety_Hygiene_Monitoring_System/config.py).
2. Override `SQLALCHEMY_DATABASE_URI` or declare an environment variable:
   ```bash
   set DATABASE_URL=mysql+pymysql://db_user:db_password@localhost:3306/safety_monitor_db
   ```
On start, the app automatically checks the connection, constructs all tables, and creates a default administrator:
- **Username**: `admin`
- **Password**: `admin123`

---

## Running the Web System

To boot the Flask dashboard server locally:
```bash
python app.py
```
Open a browser and navigate to: **`http://localhost:5000`**

### Simulated Live Feed (Webcam Alternative)
If you don't have a live RTSP camera or webcam attached:
1. Go to the **Live Monitoring** tab in the sidebar.
2. The dropdown selector defaults to **Simulated Workspace Feed**.
3. This boots a virtual camera generator rendering a factory floor with workers walking, entering restricted zones, and triggering fire/smoke alarms.
4. Try clicking on the stream viewport to draw a **Restricted Zone** polygon, and click **Save Restricted Zone** to verify real-time violations.

---

## Docker Container Deployment

To boot the complete application isolated inside Docker containers:
```bash
# Build and run containers
docker-compose up --build -d

# Verify container is running
docker ps
```
The server will bind to port **`5000`** on the host. SQLite databases, captured screenshots, generated reports, and app logfiles will persist inside local host volumes.
