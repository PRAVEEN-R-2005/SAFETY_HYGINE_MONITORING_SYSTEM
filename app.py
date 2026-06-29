import os
import json
import datetime
import time
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import bcrypt

from config import Config
from src.database import init_db, SessionLocal, User, Camera, Violation, Report, Setting, DetectionLog, AlertLog
from src.camera import CameraStream
from src.detector import Detector
from src.tracker import Tracker
from src.rule_engine import RuleEngine
from src.alert_system import AlertSystem, ui_alert_queue
from src.report_generator import ReportGenerator
from src.utils import logger, draw_restricted_area

# Initialize Database
init_db()

app = Flask(__name__)
app.config.from_object(Config)

# Auth Configuration
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        return db.get(User, int(user_id))
    finally:
        db.close()

# Shared Global ML Components
detector = Detector()
rule_engine = RuleEngine()
trackers = {} # Format: {camera_id: TrackerInstance}

# Pool of active streams to avoid resource leaks
active_streams = {}

def get_camera_stream(camera_id, stream_url):
    """Reuse existing stream handle or spin up a new one"""
    if camera_id in active_streams:
        # Check if running, if not restart
        if not active_streams[camera_id].running:
            active_streams[camera_id].start()
        return active_streams[camera_id]
        
    stream = CameraStream(stream_url)
    stream.start()
    active_streams[camera_id] = stream
    return stream

# Context processor to inject dynamic settings
@app.context_processor
def inject_settings():
    db = SessionLocal()
    try:
        sound = db.query(Setting).filter_by(key='sound_alerts_enabled').first()
        popup = db.query(Setting).filter_by(key='popup_alerts_enabled').first()
        
        return {
            'system_settings': {
                'sound_alerts_enabled': sound.value == 'true' if sound else True,
                'popup_alerts_enabled': popup.value == 'true' if popup else True
            }
        }
    finally:
        db.close()

# ================= AUTHENTICATION ROUTES =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                login_user(user)
                flash('Successfully logged in!')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('An error occurred during authentication.')
        finally:
            db.close()
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ================= DASHBOARD & ANALYTICS =================

@app.route('/')
@login_required
def dashboard():
    db = SessionLocal()
    try:
        # Stats counts
        total_cameras = db.query(Camera).filter(Camera.status == 'active').count()
        violations_today = db.query(Violation).filter(
            Violation.timestamp >= datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        ).count()
        
        # Pull workers and compliance index from last logs
        last_logs = db.query(DetectionLog).order_by(DetectionLog.timestamp.desc()).limit(10).all()
        worker_count = 0
        compliance_rate = 100.0
        if last_logs:
            worker_count = last_logs[0].worker_count
            compliance_rate = sum([log.compliance_percentage for log in last_logs]) / len(last_logs)

        # Check fire/smoke statuses in last 30 minutes
        recent_cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)
        fire_alert = db.query(Violation).filter(
            Violation.violation_type == 'Fire Alert',
            Violation.timestamp >= recent_cutoff
        ).count() > 0
        
        smoke_alert = db.query(Violation).filter(
            Violation.violation_type == 'Smoke Alert',
            Violation.timestamp >= recent_cutoff
        ).count() > 0

        # Recent logs list
        recent_violations = db.query(Violation).order_by(Violation.timestamp.desc()).limit(5).all()

        stats = {
            'worker_count': worker_count,
            'compliance_rate': compliance_rate,
            'violations_today': violations_today,
            'active_cameras': total_cameras,
            'fire_alert': fire_alert,
            'smoke_alert': smoke_alert
        }

        # Chart Data Builder (Last 7 Days)
        weekly_labels = []
        weekly_violations = []
        compliance_trends = []
        for i in range(6, -1, -1):
            day = datetime.date.today() - datetime.timedelta(days=i)
            day_start = datetime.datetime.combine(day, datetime.time.min)
            day_end = datetime.datetime.combine(day, datetime.time.max)
            
            weekly_labels.append(day.strftime('%A'))
            
            # Violations count
            v_count = db.query(Violation).filter(Violation.timestamp >= day_start, Violation.timestamp <= day_end).count()
            weekly_violations.append(v_count)
            
            # Compliance average
            day_logs = db.query(DetectionLog).filter(DetectionLog.timestamp >= day_start, DetectionLog.timestamp <= day_end).all()
            day_comp = 100.0
            if day_logs:
                day_comp = sum([l.compliance_percentage for l in day_logs]) / len(day_logs)
            compliance_trends.append(day_comp)

        # Violation Categories Doughnut Chart Data
        category_labels = ["No Helmet", "No Vest", "No Gloves", "Unauthorized Access", "Fire Alert", "Smoke Alert"]
        category_counts = []
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        for cat in category_labels:
            c = db.query(Violation).filter(
                Violation.violation_type == cat,
                Violation.timestamp >= today_start
            ).count()
            category_counts.append(c)

        chart_data = {
            'weekly_labels': weeklyLabelsConvert(weekly_labels),
            'weekly_violations': weekly_violations,
            'compliance_trends': complianceTrendsConvert(compliance_trends),
            'category_labels': category_labels,
            'category_counts': category_counts
        }

        return render_template(
            'dashboard.html', 
            active_page='dashboard', 
            stats=stats, 
            recent_violations=recent_violations,
            chart_data=chart_data
        )
    finally:
        db.close()

@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    db = SessionLocal()
    try:
        last_logs = db.query(DetectionLog).order_by(DetectionLog.timestamp.desc()).limit(5).all()
        worker_count = 0
        compliance_rate = 100.0
        if last_logs:
            worker_count = last_logs[0].worker_count
            compliance_rate = sum([log.compliance_percentage for log in last_logs]) / len(last_logs)
            
        violations_today = db.query(Violation).filter(
            Violation.timestamp >= datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        ).count()

        return jsonify({
            'worker_count': worker_count,
            'compliance_rate': compliance_rate,
            'violations_today': violations_today
        })
    finally:
        db.close()


# ================= CAMERA VIEWER & STREAMS =================

@app.route('/live-feed')
@login_required
def live_feed():
    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.status == 'active').all()
        return render_template('live_feed.html', active_page='live_feed', cameras=cameras)
    finally:
        db.close()

def video_stream_generator(camera_id):
    """
    Video frames rendering loop.
    Extracts frame -> runs ML detection -> tracks IDs -> runs rule engines
    -> saves violations -> yields JPEG output.
    """
    db = SessionLocal()
    cam = None
    stream_url = 'simulation'
    polygon = []
    
    try:
        if camera_id != 'simulation':
            cam = db.get(Camera, int(camera_id))
            if cam:
                stream_url = cam.stream_url
                polygon = cam.get_polygon()
                # Update last active time
                cam.last_active = datetime.datetime.utcnow()
                db.commit()
    except Exception as e:
        logger.error(f"Error fetching camera metadata: {e}")
        
    stream = get_camera_stream(camera_id, stream_url)
    
    # Initialize camera tracker if not existing
    if camera_id not in trackers:
        trackers[camera_id] = Tracker()
    tracker = trackers[camera_id]
    
    alert_system = AlertSystem(db_session=db)
    
    frame_count = 0
    
    try:
        while stream.running:
            success, frame = stream.read_frame()
            if not success or frame is None:
                # yield a grey frame as standby
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, "CONNECTING TO CAMERA STREAM...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', error_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(1.0)
                continue
                
            frame_count += 1
            
            # Execute AI detection every 3 frames (reduces CPU load)
            if frame_count % 3 == 0:
                # Perform detections
                detections = detector.detect(frame)
                
                # Separate Person boxes for tracker
                persons_only = [d for d in detections if d['class_id'] == 0]
                tracked_persons = tracker.update(persons_only)
                
                # Combine tracked persons back with other detections
                non_persons = [d for d in detections if d['class_id'] != 0]
                full_tracked_detections = tracked_persons + non_persons
                
                # Apply rules
                violations, tracked_workers = rule_engine.process_detections(full_tracked_detections, camera_id, polygon)
                
                # Calculate metrics for Logging
                worker_count = len(tracked_persons)
                non_compliant_count = len(set([v['track_id'] for v in violations if v['track_id'] != -1]))
                compliance_pct = 100.0
                if worker_count > 0:
                    compliance_pct = ((worker_count - non_compliant_count) / worker_count) * 100.0
                    
                # Write to database (DetectionLog) every 30 frames to avoid database congestion
                if frame_count % 30 == 0 and camera_id != 'simulation':
                    try:
                        ppe_counts = {
                            'helmets': len([d for d in detections if d['class_id'] == 1]),
                            'vests': len([d for d in detections if d['class_id'] == 2]),
                            'gloves': len([d for d in detections if d['class_id'] == 3])
                        }
                        log = DetectionLog(
                            camera_id=int(camera_id),
                            worker_count=worker_count,
                            compliance_percentage=compliance_pct,
                            ppe_stats_json=json.dumps(ppe_counts)
                        )
                        db.add(log)
                        db.commit()
                    except Exception as e:
                        db.rollback()
                        logger.error(f"Error saving detection logs to DB: {e}")

                # Process violations (Capture Evidence and Save)
                for v in violations:
                    # Draw visual boxes on temporary frame copy for screenshot evidence
                    screenshot_frame = frame.copy()
                    screenshot_frame = detector.draw_detections(screenshot_frame, full_tracked_detections)
                    screenshot_frame = draw_restricted_area(screenshot_frame, polygon, is_violation=True)
                    
                    t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_filename = f"cam_{camera_id}_{v['violation_type'].replace(' ', '_')}_{t_stamp}.jpg"
                    screenshot_path = os.path.join(Config.SCREENSHOTS_FOLDER, screenshot_filename)
                    
                    cv2.imwrite(screenshot_path, screenshot_frame)
                    
                    # Store incident to DB
                    if camera_id != 'simulation':
                        try:
                            incident = Violation(
                                camera_id=int(camera_id),
                                violation_type=v['violation_type'],
                                confidence=v['confidence'],
                                screenshot_path=screenshot_path,
                                resolved=False
                            )
                            db.add(incident)
                            db.commit()
                            
                            # Log alert event
                            alert_log = AlertLog(
                                violation_id=incident.id,
                                alert_type='Popup',
                                status='sent'
                            )
                            db.add(alert_log)
                            db.commit()
                            
                            # Dispatch alarms & SMS
                            alert_system.dispatch_alerts(incident, screenshot_path)
                        except Exception as e:
                            db.rollback()
                            logger.error(f"Database error writing violation: {e}")
                    else:
                        # For simulation mode, build dummy object to send alerts in UI
                        class MockViolation:
                            def __init__(self, c_id, v_type, conf):
                                self.camera_id = 1
                                self.violation_type = v_type
                                self.confidence = conf
                                self.timestamp = datetime.datetime.utcnow()
                        mock_v = MockViolation(1, v['violation_type'], v['confidence'])
                        alert_system.dispatch_alerts(mock_v, screenshot_path)

            # Render visuals on live stream (Boxes + Restricted area overlays)
            if 'full_tracked_detections' in locals():
                frame = detector.draw_detections(frame, full_tracked_detections)
                
            frame = draw_restricted_area(frame, polygon, is_violation=False)
            
            # Encode output
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03) # 30 FPS throttle
            
    finally:
        db.close()

@app.route('/camera_feed/<camera_id>')
@login_required
def camera_feed(camera_id):
    """Streams camera frames under multipart format"""
    return Response(video_stream_generator(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# ================= VIOLATIONS ENDPOINTS =================

@app.route('/violations')
@login_required
def violations_list():
    db = SessionLocal()
    try:
        cameras = db.query(Camera).all()
        
        # Build query
        query = db.query(Violation)
        
        # Search filter
        search = request.args.get('search')
        if search:
            query = query.filter(Violation.violation_type.like(f"%{search}%"))
            
        # Camera filter
        cam_id = request.args.get('camera_id')
        if cam_id:
            query = query.filter(Violation.camera_id == int(cam_id))
            
        # Type filter
        v_type = request.args.get('type')
        if v_type:
            query = query.filter(Violation.violation_type == v_type)
            
        # Status filter
        resolved = request.args.get('resolved')
        if resolved in ['0', '1']:
            query = query.filter(Violation.resolved == (resolved == '1'))
            
        violations = query.order_by(Violation.timestamp.desc()).all()
        
        return render_template(
            'violations.html', 
            active_page='violations', 
            violations=violations,
            cameras=cameras
        )
    finally:
        db.close()

@app.route('/violation/<int:violation_id>/resolve', methods=['POST'])
@login_required
def resolve_violation(violation_id):
    notes = request.form.get('notes')
    db = SessionLocal()
    try:
        violation = db.get(Violation, violation_id)
        if violation:
            violation.resolved = True
            violation.resolution_notes = notes
            db.commit()
            flash('Incident resolution saved successfully.')
        else:
            flash('Violation not found.')
    except Exception as e:
        db.rollback()
        logger.error(f"Error resolving violation: {e}")
        flash('Failed to resolve violation.')
    finally:
        db.close()
    return redirect(url_for('violations_list'))


# ================= CAMERA CRUD HANDLERS =================

@app.route('/cameras', methods=['GET', 'POST'])
@login_required
def cameras_list():
    db = SessionLocal()
    try:
        if request.method == 'POST':
            name = request.form.get('name')
            stream_url = request.form.get('stream_url')
            department = request.form.get('department')
            status = request.form.get('status', 'active')
            
            # Check validation
            if not name or not stream_url:
                flash('Please specify camera nickname and stream URL.')
            else:
                new_cam = Camera(name=name, stream_url=stream_url, department=department, status=status)
                db.add(new_cam)
                db.commit()
                flash('Camera registered successfully!')
                return redirect(url_for('cameras_list'))
                
        cameras = db.query(Camera).all()
        return render_template('cameras.html', active_page='cameras', cameras=cameras)
    finally:
        db.close()

@app.route('/camera/<int:camera_id>/edit', methods=['POST'])
@login_required
def edit_camera(camera_id):
    db = SessionLocal()
    try:
        cam = db.get(Camera, camera_id)
        if cam:
            cam.name = request.form.get('name')
            cam.stream_url = request.form.get('stream_url')
            cam.department = request.form.get('department')
            cam.status = request.form.get('status')
            db.commit()
            flash('Camera settings updated.')
            
            # Remove from streams cache to trigger reload
            if camera_id in active_streams:
                active_streams[camera_id].stop()
                del active_streams[camera_id]
        else:
            flash('Camera not found.')
    except Exception as e:
        db.rollback()
        logger.error(f"Error editing camera: {e}")
        flash('Failed to modify camera.')
    finally:
        db.close()
    return redirect(url_for('cameras_list'))

@app.route('/camera/<int:camera_id>/delete', methods=['POST'])
@login_required
def delete_camera(camera_id):
    db = SessionLocal()
    try:
        cam = db.get(Camera, camera_id)
        if cam:
            db.delete(cam)
            db.commit()
            flash('Camera registration removed.')
            
            if camera_id in active_streams:
                active_streams[camera_id].stop()
                del active_streams[camera_id]
        else:
            flash('Camera not found.')
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting camera: {e}")
        flash('Could not delete camera.')
    finally:
        db.close()
    return redirect(url_for('cameras_list'))

# Polygon coordinates get/set routes
@app.route('/api/camera/<int:camera_id>/restricted-area', methods=['GET', 'POST'])
@login_required
def camera_restricted_area(camera_id):
    db = SessionLocal()
    try:
        cam = db.get(Camera, camera_id)
        if not cam:
            return jsonify({'status': 'error', 'message': 'Camera not found'}), 404
            
        if request.method == 'POST':
            data = request.json
            coords = data.get('coordinates', [])
            cam.set_polygon(coords)
            db.commit()
            return jsonify({'status': 'success'})
            
        return jsonify(cam.get_polygon())
    finally:
        db.close()


# ================= REPORT LOGS & EXPORTS =================

@app.route('/reports', methods=['GET', 'POST'])
@login_required
def reports_view():
    db = SessionLocal()
    try:
        if request.method == 'POST':
            report_type = request.form.get('report_type')
            start_str = request.form.get('start_date')
            end_str = request.form.get('end_date')
            
            start_date = datetime.datetime.strptime(start_str, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_str, '%Y-%m-%d') + datetime.timedelta(hours=23, minutes=59)
            
            generator = ReportGenerator(db)
            
            try:
                if report_type == 'PDF':
                    file_path = generator.generate_pdf_report(start_date, end_date)
                else:
                    file_path = generator.generate_excel_report(start_date, end_date)
                    
                # Save entry to DB
                new_report = Report(
                    report_type=report_type,
                    start_date=start_date,
                    end_date=end_date,
                    file_path=file_path,
                    status='completed'
                )
                db.add(new_report)
                db.commit()
                flash('Report generated successfully!')
            except Exception as ex:
                logger.error(f"Report generation error: {ex}")
                flash(f'Report compilation failed: {ex}')
                
            return redirect(url_for('reports_view'))
            
        reports = db.query(Report).order_by(Report.created_at.desc()).all()
        return render_template('reports.html', active_page='reports', reports=reports)
    finally:
        db.close()

@app.route('/download-report/<int:report_id>')
@login_required
def download_report(report_id):
    db = SessionLocal()
    try:
        report = db.get(Report, report_id)
        if report and os.path.exists(report.file_path):
            directory = os.path.dirname(report.file_path)
            filename = os.path.basename(report.file_path)
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            flash("Report file not found on disk.")
            return redirect(url_for('reports_view'))
    finally:
        db.close()


# ================= APP CONFIG / SETTINGS =================

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings_view():
    db = SessionLocal()
    try:
        if request.method == 'POST':
            # Save sliders & numbers
            conf = request.form.get('confidence_threshold')
            cooldown = request.form.get('violation_cooldown')
            
            # Save toggles (if checkbox is not checked, it won't be sent)
            sound = 'true' if request.form.get('sound_alerts') else 'false'
            popup = 'true' if request.form.get('popup_alerts') else 'false'
            email = 'true' if request.form.get('email_alerts') else 'false'
            whatsapp = 'true' if request.form.get('whatsapp_alerts') else 'false'
            
            # Save credentials details
            smtp_host = request.form.get('smtp_server')
            smtp_p = request.form.get('smtp_port')
            smtp_u = request.form.get('smtp_user')
            smtp_pwd = request.form.get('smtp_pass')
            email_to = request.form.get('alert_email_to')
            
            twilio_s = request.form.get('twilio_sid')
            twilio_t = request.form.get('twilio_token')
            twilio_f = request.form.get('twilio_from')
            whatsapp_to = request.form.get('whatsapp_to')

            settings_dict = {
                'confidence_threshold': conf,
                'violation_cooldown_seconds': cooldown,
                'sound_alerts_enabled': sound,
                'popup_alerts_enabled': popup,
                'email_alerts_enabled': email,
                'whatsapp_alerts_enabled': whatsapp,
                'smtp_server': smtp_host,
                'smtp_port': smtp_p,
                'smtp_user': smtp_u,
                'smtp_pass': smtp_pwd,
                'alert_email_to': email_to,
                'twilio_sid': twilio_s,
                'twilio_token': twilio_t,
                'twilio_from': twilio_f,
                'whatsapp_to': whatsapp_to
            }

            for key, val in settings_dict.items():
                if val is not None:
                    setting = db.query(Setting).filter_by(key=key).first()
                    if setting:
                        setting.value = str(val)
                    else:
                        db.add(Setting(key=key, value=str(val)))
            db.commit()
            
            # Sync variables to Config class dynamically
            Config.CONFIDENCE_THRESHOLD = float(conf) if conf else Config.CONFIDENCE_THRESHOLD
            Config.ALERT_COOLDOWN_SECONDS = int(cooldown) if cooldown else Config.ALERT_COOLDOWN_SECONDS
            Config.SMTP_SERVER = smtp_host or Config.SMTP_SERVER
            Config.SMTP_PORT = int(smtp_p) if smtp_p else Config.SMTP_PORT
            Config.SMTP_USER = smtp_u or Config.SMTP_USER
            Config.SMTP_PASS = smtp_pwd or Config.SMTP_PASS
            Config.ALERT_EMAIL_TO = email_to or Config.ALERT_EMAIL_TO
            Config.TWILIO_ACCOUNT_SID = twilio_s or Config.TWILIO_ACCOUNT_SID
            Config.TWILIO_AUTH_TOKEN = twilio_t or Config.TWILIO_AUTH_TOKEN
            Config.TWILIO_WHATSAPP_FROM = twilio_f or Config.TWILIO_WHATSAPP_FROM
            Config.ALERT_WHATSAPP_TO = whatsapp_to or Config.ALERT_WHATSAPP_TO
            
            # Re-init detector weights threshold
            if detector.model:
                detector.load_model()
                
            flash("System configurations saved.")
            return redirect(url_for('settings_view'))

        # Fetch settings from DB to display
        rows = db.query(Setting).all()
        s_data = {r.key: r.value for r in rows}
        
        settings = {
            'confidence_threshold': float(s_data.get('confidence_threshold', 0.45)),
            'violation_cooldown_seconds': int(s_data.get('violation_cooldown_seconds', 300)),
            'sound_alerts_enabled': s_data.get('sound_alerts_enabled', 'true') == 'true',
            'popup_alerts_enabled': s_data.get('popup_alerts_enabled', 'true') == 'true',
            'email_alerts_enabled': s_data.get('email_alerts_enabled', 'true') == 'true',
            'whatsapp_alerts_enabled': s_data.get('whatsapp_alerts_enabled', 'false') == 'true',
            
            'smtp_server': s_data.get('smtp_server', ''),
            'smtp_port': s_data.get('smtp_port', '587'),
            'smtp_user': s_data.get('smtp_user', ''),
            'smtp_pass': s_data.get('smtp_pass', ''),
            'alert_email_to': s_data.get('alert_email_to', ''),
            
            'twilio_sid': s_data.get('twilio_sid', ''),
            'twilio_token': s_data.get('twilio_token', ''),
            'twilio_from': s_data.get('twilio_from', 'whatsapp:+14155238886'),
            'whatsapp_to': s_data.get('whatsapp_to', '')
        }

        return render_template('settings.html', active_page='settings', settings=settings)
    finally:
        db.close()


# ================= REAL TIME SSE STREAM =================

@app.route('/api/alerts/stream')
def alerts_stream():
    """
    Server-Sent Events (SSE) channel.
    Yields JSON records from ui_alert_queue. Includes a 2-second timeout
    and keep-alive ticks to prevent Flask connections locks.
    """
    def event_generator():
        while True:
            try:
                # Keep-alive loop checks the queue every 2 seconds
                alert = ui_alert_queue.get(timeout=2.0)
                yield f"data: {json.dumps(alert)}\n\n"
            except queue_timeout_or_empty():
                # Keep-alive ping
                yield ": keep-alive\n\n"
            except Exception as e:
                logger.error(f"Error in SSE alert stream: {e}")
                
    return Response(event_generator(), mimetype='text/event-stream')


# ================= HELPERS & JINJA CONVERTERS =================

def queue_timeout_or_empty():
    import queue
    return queue.Empty

def weeklyLabelsConvert(labels):
    return labels

def complianceTrendsConvert(trends):
    return trends

if __name__ == '__main__':
    # Start Web Server
    app.run(host='0.0.0.0', port=5000, debug=True)
