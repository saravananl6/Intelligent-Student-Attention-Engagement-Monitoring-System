import sqlite3
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

# Project modules
from camera import VideoCamera
from utils import load_model_and_encoders, now_ts, compute_session_stats_from_camera

# Configuration
APP_SECRET = "replace_this_with_a_random_secret"
DB_PATH = "app.db"
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = APP_SECRET
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging
handler = RotatingFileHandler("app_debug.log", maxBytes=2_000_000, backupCount=2)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info("App starting up...")

# Jinja filter for datetime formatting
@app.template_filter('datetime')
def format_datetime(value, fmt='%Y-%m-%d %H:%M:%S'):
    try:
        return datetime.fromtimestamp(float(value)).strftime(fmt)
    except Exception:
        return value

# Load ML model & encoders
try:
    MODEL_PATH = "face_action_ann_model.h5"
    ENCODER_PATH = "label_encoder.pkl"
    SCALER_PATH = "scaler.pkl"
    model, encoder, scaler = load_model_and_encoders(MODEL_PATH, ENCODER_PATH, SCALER_PATH)
    app.logger.info("Model and encoders loaded.")
except Exception as e:
    app.logger.exception("Failed loading model/encoders: %s", e)
    raise

# Global camera instance
camera = None

# Camera stop helper
def stop_camera():
    global camera
    try:
        if camera is None:
            return False
        try:
            if hasattr(camera, "stop"):
                try:
                    camera.stop()
                except Exception:
                    pass
            if hasattr(camera, "release"):
                try:
                    camera.release()
                except Exception:
                    pass
            if hasattr(camera, "running"):
                try:
                    camera.running = False
                except Exception:
                    pass
        except Exception as inner:
            app.logger.exception("Error while releasing camera: %s", inner)
        camera = None
        app.logger.info("Camera stopped and released.")
        return True
    except Exception as e:
        app.logger.exception("stop_camera() failed: %s", e)
        return False

# Database helpers
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    
    # Users table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        is_active INTEGER DEFAULT 1
    )''')
    
    # Subjects table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )''')
    
    # Lessons table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        teacher_id INTEGER,
        title TEXT NOT NULL,
        content TEXT,
        file_path TEXT,
        created_at REAL,
        FOREIGN KEY(subject_id) REFERENCES subjects(id),
        FOREIGN KEY(teacher_id) REFERENCES users(id)
    )''')
    
    # Reports table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        lesson_id INTEGER,
        started_at REAL,
        ended_at REAL,
        counts_json TEXT,
        avg_conf REAL,
        overall_distr REAL,
        overall_conc REAL,
        timeline_json TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(lesson_id) REFERENCES lessons(id)
    )''')
    
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# User management functions
def create_user(username, password, role):
    db = get_db()
    cur = db.cursor()
    pw_hash = generate_password_hash(password)
    try:
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                   (username, pw_hash, role))
        db.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None

def check_user(username, password):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,))
    row = cur.fetchone()
    if row and check_password_hash(row['password_hash'], password):
        return dict(row)
    return None

def get_user_by_id(uid):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = cur.fetchone()
    return dict(row) if row else None

# Subject management functions
def create_subject(name):
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO subjects (name) VALUES (?)", (name,))
        db.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None

def get_subjects():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM subjects ORDER BY name")
    return [dict(row) for row in cur.fetchall()]

# Lesson management functions
def create_lesson(subject_id, teacher_id, title, content=None, file_path=None):
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute('''
            INSERT INTO lessons (subject_id, teacher_id, title, content, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (subject_id, teacher_id, title, content, file_path, now_ts()))
        db.commit()
        return cur.lastrowid
    except Exception as e:
        app.logger.exception("Error creating lesson: %s", e)
        return None

def get_lessons_by_subject(subject_id):
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT l.*, u.username as teacher_name, s.name as subject_name
        FROM lessons l
        JOIN users u ON l.teacher_id = u.id
        JOIN subjects s ON l.subject_id = s.id
        WHERE l.subject_id = ?
        ORDER BY l.created_at DESC
    ''', (subject_id,))
    return [dict(row) for row in cur.fetchall()]

def get_lesson_by_id(lesson_id):
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT l.*, u.username as teacher_name, s.name as subject_name
        FROM lessons l
        JOIN users u ON l.teacher_id = u.id
        JOIN subjects s ON l.subject_id = s.id
        WHERE l.id = ?
    ''', (lesson_id,))
    row = cur.fetchone()
    if row:
        lesson = dict(row)
        # If it's a text file, read its content
        if lesson['file_path'] and lesson['file_path'].endswith('.txt'):
            try:
                file_path = os.path.join(app.root_path, lesson['file_path'])
                with open(file_path, 'r', encoding='utf-8') as file:
                    lesson['file_content'] = file.read()
            except Exception as e:
                app.logger.exception("Error reading text file: %s", e)
                lesson['file_content'] = "Error reading file content."
        return lesson
    return None

# Report functions
def save_report(user_id, lesson_id, stats):
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute('''
            INSERT INTO reports (user_id, lesson_id, started_at, ended_at, counts_json, 
                                avg_conf, overall_distr, overall_conc, timeline_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, lesson_id, 
            stats['started_at'], stats['ended_at'],
            json.dumps(stats['counts']),
            stats['avg_conf'],
            stats['overall_distr'],
            stats['overall_conc'],
            json.dumps(stats['timeline'])
        ))
        db.commit()
        return cur.lastrowid
    except Exception as e:
        app.logger.exception("Error saving report: %s", e)
        return None

def get_reports_by_user(user_id):
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT r.*, l.title as lesson_title, s.name as subject_name
        FROM reports r
        JOIN lessons l ON r.lesson_id = l.id
        JOIN subjects s ON l.subject_id = s.id
        WHERE r.user_id = ?
        ORDER BY r.id DESC
    ''', (user_id,))
    reports = []
    for r in cur.fetchall():
        rec = dict(r)
        try:
            rec['counts'] = json.loads(rec.get('counts_json') or "{}")
        except Exception:
            rec['counts'] = {}
        try:
            rec['timeline'] = json.loads(rec.get('timeline_json') or "[]")
        except Exception:
            rec['timeline'] = []
        reports.append(rec)
    return reports

def get_all_reports():
    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT r.*, u.username as student_name, l.title as lesson_title, s.name as subject_name
        FROM reports r
        JOIN users u ON r.user_id = u.id
        JOIN lessons l ON r.lesson_id = l.id
        JOIN subjects s ON l.subject_id = s.id
        ORDER BY r.id DESC
    ''')
    reports = []
    for r in cur.fetchall():
        rec = dict(r)
        try:
            rec['counts'] = json.loads(rec.get('counts_json') or "{}")
        except Exception:
            rec['counts'] = {}
        try:
            rec['timeline'] = json.loads(rec.get('timeline_json') or "[]")
        except Exception:
            rec['timeline'] = []
        reports.append(rec)
    return reports

# Admin functions
def get_users_by_role(role):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE role=? ORDER BY username", (role,))
    return [dict(row) for row in cur.fetchall()]

def deactivate_user(user_id):
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
        db.commit()
        return True
    except Exception as e:
        app.logger.exception("Error deactivating user: %s", e)
        return False

# File upload helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database and default data
with app.app_context():
    init_db()
    db = get_db()
    cur = db.cursor()
    
    # Create default admin
    cur.execute("SELECT * FROM users WHERE username = 'admin'")
    if cur.fetchone() is None:
        create_user('admin', 'admin123', 'admin')
        app.logger.info("Default admin created (username=admin). Change password immediately.")
    
    # Create default subjects
    default_subjects = ['Machine Learning', 'Deep Learning', 'AI']
    for subject in default_subjects:
        cur.execute("SELECT * FROM subjects WHERE name = ?", (subject,))
        if cur.fetchone() is None:
            create_subject(subject)
            app.logger.info(f"Default subject '{subject}' created.")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form['role']
        
        if not username or not password:
            return render_template('signup.html', error="Provide username & password")
        
        uid = create_user(username, password, role)
        if uid:
            return redirect(url_for('login'))
        else:
            return render_template('signup.html', error="Username already exists")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = check_user(username, password)
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            
            if user['role'] == 'student':
                return redirect(url_for('student_dashboard'))
            elif user['role'] == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            elif user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    stop_camera()
    session.clear()
    return redirect(url_for('home'))

# Student routes
@app.route('/student/dashboard')
def student_dashboard():
    if 'user_id' not in session or session.get('role') != 'student':
        return redirect(url_for('login'))
    
    subjects = get_subjects()
    return render_template('student/dashboard.html', subjects=subjects)

@app.route('/student/monitor', methods=['GET', 'POST'])
def student_monitor():
    if 'user_id' not in session or session.get('role') != 'student':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        lesson_id = request.form.get('lesson_id')
        
        if not subject_id or not lesson_id:
            return redirect(url_for('student_dashboard'))
        
        lesson = get_lesson_by_id(lesson_id)
        if not lesson:
            return redirect(url_for('student_dashboard'))
        
        return render_template('student/monitor.html', lesson=lesson)
    
    subject_id = request.args.get('subject_id')
    if not subject_id:
        return redirect(url_for('student_dashboard'))
    
    lessons = get_lessons_by_subject(subject_id)
    return render_template('student/monitor.html', lessons=lessons, subject_id=subject_id)

@app.route('/student/report/<int:report_id>')
def student_report(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM reports WHERE id=?", (report_id,))
    row = cur.fetchone()
    
    if not row:
        return "Report not found", 404
    
    report = dict(row)
    try:
        report['counts'] = json.loads(report.get('counts_json') or "{}")
    except Exception:
        report['counts'] = {}
    try:
        report['timeline'] = json.loads(report.get('timeline_json') or "[]")
    except Exception:
        report['timeline'] = []
    
    # Check if the user is authorized to view this report
    user_id = session.get('user_id')
    user_role = session.get('role')
    
    # Allow access if:
    # 1. User is the student who owns the report, OR
    # 2. User is a teacher, OR
    # 3. User is an admin
    if user_role == 'student' and user_id != report['user_id']:
        return "Forbidden", 403
    
    # Generate recommendation based on distraction level
    od = float(report.get('overall_distr') or 0.0)
    if od < 10:
        rec = "Great focus — keep this study habit. Short breaks (~5 min) every 50 minutes recommended."
    elif od < 30:
        rec = "Moderate distractions detected. Remove background noises, close unrelated tabs, try Pomodoro (25/5)."
    else:
        rec = "High distraction detected. Consider turning off notifications, move to quiet place, and take a 10–15 minute break."
    
    return render_template('student/report.html', report=report, recommendation=rec)

@app.route('/report/<int:rid>')
def view_report(rid):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM reports WHERE id=?", (rid,))
        row = cur.fetchone()
        if not row:
            return "Report not found", 404
        
        report = dict(row)
        try:
            report['counts'] = json.loads(report.get('counts_json') or "{}")
        except Exception:
            report['counts'] = {}
        try:
            report['timeline'] = json.loads(report.get('timeline_json') or "[]")
        except Exception:
            report['timeline'] = []
        
        uid = session.get('user_id')
        user_role = session.get('role')
        
        # Allow access if:
        # 1. User is the student who owns the report, OR
        # 2. User is a teacher, OR
        # 3. User is an admin
        if (not user_role) or (user_role == 'student' and uid != report['user_id']):
            return "Forbidden", 403
        
        # Generate recommendation based on distraction level
        od = float(report.get('overall_distr') or 0.0)
        if od < 10:
            rec = "Great focus — keep this study habit. Short breaks (~5 min) every 50 minutes recommended."
        elif od < 30:
            rec = "Moderate distractions detected. Remove background noises, close unrelated tabs, try Pomodoro (25/5)."
        else:
            rec = "High distraction detected. Consider turning off notifications, move to quiet place, and take a 10–15 minute break."
        
        return render_template('student/report.html', report=report, recommendation=rec)
    except Exception:
        app.logger.exception("Error viewing report %s", rid)
        return ("An error occurred while loading this report. The error has been logged to app_debug.log."), 500

@app.route('/student/my_reports')
def student_my_reports():
    if 'user_id' not in session or session.get('role') != 'student':
        return redirect(url_for('login'))
    
    reports = get_reports_by_user(session['user_id'])
    return render_template('student/my_reports.html', reports=reports)

# Teacher routes
@app.route('/teacher/dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return redirect(url_for('login'))
    
    subjects = get_subjects()
    return render_template('teacher/dashboard.html', subjects=subjects)

@app.route('/teacher/create_lesson', methods=['GET', 'POST'])
def teacher_create_lesson():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        subject_id = request.form.get('subject_id')
        title = request.form.get('title')
        content = request.form.get('content')
        file = request.files.get('file')
        
        if not subject_id or not title:
            return render_template('teacher/create_lesson.html', 
                                  subjects=get_subjects(), 
                                  error="Subject and title are required")
        
        file_path = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        
        lesson_id = create_lesson(subject_id, session['user_id'], title, content, file_path)
        if lesson_id:
            return redirect(url_for('teacher_dashboard'))
        else:
            return render_template('teacher/create_lesson.html', 
                                  subjects=get_subjects(), 
                                  error="Failed to create lesson")
    
    subjects = get_subjects()
    return render_template('teacher/create_lesson.html', subjects=subjects)

@app.route('/teacher/student_logs')
def teacher_student_logs():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return redirect(url_for('login'))
    
    reports = get_all_reports()
    return render_template('teacher/student_logs.html', reports=reports)

# Admin routes
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    stop_camera()
    return render_template('admin/dashboard.html')

@app.route('/admin/manage_users')
def admin_manage_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    students = get_users_by_role('student')
    teachers = get_users_by_role('teacher')
    return render_template('admin/manage_users.html', students=students, teachers=teachers)

@app.route('/admin/deactivate_user/<int:user_id>', methods=['POST'])
def admin_deactivate_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    if deactivate_user(user_id):
        return redirect(url_for('admin_manage_users'))
    else:
        return "Failed to deactivate user", 500

@app.route('/admin/manage_subjects', methods=['GET', 'POST'])
def admin_manage_subjects():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        subject_name = request.form.get('subject_name')
        if subject_name:
            create_subject(subject_name)
        return redirect(url_for('admin_manage_subjects'))
    
    subjects = get_subjects()
    return render_template('admin/manage_subjects.html', subjects=subjects)

@app.route('/admin/system_status')
def admin_system_status():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    reports = get_all_reports()
    return render_template('admin/system_status.html', reports=reports)

# Camera and monitoring routes
@app.route('/video_feed')
def video_feed():
    global camera
    # If camera not started, create camera
    if camera is None:
        try:
            camera = VideoCamera(model, encoder, scaler)
            app.logger.info("Camera started for streaming.")
        except Exception as e:
            app.logger.exception("Failed to start camera: %s", e)
            return ("Camera could not be started. Check server logs.", 500)

    def gen():
        while True:
            try:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.05)
            except GeneratorExit:
                break
            except Exception:
                app.logger.exception("Error streaming frame")
                break

    return app.response_class(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_level', methods=['POST'])
def audio_level():
    payload = request.get_json() or {}
    rms = float(payload.get('rms', 0.0))
    ts = float(payload.get('timestamp', time.time()))
    uid = session.get('user_id')
    if uid is None:
        return jsonify(success=False)
    if not hasattr(app, 'audio_store'):
        app.audio_store = {}
    if uid not in app.audio_store:
        app.audio_store[uid] = []
    app.audio_store[uid].append((ts, rms))
    app.audio_store[uid] = [(t,r) for (t,r) in app.audio_store[uid] if time.time()-t <= 600]
    return jsonify(success=True)

@app.route('/start_session', methods=['POST'])
def start_session():
    if 'user_id' not in session or session.get('role') != 'student':
        return jsonify(success=False, message="login required")
    
    # Get JSON data instead of form data
    data = request.get_json() or {}
    client_id = data.get('client_id')
    lesson_id = data.get('lesson_id')
    
    if not lesson_id:
        return jsonify(success=False, message="lesson_id required")
    
    uid = session['user_id']
    
    global camera
    try:
        # Stop any existing camera
        if camera is not None:
            stop_camera()
        
        # Create new camera instance
        app.logger.info("Creating new camera instance for user %s", uid)
        camera = VideoCamera(model, encoder, scaler)
        app.logger.info("Camera started successfully for user %s", uid)
    except Exception as e:
        app.logger.exception("Failed to start camera in start_session: %s", e)
        return jsonify(success=False, message=f"camera start failed: {str(e)}")
    
    # Initialize sessions dictionary if it doesn't exist
    if not hasattr(app, 'sessions'):
        app.sessions = {}
    
    # Initialize audio_store if it doesn't exist
    if not hasattr(app, 'audio_store'):
        app.audio_store = {}
    
    # Create session data
    start_time = now_ts()
    app.sessions[uid] = {
        'started_at': start_time, 
        'ended_at': None, 
        'lesson_id': lesson_id,
        'client_id': client_id
    }
    
    # Initialize audio data for this user
    app.audio_store[uid] = []
    
    app.logger.info("Session started successfully for user %s", uid)
    return jsonify(success=True, started_at=start_time)

@app.route('/stop_session', methods=['POST'])
def stop_session():
    if 'user_id' not in session or session.get('role') != 'student':
        return jsonify(success=False, message="login required")
    
    uid = session['user_id']
    if not hasattr(app, 'sessions') or uid not in app.sessions:
        return jsonify(success=False, message="no session running")
    
    # Get JSON data
    data = request.get_json() or {}
    client_id = data.get('client_id')
    
    app.sessions[uid]['ended_at'] = now_ts()
    started_at = app.sessions[uid]['started_at']
    ended_at = app.sessions[uid]['ended_at']
    lesson_id = app.sessions[uid]['lesson_id']
    
    face_actions = []
    try:
        if camera:
            face_actions = camera.get_face_actions()
    except Exception:
        app.logger.exception("Failed reading face actions from camera")
    
    audio_data = app.audio_store.get(uid, []) if hasattr(app, 'audio_store') else []
    
    # Compute stats
    stats = compute_session_stats_from_camera(face_actions, audio_data, started_at, ended_at)
    stats['started_at'] = started_at
    stats['ended_at'] = ended_at
    
    # Save report to DB
    try:
        report_id = save_report(uid, lesson_id, stats)
        app.logger.info("Saved report %s for user %s", report_id, uid)
    except Exception:
        app.logger.exception("Failed saving report to DB")
        report_id = None
    
    # Stop camera
    try:
        stop_camera()
    except Exception:
        app.logger.exception("Error stopping camera in stop_session")
    
    # Clear session
    try:
        del app.sessions[uid]
    except Exception:
        pass
    
    stats['report_id'] = report_id
    stats['saved'] = bool(report_id)
    return jsonify(success=True, **stats)

@app.route('/stop_camera', methods=['POST','GET'])
def stop_camera_route():
    ok = stop_camera()
    return jsonify(success=ok)

# Run app
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        try:
            stop_camera()
        except Exception:
            pass