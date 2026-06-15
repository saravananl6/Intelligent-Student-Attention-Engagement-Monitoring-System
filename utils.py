from tensorflow.keras.models import load_model
import joblib, time

def load_model_and_encoders(model_path, encoder_path, scaler_path):
    model = load_model(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    return model, encoder, scaler

def now_ts():
    return time.time()

def compute_session_stats_from_camera(face_actions, audio_data, start, end):
    # face_actions: list of (ts,label,conf)
    # audio_data: list of (ts,rms)
    face_window = [(ts,label,conf) for (ts,label,conf) in face_actions if ts >= start and ts <= end]
    counts = {'face_tilt':0, 'eye_close':0, 'yawn':0, 'concentration':0, 'unknown':0}
    conf_sum = 0.0
    for ts, label, conf in face_window:
        key = label.lower() if label else 'unknown'
        if key not in counts:
            key = 'unknown'
        counts[key] += 1
        conf_sum += conf
    total_face = len(face_window)
    avg_conf = (conf_sum / total_face) if total_face > 0 else 0.0

    audio_window = [(t,r) for (t,r) in audio_data if t >= start and t <= end]
    audio_total = len(audio_window)
    audio_distr = sum(1 for (t,r) in audio_window if r > 0.02)
    audio_distr_pct = (audio_distr / audio_total * 100) if audio_total > 0 else 0.0

    face_distr_count = counts['face_tilt'] + counts['eye_close'] + counts['yawn'] + counts['unknown']
    face_distr_pct = (face_distr_count / total_face * 100) if total_face > 0 else 0.0
    face_conc_pct = (counts['concentration'] / total_face * 100) if total_face > 0 else 100.0

    overall_distr = (face_distr_pct * 0.7) + (audio_distr_pct * 0.3)
    overall_conc = 100 - overall_distr

    # per-second timeline
    seconds = int(max(1, round(end - start)))
    timeline = []
    for s in range(seconds + 1):
        t0 = start + s; t1 = t0 + 1
        face_seg = [lbl for (ts, lbl, cf) in face_window if ts >= t0 and ts < t1]
        distracted = sum(1 for lbl in face_seg if lbl in ['face_tilt','yawn','eye_close','unknown'])
        concentrated = sum(1 for lbl in face_seg if lbl == 'concentration')
        audio_seg = [r for (ts, r) in audio_window if ts >= t0 and ts < t1]
        audio_flag = 1 if (len([r for r in audio_seg if r > 0.02]) > 0) else 0
        timeline.append({'second': s, 'distracted_count': distracted, 'concentrated_count': concentrated, 'audio_distracted': audio_flag})

    return {
        'counts': counts,
        'avg_conf': avg_conf,
        'face_distr_pct': face_distr_pct,
        'audio_distr_pct': audio_distr_pct,
        'overall_distr': overall_distr,
        'overall_conc': overall_conc,
        'timeline': timeline,
        'session_seconds': seconds
    }