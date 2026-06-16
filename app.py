"""
DeepEmotion AI — Real-Time Voice Emotion Detection and Abnormal Behavior Alert System
Run with: python app.py
Features: Real-time distress detection, emergency risk levels, emotion timeline,
          speech-to-text + sentiment analysis, spectrogram visualization, multi-language support.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import tempfile
import h5py
import numpy as np
import joblib
import librosa
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Model Loading (Keras CNN — trained on 40 MFCCs)
# ─────────────────────────────────────────────
model = None
model_loaded = False
scaler = None

def build_model():
    """Build the exact CNN architecture matching model_trainer.py."""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Activation, Dropout, MaxPooling1D, Flatten, Dense

    model = Sequential([
        Conv1D(128, 5, padding='same', input_shape=(40, 1)),
        Activation('relu'),
        Dropout(0.1),
        MaxPooling1D(pool_size=8, strides=8, padding='valid'),
        Conv1D(128, 5, padding='same'),
        Activation('relu'),
        Dropout(0.1),
        Flatten(),
        Dense(8, activation='softmax')
    ])
    return model

def load_model():
    """Load the Keras CNN model with weights from the .h5 file."""
    global model, model_loaded, scaler
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    model_path = os.path.join('models', 'Emotion_Voice_Detection_Model.h5')
    if not os.path.exists(model_path):
        alt_path = r'C:\Users\rohit\OneDrive\Desktop\EmotionDetection-master\WebApplication\models\Emotion_Voice_Detection_Model.h5'
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        alt_path = r'C:\Users\rohit\OneDrive\Desktop\EmotionDetection-master\Emotion-Detection-from-Audio\model\Emotion_Voice_Detection_Model.h5'
        if os.path.exists(alt_path):
            model_path = alt_path

    if not os.path.exists(model_path):
        print("[MODEL] No trained model found. Demo mode active.")
        model_loaded = False
        return

    try:
        scaler_path = os.path.join('models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)
            print("[MODEL] Scaler loaded")
        else:
            scaler = None

        model = build_model()
        model.compile()
        model.load_weights(model_path)
        print(f"[MODEL] Keras model loaded from {model_path}")
        model_loaded = True

    except Exception as e:
        print(f"[MODEL] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False

# Load immediately — before Flask workers (and the debug reloader) fork
print("[MODEL] Loading model...")
load_model()

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from collections import deque
import io
import base64
import threading
import json

# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'}

# ─────────────────────────────────────────────
# Emotion Metadata (existing + new distress-relevant entries)
# ─────────────────────────────────────────────
EMOTION_META = {
    'neutral':   {'emoji': '😐', 'color': '#94A3B8', 'description': 'No strong emotional tone detected.'},
    'calm':      {'emoji': '😌', 'color': '#67E8F9', 'description': 'Relaxed and composed vocal delivery.'},
    'happy':     {'emoji': '😄', 'color': '#FDE047', 'description': 'Positive energy and joy in the voice.'},
    'sad':       {'emoji': '😢', 'color': '#818CF8', 'description': 'Low energy and melancholic vocal tone.'},
    'angry':     {'emoji': '😠', 'color': '#F87171', 'description': 'High intensity and aggressive vocal patterns.'},
    'fearful':   {'emoji': '😨', 'color': '#C084FC', 'description': 'Tension and anxiety detected in voice.'},
    'disgust':   {'emoji': '🤢', 'color': '#4ADE80', 'description': 'Strong aversion expressed through voice.'},
    'surprised': {'emoji': '😲', 'color': '#FB923C', 'description': 'Sudden shift in vocal energy detected.'},
}

LABEL_CONVERSION = {
    '0': 'neutral', '1': 'calm', '2': 'happy', '3': 'sad',
    '4': 'angry', '5': 'fearful', '6': 'disgust', '7': 'surprised'
}

# ─────────────────────────────────────────────
# Risk Level Configuration
# ─────────────────────────────────────────────
RISK_COLORS = {
    'critical': '#FF1744',
    'high':     '#FF6D00',
    'medium':   '#FFD600',
    'low':      '#00C853',
}

RISK_LABELS = {
    'critical': '🚨 CRITICAL — Immediate Attention Required',
    'high':     '⚠️ HIGH RISK — Potential Emergency Situation',
    'medium':   '🟡 MEDIUM — Elevated Distress Detected',
    'low':      '🟢 LOW — Normal Emotional State',
}

EMERGENCY_CONTACTS = {
    'police':    '100',
    'ambulance': '102',
    'fire':      '101',
}

# ─────────────────────────────────────────────
# Feature Extraction (must match neural_network.py exactly)
# ─────────────────────────────────────────────
def extract_features(file_path, sr=22050, duration=3.0, offset=0.0):
    """
    Extract 40 MFCC features — must match create_features.py exactly:
    res_type='kaiser_fast', offset=0.0, axis=0 mean.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=offset, res_type='kaiser_fast')
        if len(audio) < 1000:
            return None
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        features = mfccs.reshape(40, 1)
        x = np.expand_dims(features, axis=0)
        return x.astype(np.float32)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def get_audio_properties(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        return {'duration': round(duration, 2), 'sample_rate': sr, 'samples': len(audio)}
    except:
        return {'duration': 0, 'sample_rate': 22050, 'samples': 0}


# ─────────────────────────────────────────────
# Distress Engine
# ─────────────────────────────────────────────
class DistressEngine:
    """Computes distress score (0-100) and risk level from emotion probabilities."""

    DISTRESS_WEIGHTS = {
        'fearful': 1.0,    # primary distress indicator
        'angry': 0.7,      # can indicate distress/frustration
        'sad': 0.6,        # emotional pain
        'disgust': 0.4,    # mild distress
        'surprised': 0.2,  # startle response
        'neutral': 0.0,
        'calm': 0.0,
        'happy': 0.0,
    }

    def compute(self, probabilities: dict) -> dict:
        """Returns distress score (0-100), risk level, and breakdown."""
        score = sum(
            self.DISTRESS_WEIGHTS.get(emotion, 0) * (prob / 100)
            for emotion, prob in probabilities.items()
        ) * 100

        if score >= 80:
            level = 'critical'
        elif score >= 60:
            level = 'high'
        elif score >= 40:
            level = 'medium'
        else:
            level = 'low'

        return {
            'distress_score': round(score, 1),
            'risk_level': level,
            'risk_color': RISK_COLORS.get(level, '#00C853'),
            'risk_label': RISK_LABELS.get(level, '🟢 LOW'),
            'alert_triggered': level in ('critical', 'high'),
            'breakdown': {
                emotion: round(self.DISTRESS_WEIGHTS.get(emotion, 0) * (prob / 100) * 100, 1)
                for emotion, prob in probabilities.items()
            }
        }

    def detect_escalation(self, timeline: list) -> dict:
        """Detect if emotions are escalating toward distress."""
        if len(timeline) < 3:
            return {'escalating': False, 'trend': 'stable', 'velocity': 'none'}

        recent = timeline[-5:]
        scores = [self.compute(t.get('probabilities', {}))['distress_score'] for t in recent]

        # Check for consistent upward trend
        increasing = sum(1 for i in range(len(scores)-1) if scores[i+1] > scores[i])
        if increasing >= len(scores) - 1 and scores[-1] > scores[0] + 15:
            return {'escalating': True, 'trend': 'rising', 'velocity': 'accelerating'}
        elif scores[-1] > scores[0] + 10:
            return {'escalating': True, 'trend': 'rising', 'velocity': 'gradual'}
        elif scores[-1] < scores[0] - 10:
            return {'escalating': False, 'trend': 'decreasing', 'velocity': 'normal'}
        return {'escalating': False, 'trend': 'stable', 'velocity': 'none'}


distress_engine = DistressEngine()

# ─────────────────────────────────────────────
# Spectrogram Generator
# ─────────────────────────────────────────────
class SpectrogramGenerator:
    """Generates waveform and mel spectrogram images from audio."""

    def generate_spectrogram(self, audio_path: str) -> str:
        """Generate combined waveform + mel spectrogram as base64 PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import librosa.display

            y, sr = librosa.load(audio_path, sr=22050)

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            # Waveform
            librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#6c63ff')
            axes[0].set_title('Waveform', fontsize=12, color='#e8eaf0')
            axes[0].set_facecolor('#111218')
            axes[0].tick_params(colors='#e8eaf0')
            for spine in axes[0].spines.values():
                spine.set_color('#1e2130')

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                     ax=axes[1], cmap='viridis')
            axes[1].set_title('Mel Spectrogram', fontsize=12, color='#e8eaf0')
            axes[1].tick_params(colors='#e8eaf0')
            for spine in axes[1].spines.values():
                spine.set_color('#1e2130')

            fig.patch.set_facecolor('#111218')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='#111218')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return img_base64
        except Exception as e:
            print(f"Spectrogram error: {e}")
            return None

    def generate_waveform(self, audio_path: str) -> str:
        """Generate waveform as base64 PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import librosa.display

            y, sr = librosa.load(audio_path, sr=22050)

            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#6c63ff')
            ax.set_facecolor('#111218')
            ax.set_title('Waveform', fontsize=12, color='#e8eaf0')
            fig.patch.set_facecolor('#111218')
            ax.tick_params(colors='#e8eaf0')
            for spine in ax.spines.values():
                spine.set_color('#1e2130')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=80, facecolor='#111218')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return img_base64
        except Exception as e:
            print(f"Waveform error: {e}")
            return None


spectrogram_generator = SpectrogramGenerator()

# ─────────────────────────────────────────────
# Speech-to-Text Engine
# ─────────────────────────────────────────────
class SpeechToTextEngine:
    """Handles speech-to-text with Whisper + multilingual sentiment analysis."""

    def __init__(self):
        self.model = None
        self.sentiment_analyzer = None
        self._loading = False
        self._load_done = False
        self._load_error = None

    def _ensure_loaded(self):
        """Load models on first use (not at import time) to avoid blocking Flask startup."""
        if self._load_done:
            return
        if self._loading:
            return
        self._loading = True
        self._load_error = None

        # Load Whisper
        try:
            import whisper as _w
            self.model = _w.load_model("small")
            print("[STT] Whisper small model loaded")
        except Exception as e:
            self._load_error = str(e)
            print(f"[STT] Could not load Whisper: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

        # Load Sentiment
        try:
            from transformers import pipeline as _pipe, AutoModelForSequenceClassification, AutoTokenizer
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.sentiment_analyzer = _pipe(
                "sentiment-analysis",
                model=model_name,
                device="cuda" if False else "cpu"
            )
            print("[STT] Sentiment analyzer loaded")
        except Exception as e:
            print(f"[STT] Could not load sentiment analyzer: {e}")
            import traceback
            traceback.print_exc()
            self.sentiment_analyzer = None

        self._load_done = True
        self._loading = False

    def get_status(self) -> dict:
        """Return loading status for diagnostics."""
        return {
            'whisper_loaded': self.model is not None,
            'sentiment_loaded': self.sentiment_analyzer is not None,
            'load_error': self._load_error,
            'load_done': self._load_done,
        }

    def transcribe(self, audio_path: str, language: str = 'en') -> dict:
        """Transcribe audio and analyze sentiment."""
        self._ensure_loaded()
        if not self.model:
            return {'text': '', 'language': language, 'error': 'STT model not loaded',
                    'sentiment': {'label': 'neutral', 'score': 0.5},
                    'hidden_distress': {'detected': False, 'confidence': 0, 'indicators': []}}

        try:
            # Load audio using librosa (avoids ffmpeg dependency), then convert to mel
            audio, sr = librosa.load(audio_path, sr=16000, duration=30.0, res_type='kaiser_fast')
            audio = audio.astype(np.float32)
            import whisper as _whisper
            mel = _whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels)
            # Pad if needed
            n_frames = mel.shape[-1]
            if n_frames < self.model.dims.n_audio_ctx:
                mel = _whisper.pad_or_truncate(mel, self.model.dims.n_audio_ctx)
            result = self.model.decode(mel, language=language if language not in ('mixed', 'auto') else 'en', task='transcribe')
            text = result.text.strip()

            sentiment = self._analyze_sentiment(text)
            hidden_distress = self._detect_hidden_distress(text, sentiment)

            return {
                'text': text,
                'language': result.get('language', language),
                'sentiment': sentiment,
                'hidden_distress': hidden_distress,
                'segments': result.get('segments', []),
            }
        except Exception as e:
            return {'text': '', 'language': language, 'error': str(e),
                    'sentiment': {'label': 'neutral', 'score': 0.5},
                    'hidden_distress': {'detected': False, 'confidence': 0, 'indicators': []}}

    def _analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of transcribed text."""
        if not text.strip() or not self.sentiment_analyzer:
            return {'label': 'neutral', 'score': 0.5}

        try:
            # Normalize Hinglish before sentiment analysis
            normalized = self._normalize_hinglish(text)
            result = self.sentiment_analyzer(normalized[:512])[0]
            return {
                'label': result['label'],  # positive, negative, neutral
                'score': round(float(result['score']), 3)
            }
        except Exception as e:
            print(f"Sentiment error: {e}")
            return {'label': 'neutral', 'score': 0.5}

    def _normalize_hinglish(self, text: str) -> str:
        """Normalize common Hinglish distress phrases."""
        replacements = {
            'bachao': 'help', 'bhoot': 'ghost fear', 'dar': 'fear',
            'mar': 'kill', 'gaali': 'abuse', 'sadak': 'road',
            'chalo': 'lets go', 'bhago': 'run flee',
        }
        lower = text.lower()
        for word, replacement in replacements.items():
            if word in lower:
                text += ' ' + replacement
        return text

    def _detect_hidden_distress(self, text: str, sentiment: dict) -> dict:
        """Detect hidden distress from speech content."""
        if not text.strip():
            return {'detected': False, 'confidence': 0, 'indicators': []}

        text_lower = text.lower()
        distress_keywords = {
            'emergency': ['help', 'save me', 'please', 'emergency', 'accident', 'ambulance'],
            'fear': ['scared', 'afraid', 'frightened', 'terrified', 'bhoot', 'dar', 'ghost', 'fraid'],
            'danger': ['thief', 'robber', 'attacking', 'chasing', 'police', 'kill', 'murder'],
            'pain': ['hurt', 'pain', 'injured', 'bleeding', 'hospital', 'doctor', 'wound'],
            'panic': ['run', 'escape', 'fast', 'quick', 'now', 'fast', 'urgent'],
        }

        indicators = []
        for category, keywords in distress_keywords.items():
            if any(kw in text_lower for kw in keywords):
                indicators.append(category)

        hidden = len(indicators) > 0 or sentiment['label'] == 'negative'
        return {
            'detected': hidden,
            'confidence': round(min(0.9, 0.3 + 0.15 * len(indicators)), 2),
            'indicators': indicators,
            'sentiment_triggered': sentiment['label'] == 'negative'
        }


stt_engine = SpeechToTextEngine()

# ─────────────────────────────────────────────
# Session Tracker
# ─────────────────────────────────────────────
class SessionTracker:
    """Tracks emotion history and session statistics."""

    def __init__(self):
        self.history = []
        self.start_time = datetime.now()
        self._lock = threading.Lock()

    def add(self, prediction: dict):
        with self._lock:
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': prediction.get('emotion', 'neutral'),
                'confidence': prediction.get('confidence', 0),
                'distress_score': prediction.get('distress_score', 0),
                'risk_level': prediction.get('risk_level', 'low'),
                'probabilities': prediction.get('probabilities', {}),
            })

    def get_summary(self) -> dict:
        with self._lock:
            if not self.history:
                return {
                    'total_predictions': 0,
                    'session_duration_seconds': 0,
                    'dominant_emotion': 'none',
                    'avg_confidence': 0,
                    'max_distress_score': 0,
                    'current_risk_level': 'low',
                    'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                    'emotion_distribution': {},
                    'timeline': [],
                    'escalation': {'escalating': False, 'trend': 'stable', 'velocity': 'none'},
                }

            emotions = [h['emotion'] for h in self.history]
            distress_scores = [h['distress_score'] for h in self.history]

            risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            emotion_counts = {}
            for h in self.history:
                risk_counts[h['risk_level']] = risk_counts.get(h['risk_level'], 0) + 1
                emotion_counts[h['emotion']] = emotion_counts.get(h['emotion'], 0) + 1

            escalation = distress_engine.detect_escalation(self.history)

            return {
                'total_predictions': len(self.history),
                'session_duration_seconds': (datetime.now() - self.start_time).seconds,
                'dominant_emotion': max(set(emotions), key=emotions.count) if emotions else 'none',
                'avg_confidence': round(sum(p['confidence'] for p in self.history) / len(self.history), 1),
                'max_distress_score': max(distress_scores) if distress_scores else 0,
                'current_risk_level': self.history[-1]['risk_level'] if self.history else 'low',
                'risk_distribution': risk_counts,
                'emotion_distribution': emotion_counts,
                'timeline': self.history[-60:],  # last 60 predictions
                'escalation': escalation,
            }

    def get_history(self, limit: int = 100) -> list:
        with self._lock:
            return self.history[-limit:]

    def clear(self):
        with self._lock:
            self.history = []
            self.start_time = datetime.now()


session_tracker = SessionTracker()

# ─────────────────────────────────────────────
# Emergency Alert Manager
# ─────────────────────────────────────────────
class EmergencyAlertManager:
    """Logs and manages emergency alerts."""

    def __init__(self):
        self.alerts = []
        self._lock = threading.Lock()

    def trigger(self, data: dict) -> dict:
        with self._lock:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'risk_level': data.get('risk_level', 'low'),
                'distress_score': data.get('distress_score', 0),
                'emotion': data.get('emotion', 'unknown'),
                'context': data.get('context', {}),
            }
            self.alerts.append(alert)
            print(f"[ALERT] Emergency triggered: {alert['risk_level'].upper()} — Score: {alert['distress_score']}")
            return alert

    def get_recent(self, limit: int = 10) -> list:
        with self._lock:
            return self.alerts[-limit:]

    def clear(self):
        with self._lock:
            self.alerts = []


alert_manager = EmergencyAlertManager()

# ─────────────────────────────────────────────
# Prediction Helper (enhanced)
# ─────────────────────────────────────────────
def predict_emotion(file_path):
    global model, model_loaded, scaler

    features = extract_features(file_path)
    if features is None:
        return None

    if model_loaded and model is not None:
        try:
            # Apply scaler if available (needed for new model trained with StandardScaler)
            if scaler is not None:
                flat = features.reshape(1, -1)
                scaled = scaler.transform(flat)
                features_scaled = scaled.reshape(1, 40, 1).astype(np.float32)
                predictions = model.predict(features_scaled, verbose=0)
            else:
                predictions = model.predict(features, verbose=0)

            pred_class = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_class])
            emotion = LABEL_CONVERSION.get(str(pred_class), 'neutral')
            emotions = list(LABEL_CONVERSION.values())
            prob_map = {emotions[i]: round(float(predictions[0][i]) * 100, 1) for i in range(len(emotions))}
            top3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    else:
        emotions = list(EMOTION_META.keys())
        np.random.seed(int(time.time()) % 1000)
        probs = np.random.dirichlet(np.ones(len(emotions)) * 0.5)
        emotion_idx = np.argmax(probs)
        emotion = emotions[emotion_idx]
        confidence = float(probs[emotion_idx])
        prob_map = {e: round(float(p) * 100, 1) for e, p in zip(emotions, probs)}
        top3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]

    meta = EMOTION_META.get(emotion, {})

    # Compute distress
    distress = distress_engine.compute(prob_map)

    result = {
        'emotion': emotion,
        'confidence': round(confidence * 100, 1),
        'emoji': meta.get('emoji', '🎤'),
        'color': meta.get('color', '#ffffff'),
        'description': meta.get('description', ''),
        'probabilities': prob_map,
        'top3': top3,
        'demo_mode': not model_loaded,
        # Distress fields
        'distress_score': distress['distress_score'],
        'risk_level': distress['risk_level'],
        'risk_color': distress['risk_color'],
        'risk_label': distress['risk_label'],
        'alert_triggered': distress['alert_triggered'],
    }

    return result


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(input_path, output_path):
    ffmpeg_path = r'C:\Users\rohit\AppData\Local\Temp\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = 'ffmpeg'
    try:
        import subprocess
        result = subprocess.run(
            [ffmpeg_path, '-y', '-i', input_path, '-ar', '22050', '-ac', '1', output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print(f"ffmpeg error: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("ffmpeg not found")
        return False
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/api/status')
def status():
    stt_status = stt_engine.get_status()
    return jsonify({
        'model_loaded': model_loaded,
        'demo_mode': not model_loaded,
        'emotions': list(EMOTION_META.keys()),
        'emotion_meta': EMOTION_META,
        'risk_levels': RISK_LABELS,
        'emergency_contacts': EMERGENCY_CONTACTS,
        'stt_available': stt_engine.model is not None,
        'stt_status': stt_status,
        'scaler_loaded': scaler is not None,
    })

@app.route('/api/predict/file', methods=['POST'])
def predict_from_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    save_name = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    file.save(file_path)

    try:
        audio_props = get_audio_properties(file_path)
        result = predict_emotion(file_path)
        if result is None:
            return jsonify({'error': 'Could not process audio. File may be too short or corrupted.'}), 400

        # Generate spectrogram
        spectrogram = spectrogram_generator.generate_spectrogram(file_path)
        waveform = spectrogram_generator.generate_waveform(file_path)
        result['spectrogram'] = spectrogram
        result['waveform'] = waveform
        result['audio_properties'] = audio_props
        result['filename'] = filename

        # Update session tracker
        session_tracker.add(result)

        # Trigger alert if needed
        if result['alert_triggered']:
            alert_manager.trigger(result)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        try:
            os.remove(file_path)
        except:
            pass

@app.route('/api/predict/live', methods=['POST'])
def predict_from_live():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data received'}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
        webm_path = tmp_webm.name
        audio_file.save(webm_path)

    try:
        success = convert_to_wav(webm_path, wav_path)
        if not success:
            return jsonify({'error': 'Could not process recording format. Please try again.'}), 400

        audio_props = get_audio_properties(wav_path)
        if audio_props['duration'] < 0.5:
            return jsonify({'error': 'Recording too short. Please speak for at least 1 second.'}), 400

        result = predict_emotion(wav_path)
        if result is None:
            return jsonify({'error': 'Could not analyze recording. Please try again.'}), 400

        # Generate spectrogram
        spectrogram = spectrogram_generator.generate_spectrogram(wav_path)
        waveform = spectrogram_generator.generate_waveform(wav_path)
        result['spectrogram'] = spectrogram
        result['waveform'] = waveform
        result['audio_properties'] = audio_props
        result['filename'] = 'Live Recording'

        # Update session tracker
        session_tracker.add(result)

        # Trigger alert if needed
        if result['alert_triggered']:
            alert_manager.trigger(result)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        try:
            os.remove(webm_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

@app.route('/api/analyze/full', methods=['POST'])
def full_analysis():
    """Complete analysis: emotion + distress + STT + spectrogram."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data'}), 400

    file = request.files['audio']
    language = request.form.get('language', 'en')

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
        webm_path = tmp_webm.name
        file.save(webm_path)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        convert_to_wav(webm_path, wav_path)
        audio_props = get_audio_properties(wav_path)

        # Emotion prediction
        result = predict_emotion(wav_path)
        if not result:
            return jsonify({'error': 'Could not process audio'}), 400

        # Spectrogram
        result['spectrogram'] = spectrogram_generator.generate_spectrogram(wav_path)
        result['waveform'] = spectrogram_generator.generate_waveform(wav_path)
        result['audio_properties'] = audio_props

        # Speech-to-text + sentiment (only if audio is long enough)
        if audio_props['duration'] >= 1.5 and stt_engine.model:
            stt_result = stt_engine.transcribe(wav_path, language=language)
            result['transcript'] = stt_result

        # Update session
        session_tracker.add(result)

        if result['alert_triggered']:
            alert_manager.trigger(result)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(webm_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Speech-to-text only."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio data'}), 400

    file = request.files['audio']
    language = request.form.get('language', 'en')

    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
        webm_path = tmp_webm.name
        file.save(webm_path)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        convert_to_wav(webm_path, wav_path)
        result = stt_engine.transcribe(wav_path, language=language)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(webm_path)
        except:
            pass
        try:
            os.remove(wav_path)
        except:
            pass

@app.route('/api/analyze/distress', methods=['POST'])
def analyze_distress():
    """Just distress score from audio probabilities."""
    data = request.get_json()
    if not data or 'probabilities' not in data:
        return jsonify({'error': 'Missing probabilities'}), 400

    distress = distress_engine.compute(data['probabilities'])
    return jsonify(distress)

@app.route('/api/session/summary')
def session_summary():
    return jsonify(session_tracker.get_summary())

@app.route('/api/session/history')
def session_history():
    limit = request.args.get('limit', 100, type=int)
    return jsonify(session_tracker.get_history(limit=limit))

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    session_tracker.clear()
    alert_manager.clear()
    return jsonify({'status': 'cleared'})

@app.route('/api/emotions')
def get_emotions():
    return jsonify(EMOTION_META)

@app.route('/api/emergency/alerts')
def emergency_alerts():
    limit = request.args.get('limit', 10, type=int)
    return jsonify({'alerts': alert_manager.get_recent(limit)})

@app.route('/api/emergency/trigger', methods=['POST'])
def trigger_emergency():
    data = request.get_json() or {}
    alert = alert_manager.trigger(data)
    return jsonify({'alert': alert, 'timestamp': alert['timestamp']})

@app.route('/api/visualize/spectrogram', methods=['GET'])
@app.route('/api/visualize/spectrogram/<filename>', methods=['GET'])
def visualize_spectrogram(filename=None):
    """Return spectrogram PNG for a previously uploaded file or generate from temp."""
    # For temp spectrograms, data is passed as base64 in query param
    data_b64 = request.args.get('data', '')
    if data_b64:
        return jsonify({'spectrogram': data_b64})

    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            spectrogram = spectrogram_generator.generate_spectrogram(file_path)
            return jsonify({'spectrogram': spectrogram})

    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Voice Crisis Detector — AI Emergency Detection System")
    print("="*60)
    print(f"  Model status:     {'Loaded' if model_loaded else 'Demo mode'}")
    print(f"  Whisper STT:     {'Available' if stt_engine.model else 'Not loaded'}")
    print(f"  Sentiment:       {'Available' if stt_engine.sentiment_analyzer else 'Not loaded'}")
    print(f"  Open:           http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)