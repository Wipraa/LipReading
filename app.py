"""
SIBI Lip Reading — Streamlit Web App
Mendukung 3 model sekaligus:
  1. Baseline RGB Conv-LSTM
  2. LK P17 Single-Stream Conv-LSTM
  3. Dual-Stream RGB + LK P17 Conv-LSTM

Jalankan:
  cd /home/wipra-ranum/Documents/Skripsi/LipReading
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.interpolate import griddata
from pathlib import Path
import tempfile, os, json, urllib.request

# ─────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────
BASE_DIR     = Path('/home/wipra-ranum/Documents/Skripsi/LipReading')
PROJECT_ROOT = BASE_DIR / 'Lip_Reading_Gabungan'
MODELS_DIR   = PROJECT_ROOT / 'models'
MP_MODEL     = BASE_DIR / 'face_landmarker.task'
STATS_FILE   = PROJECT_ROOT / 'preprocessed' / 'dependent' / 'flows_lk_p17' / 'norm_stats.json'
CONFIG_FILE  = PROJECT_ROOT / 'config.json'

MODEL_OPTIONS = {
    'Baseline RGB Conv-LSTM':             MODELS_DIR / 'baseline_gabungan_best.pth',
    'LK P17 Single-Stream Conv-LSTM':     MODELS_DIR / 'lk_p17_gabungan_best.pth',
    'Dual-Stream RGB + LK P17 Conv-LSTM': MODELS_DIR / 'dualstream_lk_p17_gabungan_best.pth',
}

# Preprocessing constants — identik notebook 03 & 04c
RESIZE_TO    = (224, 224)
LIP_SIZE     = (96, 96)
LIP_MARGIN   = 0.30
TARGET_N     = 30
RGB_SEQ_LEN  = 30
FLOW_SEQ_LEN = 29

# LK P17 parameters — identik notebook 04c
FEAT_PARAMS = dict(maxCorners=300, qualityLevel=0.005, minDistance=5, blockSize=6)
LK_PARAMS   = dict(
    winSize=(15, 15), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    minEigThreshold=1e-4
)

# Lip landmark — identik notebook 03
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
             291, 409, 270, 269, 267,  0, 37,  39,  40, 185]
INNER_LIP = [78,  95, 88, 178, 87, 14, 317, 402, 318, 324,
             308, 415, 310, 311, 312, 13,  82,  81,  80, 191]
LIP_LANDMARKS = list(set(OUTER_LIP + INNER_LIP))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────
# ARSITEKTUR MODEL — identik notebook 05, 06c, 07c
# ─────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, h_ch, k=3):
        super().__init__()
        self.h_ch = h_ch
        self.conv = nn.Conv2d(in_ch + h_ch, 4 * h_ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], dim=1)), self.h_ch, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, dev):
        return (torch.zeros(B, self.h_ch, H, W, device=dev),
                torch.zeros(B, self.h_ch, H, W, device=dev))


def _cblock(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1),
        nn.BatchNorm2d(oc), nn.ReLU(), nn.MaxPool2d(2)
    )


def _run_convlstm(x, lstm1, lstm2):
    """Helper: jalankan dua layer ConvLSTM terhadap sekuens x [B,T,C,H,W]."""
    B, T, c, h, w = x.shape
    h1, c1 = lstm1.init_hidden(B, h, w, x.device)
    h2, c2 = lstm2.init_hidden(B, h, w, x.device)
    for t in range(T):
        h1, c1 = lstm1(x[:, t], (h1, c1))
        h2, c2 = lstm2(h1, (h2, c2))
    return h2.mean(dim=[-2, -1])   # [B, 64]


class BaselineModel(nn.Module):
    """Input: [B, 3, 30, 96, 96] RGB."""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.cnn   = nn.Sequential(_cblock(3, 32), _cblock(32, 64), _cblock(64, 128))
        self.lstm1 = ConvLSTMCell(128, 128)
        self.lstm2 = ConvLSTMCell(128, 64)
        self.head  = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.cnn(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        _, c, h, w = feat.shape
        feat = feat.view(B, T, c, h, w)
        return self.head(_run_convlstm(feat, self.lstm1, self.lstm2))


class LKFlowModel(nn.Module):
    """Input: [B, 3, 29, 96, 96] (u, v, magnitude)."""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.cnn   = nn.Sequential(_cblock(3, 32), _cblock(32, 64), _cblock(64, 128))
        self.lstm1 = ConvLSTMCell(128, 128)
        self.lstm2 = ConvLSTMCell(128, 64)
        self.head  = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.cnn(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        _, c, h, w = feat.shape
        feat = feat.view(B, T, c, h, w)
        return self.head(_run_convlstm(feat, self.lstm1, self.lstm2))


class StreamEncoder(nn.Module):
    """Encoder per-stream → feature vector [B, 64]."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.cnn   = nn.Sequential(_cblock(in_ch, 32), _cblock(32, 64), _cblock(64, 128))
        self.lstm1 = ConvLSTMCell(128, 128)
        self.lstm2 = ConvLSTMCell(128, 64)

    def forward(self, x):
        B, C, T, H, W = x.shape
        feat = self.cnn(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        _, c, h, w = feat.shape
        feat = feat.view(B, T, c, h, w)
        return _run_convlstm(feat, self.lstm1, self.lstm2)


class AttentionFusion(nn.Module):
    """Learned sigmoid gate per stream."""
    def __init__(self, feat_dim=64):
        super().__init__()
        self.w_rgb  = nn.Linear(feat_dim * 2, feat_dim)
        self.w_flow = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, rgb_feat, flow_feat):
        combined  = torch.cat([rgb_feat, flow_feat], dim=1)
        gate_rgb  = torch.sigmoid(self.w_rgb(combined))
        gate_flow = torch.sigmoid(self.w_flow(combined))
        return gate_rgb * rgb_feat + gate_flow * flow_feat


class DualStreamModel(nn.Module):
    """RGB [B,3,30,96,96] + Flow [B,3,29,96,96] → logits."""
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.enc_rgb  = StreamEncoder(in_ch=3)
        self.enc_flow = StreamEncoder(in_ch=3)
        self.fusion   = AttentionFusion(feat_dim=64)
        self.head     = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, rgb, flow):
        return self.head(self.fusion(self.enc_rgb(rgb), self.enc_flow(flow)))


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    return cfg['data']['classes']


@st.cache_resource
def load_norm_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            s = json.load(f)
        return (np.array(s['clip_min'], dtype=np.float32),
                np.array(s['clip_max'], dtype=np.float32),
                np.array(s['mean'],     dtype=np.float32),
                np.array(s['std'],      dtype=np.float32))
    return None, None, None, None


@st.cache_resource
def load_mediapipe():
    if not MP_MODEL.exists():
        st.info('Mengunduh face_landmarker.task (~30 MB)...')
        url = ('https://storage.googleapis.com/mediapipe-models/'
               'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
        urllib.request.urlretrieve(url, MP_MODEL)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MP_MODEL)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


@st.cache_resource
def load_model(model_name: str, num_classes: int):
    path = MODEL_OPTIONS[model_name]
    if not path.exists():
        return None, f'File tidak ditemukan: `{path}`'
    if model_name.startswith('Baseline'):
        m = BaselineModel(num_classes).to(DEVICE)
    elif model_name.startswith('LK P17 Single'):
        m = LKFlowModel(num_classes).to(DEVICE)
    else:
        m = DualStreamModel(num_classes).to(DEVICE)
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    m.load_state_dict(state)
    m.eval()
    return m, None


# ─────────────────────────────────────────────────────────────
# PREPROCESSING — identik notebook 03 & 04c
# ─────────────────────────────────────────────────────────────

def extract_all_frames(video_path):
    cap, frames = cv2.VideoCapture(str(video_path)), []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, RESIZE_TO))
    cap.release()
    return frames


def get_lip_bbox(landmarks, h, w):
    coords = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                       for i in LIP_LANDMARKS])
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    bw, bh  = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * LIP_MARGIN))
    y0 = max(0, int(y0 - bh * LIP_MARGIN))
    x1 = min(w, int(x1 + bw * LIP_MARGIN))
    y1 = min(h, int(y1 + bh * LIP_MARGIN))
    return x0, y0, x1, y1


def detect_lips_all_frames(frames, detector):
    lip_regions, detect_mask = [], []
    for frame in frames:
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        if result.face_landmarks:
            lm           = result.face_landmarks[0]
            x0, y0, x1, y1 = get_lip_bbox(lm, h, w)
            crop         = frame[y0:y1, x0:x1]
            if crop.size > 0:
                lip_regions.append(crop)
                detect_mask.append(True)
                continue
        lip_regions.append(None)
        detect_mask.append(False)
    return lip_regions, detect_mask


def select_top30_by_lip_motion(lip_regions, detect_mask):
    valid = [i for i, ok in enumerate(detect_mask) if ok]
    if not valid:
        return None
    thumbs = {i: cv2.cvtColor(cv2.resize(lip_regions[i], (32, 32)),
                               cv2.COLOR_BGR2GRAY).astype(np.float32)
              for i in valid}
    scores, prev = {}, None
    for i in valid:
        scores[i] = 0.0 if prev is None else np.abs(thumbs[i] - thumbs[prev]).mean()
        prev = i
    if len(valid) <= TARGET_N:
        sel = valid.copy()
        while len(sel) < TARGET_N:
            sel.append(valid[-1])
        return sel
    top = sorted(scores, key=scores.get, reverse=True)[:TARGET_N]
    return sorted(top)


def crop_selected_frames(lip_regions, selected):
    crops = []
    for i in selected:
        c = lip_regions[i]
        crops.append(cv2.resize(c, LIP_SIZE) if c is not None
                     else np.zeros((*LIP_SIZE, 3), dtype=np.uint8))
    return np.array(crops, dtype=np.uint8)   # [30, 96, 96, 3]


def sparse_to_dense(src_pts, flow_vecs, H, W):
    if len(src_pts) < 4:
        return np.zeros((H, W, 3), dtype=np.float32)
    gy, gx = np.mgrid[0:H, 0:W]
    grid   = np.column_stack([gx.ravel(), gy.ravel()])
    u = griddata(src_pts, flow_vecs[:, 0], grid, method='linear',
                 fill_value=0.0).reshape(H, W).astype(np.float32)
    v = griddata(src_pts, flow_vecs[:, 1], grid, method='linear',
                 fill_value=0.0).reshape(H, W).astype(np.float32)
    return np.stack([u, v, np.sqrt(u**2 + v**2)], axis=-1)


def compute_lk_flow(frames_gray):
    T, H, W = frames_gray.shape
    if T < 2:
        return None
    flows = []
    for t in range(T - 1):
        p0 = cv2.goodFeaturesToTrack(frames_gray[t], mask=None, **FEAT_PARAMS)
        if p0 is None or len(p0) < 4:
            flows.append(np.zeros((H, W, 3), dtype=np.float32))
            continue
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            frames_gray[t], frames_gray[t + 1], p0, None, **LK_PARAMS
        )
        mask = status.ravel() == 1
        if mask.sum() < 4:
            flows.append(np.zeros((H, W, 3), dtype=np.float32))
            continue
        old = p0[mask].reshape(-1, 2)
        new = p1[mask].reshape(-1, 2)
        flows.append(sparse_to_dense(old, new - old, H, W))
    return np.array(flows, dtype=np.float32)   # [29, 96, 96, 3]


def normalize_flow(flows, clip_min, clip_max, mean, std):
    out = flows.copy()
    if clip_min is not None:
        for ch in range(3):
            out[..., ch] = np.clip(out[..., ch], clip_min[ch], clip_max[ch])
            out[..., ch] = (out[..., ch] - mean[ch]) / std[ch]
    else:
        p2, p98 = np.percentile(out, 2), np.percentile(out, 98)
        out = np.clip(out, p2, p98)
        out = (out - out.mean()) / (out.std() + 1e-8)
    return out.astype(np.float32)


def fix_len(seq, target):
    T = len(seq)
    if T == target:
        return seq
    if T < target:
        pad = np.zeros((target - T, *seq.shape[1:]), dtype=seq.dtype)
        return np.concatenate([seq, pad])
    return seq[np.linspace(0, T - 1, target, dtype=int)]


def preprocess_video(video_path, detector, clip_min, clip_max, mean, std, model_name):
    frames = extract_all_frames(video_path)
    if not frames:
        return None, 'Gagal membaca frame dari video.'

    lip_regions, detect_mask = detect_lips_all_frames(frames, detector)
    if not any(detect_mask):
        return None, ('Wajah / bibir tidak terdeteksi dalam video. '
                      'Pastikan wajah tampak jelas dan pencahayaan cukup.')

    selected = select_top30_by_lip_motion(lip_regions, detect_mask)
    if selected is None:
        return None, 'Seleksi frame gagal.'

    lip_seq = crop_selected_frames(lip_regions, selected)   # [30, 96, 96, 3] uint8

    # ── Baseline (RGB only) ─────────────────────────────────
    if model_name.startswith('Baseline'):
        rgb = fix_len(lip_seq.astype(np.float32) / 255.0, RGB_SEQ_LEN)
        t   = torch.from_numpy(rgb.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(DEVICE)
        return t, None   # [1, 3, 30, 96, 96]

    # ── LK optical flow ─────────────────────────────────────
    gray  = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in lip_seq])
    flows = compute_lk_flow(gray)                           # [29, 96, 96, 3]
    if flows is None:
        return None, 'Perhitungan optical flow gagal.'
    flows  = normalize_flow(flows, clip_min, clip_max, mean, std)
    flows  = fix_len(flows, FLOW_SEQ_LEN)
    flow_t = torch.from_numpy(flows.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(DEVICE)
    # [1, 3, 29, 96, 96]

    if model_name.startswith('LK P17 Single'):
        return flow_t, None

    # ── Dual-Stream ─────────────────────────────────────────
    rgb   = fix_len(lip_seq.astype(np.float32) / 255.0, RGB_SEQ_LEN)
    rgb_t = torch.from_numpy(rgb.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(DEVICE)
    return (rgb_t, flow_t), None   # [1,3,30,96,96] & [1,3,29,96,96]


@torch.no_grad()
def run_inference(model, inputs, model_name):
    if model_name.startswith('Dual'):
        rgb_t, flow_t = inputs
        logits = model(rgb_t, flow_t)
    else:
        logits = model(inputs)
    probs    = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return pred_idx, float(probs[pred_idx]), probs


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title='SIBI Lip Reading', page_icon='👁️', layout='centered')

st.title('👁️ SIBI Lip Reading')
st.markdown('**Pembacaan Gerak Bibir SIBI — Optical Flow Conv-LSTM**')
st.markdown('Universitas Udayana · Informatika · Tugas Akhir 2025')
st.divider()

# Config & stats
try:
    CLASSES = load_config()
except Exception as e:
    st.error(f'Gagal membaca `config.json`: {e}')
    st.stop()

NUM_CLASSES  = len(CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
clip_min, clip_max, mean, std = load_norm_stats()

if clip_min is None:
    st.warning('⚠️ `norm_stats.json` tidak ditemukan — normalisasi menggunakan statistik lokal video.')

# Sidebar
with st.sidebar:
    st.header('⚙️ Pengaturan')
    selected_model_name = st.selectbox('Pilih Model', list(MODEL_OPTIONS.keys()))

    path = MODEL_OPTIONS[selected_model_name]
    if path.exists():
        st.success(f'✅ Model tersedia ({path.stat().st_size / 1e6:.1f} MB)')
    else:
        st.error(f'❌ Tidak ditemukan:\n`{path.name}`')

    st.divider()
    st.markdown(f'**Device** : `{DEVICE}`')
    st.markdown(f'**Classes** : {NUM_CLASSES} kelas')
    with st.expander('Daftar kelas'):
        st.write(', '.join(CLASSES))

# Load model & mediapipe
with st.spinner(f'Memuat {selected_model_name}...'):
    model, model_err = load_model(selected_model_name, NUM_CLASSES)
if model_err:
    st.error(f'❌ {model_err}')
    st.stop()

with st.spinner('Memuat MediaPipe FaceLandmarker...'):
    try:
        detector = load_mediapipe()
    except Exception as e:
        st.error(f'❌ MediaPipe gagal dimuat: {e}')
        st.stop()

# Status card
c1, c2, c3 = st.columns(3)
c1.metric('Model', selected_model_name.split(' ')[0])
c2.metric('Kelas', str(NUM_CLASSES))
c3.metric('Device', str(DEVICE).upper())
st.divider()

# Upload
st.subheader('📤 Upload Video SIBI')
INPUT_INFO = {
    'Baseline RGB Conv-LSTM':             'RGB lip crop [30 frame]',
    'LK P17 Single-Stream Conv-LSTM':     'Optical Flow u,v,mag [29 frame] — P17',
    'Dual-Stream RGB + LK P17 Conv-LSTM': 'RGB [30f] + Optical Flow [29f] — P17',
}
st.caption(f'Input model ini: **{INPUT_INFO[selected_model_name]}**')

uploaded = st.file_uploader(
    'Upload video mentah (.mp4 / .avi / .mov / .mkv)',
    type=['mp4', 'avi', 'mov', 'mkv']
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(tmp_path)
    st.caption(f'📁 {uploaded.name}  ·  {uploaded.size / 1024:.1f} KB')
    st.divider()

    if st.button('🔍 Prediksi Label', type='primary', use_container_width=True):
        bar = st.progress(0, text='Memulai preprocessing...')
        try:
            bar.progress(10, text='Mengekstrak & mendeteksi frame bibir...')
            inputs, err = preprocess_video(
                tmp_path, detector,
                clip_min, clip_max, mean, std,
                selected_model_name
            )
            if err:
                st.error(f'❌ {err}')
            else:
                bar.progress(75, text='Menjalankan Conv-LSTM...')
                pred_idx, confidence, all_probs = run_inference(model, inputs, selected_model_name)
                pred_label = IDX_TO_CLASS[pred_idx]
                bar.progress(100, text='Selesai!')

                st.divider()
                st.subheader('📊 Hasil Prediksi')

                col_a, col_b = st.columns(2)
                col_a.metric('🎯 Prediksi', pred_label)
                col_b.metric('📈 Confidence', f'{confidence * 100:.1f}%')

                st.markdown('#### Distribusi Probabilitas')
                st.bar_chart({CLASSES[i]: float(all_probs[i]) for i in range(NUM_CLASSES)},
                             height=280)

                st.markdown('#### Top-3')
                for rank, i in enumerate(np.argsort(all_probs)[::-1][:3], 1):
                    icon  = ['🥇', '🥈', '🥉'][rank - 1]
                    bar_n = int(all_probs[i] * 20)
                    st.write(f'{icon} **{CLASSES[i]}** — {all_probs[i]*100:.1f}%  '
                             f'`{"█"*bar_n}{"░"*(20-bar_n)}`')

                if confidence >= 0.60:
                    st.success(f'✅ Model cukup yakin → **{pred_label}** ({confidence*100:.1f}%)')
                elif confidence >= 0.35:
                    st.warning(f'⚠️ Kemungkinan **{pred_label}**, confidence sedang '
                               f'({confidence*100:.1f}%). Perhatikan distribusi.')
                else:
                    st.error('❌ Confidence rendah — pastikan wajah jelas, '
                             'pencahayaan cukup, dan kelas ada dalam dataset model.')

        except Exception as e:
            st.error(f'❌ Error tidak terduga: {e}')
            st.exception(e)
        finally:
            bar.empty()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

st.divider()
st.caption('SIBI Lip Reading · Universitas Udayana · Informatika · 2025')
