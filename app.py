import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.io import wavfile
from scipy.signal import correlate
from dtw import dtw

st.set_page_config(page_title="音声マッチ＆トリミングツール", layout="centered")

st.title("🎵 音声マッチ＆トリミングツール（軽量版）")
st.caption("音源A（リファレンス）と録音Bを比較し、類似する部分を検出して30秒ずつトリミングします。")

# --- アップロード ---
file_a = st.file_uploader("音源Aをアップロード", type=["wav"])
file_b = st.file_uploader("録音Bをアップロード", type=["wav"])

trim_sec = st.number_input("トリミング時間（秒）", min_value=5, max_value=120, value=30)

def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

def resample_if_needed(y, sr, target_sr=16000):
    """Streamlit Cloudで安定動作するように簡易リサンプリング"""
    if sr == target_sr:
        return y, sr
    x_old = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), int(len(y) * target_sr / sr))
    y_resampled = np.interp(x_new, x_old, y)
    return y_resampled.astype(np.float32), target_sr

def extract_feature(y, frame_size=2048, hop=512):
    """波形のエネルギー包絡を特徴として抽出"""
    feature = []
    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i+frame_size]
        energy = np.sum(frame ** 2)
        feature.append(energy)
    return np.array(feature)

def find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec):
    y_a, sr_a = resample_if_needed(y_a, sr_a)
    y_b, sr_b = resample_if_needed(y_b, sr_b)

    # 特徴量抽出（波形エネルギー）
    feat_a = extract_feature(normalize_audio(y_a))
    feat_b = extract_feature(normalize_audio(y_b))

    # DTWで最小距離区間を探す
    _, _, _, path = dtw(feat_a.reshape(-1, 1), feat_b.reshape(-1, 1), dist=lambda x, y: np.abs(x - y))
    idx_a, idx_b = np.array(path[0]), np.array(path[1])
    start_a = int(np.percentile(idx_a, 10))
    start_b = int(np.percentile(idx_b, 10))

    # トリミング
    trim_len_a = int(sr_a * trim_sec)
    trim_len_b = int(sr_b * trim_sec)

    start_a_samp = int(start_a * 512)
    start_b_samp = int(start_b * 512)

    trimmed_a = y_a[start_a_samp:start_a_samp + trim_len_a]
    trimmed_b = y_b[start_b_samp:start_b_samp + trim_len_b]

    buf_a = io.BytesIO()
    buf_b = io.BytesIO()
    sf.write(buf_a, trimmed_a, sr_a, format='WAV')
    sf.write(buf_b, trimmed_b, sr_b, format='WAV')
    buf_a.seek(0)
    buf_b.seek(0)

    return buf_a, buf_b

if st.button("マッチしてトリミング実行", type="primary"):
    if not file_a or not file_b:
        st.error("⚠️ 両方の音声ファイルをアップロードしてください。")
    else:
        with st.spinner("処理中...少しお待ちください"):
            sr_a, y_a = wavfile.read(file_a)
            sr_b, y_b = wavfile.read(file_b)

            # 正規化してfloat化
            y_a = y_a.astype(np.float32)
            y_b = y_b.astype(np.float32)
            y_a = normalize_audio(y_a)
            y_b = normalize_audio(y_b)

            buf_a, buf_b = find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec)

        st.success("✅ トリミング完了！")

        st.audio(buf_a, format="audio/wav")
        st.download_button("音源A（トリミング済）をダウンロード", buf_a, file_name="trimmed_A.wav")

        st.audio(buf_b, format="audio/wav")
        st.download_button("録音B（トリミング済）をダウンロード", buf_b, file_name="trimmed_B.wav")
