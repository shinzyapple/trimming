import streamlit as st
import numpy as np
import soundfile as sf
import io
from dtw import dtw
from pydub import AudioSegment

st.set_page_config(page_title="音声マッチ＆トリミングツール", layout="centered")

st.title("🎵 音声マッチ＆トリミングツール（軽量版・MP3/FLAC対応）")
st.caption("音源Aと録音Bを比較し、類似する部分を検出して30秒ずつトリミングします。")

# --- アップロード ---
file_a = st.file_uploader("音源Aをアップロード", type=["wav","mp3","flac"])
file_b = st.file_uploader("録音Bをアップロード", type=["wav","mp3","flac"])
trim_sec = st.number_input("トリミング時間（秒）", min_value=5, max_value=120, value=30)

# --- 音声読み込み関数 ---
def load_audio(file) -> tuple[np.ndarray,int]:
    """pydubで任意形式の音声を読み込み numpy 配列に変換"""
    audio = AudioSegment.from_file(file)
    y = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        y = y.reshape((-1, audio.channels)).mean(axis=1)  # モノラル化
    sr = audio.frame_rate
    return y / np.max(np.abs(y)), sr

def extract_feature(y, frame_size=2048, hop=512):
    """波形エネルギーを特徴量として抽出"""
    feature = []
    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i+frame_size]
        energy = np.sum(frame ** 2)
        feature.append(energy)
    return np.array(feature)

def find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec):
    feat_a = extract_feature(y_a)
    feat_b = extract_feature(y_b)

    _, _, _, path = dtw(feat_a.reshape(-1,1), feat_b.reshape(-1,1), dist=lambda x,y: np.abs(x-y))
    idx_a, idx_b = np.array(path[0]), np.array(path[1])
    start_a = int(np.percentile(idx_a, 10))
    start_b = int(np.percentile(idx_b, 10))

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

# --- ボタン処理 ---
if st.button("マッチしてトリミング実行"):
    if not file_a or not file_b:
        st.error("⚠️ 両方の音声ファイルをアップロードしてください。")
    else:
        with st.spinner("処理中...少しお待ちください"):
            y_a, sr_a = load_audio(file_a)
            y_b, sr_b = load_audio(file_b)

            buf_a, buf_b = find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec)

        st.success("✅ トリミング完了！")
        st.audio(buf_a, format="audio/wav")
        st.download_button("音源A（トリミング済）をダウンロード", buf_a, file_name="trimmed_A.wav")
        st.audio(buf_b, format="audio/wav")
        st.download_button("録音B（トリミング済）をダウンロード", buf_b, file_name="trimmed_B.wav")
