import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
from dtw import dtw

st.set_page_config(page_title="音声マッチ＆トリミングツール", layout="centered")

st.title("🎵 音声マッチ＆トリミングツール")
st.caption("音源A（リファレンス）と録音Bを比較し、類似する部分を検出して30秒ずつトリミングします。")

# --- アップロード ---
file_a = st.file_uploader("音源Aをアップロード", type=["wav", "mp3"])
file_b = st.file_uploader("録音Bをアップロード", type=["wav", "mp3"])

trim_sec = st.number_input("トリミング時間（秒）", min_value=5, max_value=120, value=30)

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc

def find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec):
    # MFCC特徴を抽出
    mfcc_a = extract_mfcc(y_a, sr_a)
    mfcc_b = extract_mfcc(y_b, sr_b)

    # DTWで類似区間を検出
    dist, cost, acc_cost, path = dtw(mfcc_a.T, mfcc_b.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    idx_a, idx_b = np.array(path[0]), np.array(path[1])
    start_a = int(np.percentile(idx_a, 10))
    start_b = int(np.percentile(idx_b, 10))

    len_a = int(sr_a * trim_sec)
    len_b = int(sr_b * trim_sec)

    end_a = min(start_a + len_a, len(y_a))
    end_b = min(start_b + len_b, len(y_b))

    trimmed_a = y_a[start_a:end_a]
    trimmed_b = y_b[start_b:end_b]

    # バイナリデータとして出力
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
            # 音声読み込み
            y_a, sr_a = librosa.load(file_a, sr=None)
            y_b, sr_b = librosa.load(file_b, sr=None)

            buf_a, buf_b = find_and_trim(y_a, sr_a, y_b, sr_b, trim_sec)

        st.success("✅ トリミング完了！")

        st.audio(buf_a, format="audio/wav", start_time=0)
        st.download_button("音源A（トリミング済）をダウンロード", buf_a, file_name="trimmed_A.wav")

        st.audio(buf_b, format="audio/wav", start_time=0)
        st.download_button("録音B（トリミング済）をダウンロード", buf_b, file_name="trimmed_B.wav")
