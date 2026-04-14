"""
AI 電商素材生成器
基於 Google AI Studio (Gemini) 的全自動電商素材生成流程

流程：上傳襪子平拍照 → 自動產出提示詞 → 生成模特兒實穿照 → 生成社群文案
"""

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import anthropic
import base64
import io
import json
import os
import random
import pathlib
import time as _time
import zipfile

# ─────────────────────────────────────────
# 場景描述設定（統一管理，Step 2/3 共用）
# ─────────────────────────────────────────
SCENE_CONFIG = {
    "簡約室內（白色大理石地板）": [
        "indoor setting with white marble floor, clean minimal Scandinavian interior, soft natural window light, warm white tones, potted green plants as accents",
        "bright minimalist room with white marble tiles, sheer curtain diffused sunlight, light wood furniture, airy and clean atmosphere",
    ],
    "咖啡廳外拍（暖陽散景）": [
        "outdoor European-style cafe terrace, warm golden afternoon sunlight, blurred bokeh of cafe umbrellas and greenery, vintage rattan chairs, cobblestone ground",
        "cozy sidewalk cafe with string lights, late afternoon golden hour glow, espresso cup on table as prop, warm film photography tones with gentle lens flare",
    ],
    "清爽白背景（電商主圖）": [
        "pure white studio cyclorama background, soft diffused studio lighting from above, bright and airy atmosphere, clean e-commerce hero shot",
        "seamless white backdrop with subtle shadow on floor, rim lighting from behind, professional product photography studio setup",
    ],
    "日系街道散步（清新文青）": [
        "quiet Japanese-style narrow alley with old wooden houses, dappled sunlight through leaves, a bicycle leaning on the wall, soft pastel color palette, Fuji film aesthetic",
        "charming Japanese shopping street with small potted plants lining the path, morning soft light, vintage signboards, gentle nostalgic atmosphere",
    ],
    "校園青春（活力陽光）": [
        "university campus green lawn with large trees, bright midday sunlight, red brick building in background, youthful energetic atmosphere, clear blue sky",
        "school courtyard with wooden bench and scattered autumn leaves, warm afternoon light through campus trees, cheerful and vibrant student life atmosphere",
    ],
    "都市街拍（時尚潮流）": [
        "urban city street with modern glass buildings, crosswalk and traffic blur in background, overcast diffused light, street fashion editorial style, concrete and steel tones",
        "trendy neighborhood with colorful murals and graffiti wall, neon sign reflections on wet pavement after rain, edgy metropolitan vibe, high contrast cinematic look",
    ],
    "公園遊樂場（可愛活潑）": [
        "colorful children's playground in a sunny park, bright yellow slide and climbing frame in background, soft green rubber mat ground, vivid primary color play equipment, cheerful and playful atmosphere, warm natural sunlight, youthful and energetic mood",
        "outdoor playground with swings and rope climbing net, colorful plastic play structures, green grass and rubber safety surface, bright midday sun with clear blue sky, fun and lively atmosphere, candy-colored background tones",
    ],
}

# ─────────────────────────────────────────
# 影片 Prompt 模板（4 種風格）
# ─────────────────────────────────────────
VIDEO_PROMPT_TEMPLATES = {
    "A": {
        "name": "日系甜美",
        "scenes": ["簡約室內（白色大理石地板）", "清爽白背景（電商主圖）", "日系街道散步（清新文青）"],
        "prompt": (
            "Model wearing socks walks naturally, then lifts foot to show sock details, "
            "camera gradually zooms in to focus on the foot and sock pattern close-up, "
            "holds for 2 seconds on the pattern details, then in the last 5 seconds camera slowly zooms out "
            "to reveal the full body, model strikes a sweet cute ending pose with a gentle smile, "
            "soft dreamy Japanese-style BGM throughout."
        ),
    },
    "B": {
        "name": "韓系街拍",
        "scenes": ["都市街拍（時尚潮流）", "公園遊樂場（可愛活潑）"],
        "prompt": (
            "Model wearing socks walks briskly on the street with a side-tracking camera angle, "
            "stops at a spot and steps one foot onto a low ledge to showcase the socks, "
            "camera enters slow-motion close-up on sock details, holds for 2 seconds, "
            "then in the last 5 seconds camera pulls back wide, model turns back and flashes a finger heart ending pose, "
            "upbeat Korean lo-fi hip-hop BGM throughout."
        ),
    },
    "C": {
        "name": "咖啡廳慵懶",
        "scenes": ["咖啡廳外拍（暖陽散景）"],
        "prompt": (
            "Model sitting by a cafe window crosses legs and gently swings foot to display socks, "
            "camera slowly pushes in from beside a coffee cup towards the foot and sock pattern close-up, "
            "holds for 2 seconds on details, then in the last 5 seconds camera gently pulls back to full body, "
            "model picks up coffee cup and smiles warmly as ending pose, "
            "warm bossa nova acoustic BGM throughout."
        ),
    },
    "D": {
        "name": "校園青春",
        "scenes": ["校園青春（活力陽光）"],
        "prompt": (
            "Model wearing socks jogs lightly through a school corridor or playground, "
            "reaches a bench and lifts foot onto it to display socks, "
            "camera quickly zooms in to sock pattern close-up, holds for 2 seconds, "
            "then in the last 5 seconds camera zooms out wide, "
            "model sits on the bench with hands cradling chin in a sweet ending pose, "
            "bright Japanese school pop BGM throughout."
        ),
    },
}

_VIDEO_PROMPT_FIXED_SUFFIX = (
    " Keep the EXACT SAME outfit, socks, shoes, and background as shown in the image. "
    "CRITICAL: The socks must maintain the EXACT SAME pattern, color, and design as shown in the reference image throughout the entire video. "
    "Pay special attention to preserving the sock pattern details accurately — do NOT alter or simplify the sock design."
)


def _match_video_prompt_template(scene_key: str) -> str:
    """根據場景自動匹配最適合的影片 prompt 模板，回傳模板 key（A/B/C/D）"""
    for tmpl_key, tmpl in VIDEO_PROMPT_TEMPLATES.items():
        if scene_key in tmpl["scenes"]:
            return tmpl_key
    return random.choice(list(VIDEO_PROMPT_TEMPLATES.keys()))


# ─────────────────────────────────────────
# API 重試包裝函式（最多 3 次，指數退避）
# ─────────────────────────────────────────
def retry_api_call(fn, *args, max_retries=3, base_delay=2, **kwargs):
    """呼叫 fn(*args, **kwargs)，失敗時最多重試 max_retries 次（指數退避），並在 UI 顯示重試狀態"""
    status_placeholder = st.empty()
    for attempt in range(max_retries):
        try:
            result = fn(*args, **kwargs)
            status_placeholder.empty()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                next_attempt = attempt + 2
                for remaining in range(delay, 0, -1):
                    status_placeholder.warning(
                        f"⏳ 生成失敗，{remaining} 秒後重試 (第 {next_attempt}/{max_retries} 次)..."
                    )
                    _time.sleep(1)
                status_placeholder.info(f"🔄 正在重試中 (第 {next_attempt}/{max_retries} 次)...")
            else:
                status_placeholder.error(f"❌ 重試 {max_retries} 次後仍失敗")
                raise

# ─────────────────────────────────────────
# 費用計算
# ─────────────────────────────────────────
_CLAUDE_IN_PRICE  = 3.0  / 1_000_000   # Claude Sonnet 4.6: $3 / 1M input tokens
_CLAUDE_OUT_PRICE = 15.0 / 1_000_000   # Claude Sonnet 4.6: $15 / 1M output tokens
_GEMINI_IMG_PRICE = 0.04                # Gemini 3.1 Flash Image: ~$0.04 / 張（預估）
_KLING_STD_PRICE  = 0.028               # Kling v3 std: ~$0.028 / 秒（預估）
_KLING_PRO_PRICE  = 0.056               # Kling v3 pro: ~$0.056 / 秒（預估）

def _cost_claude(inp: int, out: int) -> float:
    return inp * _CLAUDE_IN_PRICE + out * _CLAUDE_OUT_PRICE

def _cost_gemini_images(n: int = 1) -> float:
    return n * _GEMINI_IMG_PRICE

def _cost_kling(duration_sec: int, mode: str = "std") -> float:
    rate = _KLING_PRO_PRICE if mode == "pro" else _KLING_STD_PRICE
    return duration_sec * rate


def _generate_bgm_wav(duration_sec: float, volume: float = 0.3) -> str:
    """用 sine wave 和弦生成簡單的背景音樂 WAV 檔，不需外部音樂檔案。"""
    import wave as _wave
    import numpy as _np
    import tempfile as _tmp

    sample_rate = 44100
    n = int(sample_rate * duration_sec)
    t = _np.linspace(0, duration_sec, n, endpoint=False)

    # C 大調和弦：C4-E4-G4-C5，加上 C3 低音根音
    chord_freqs = [261.63, 329.63, 392.00, 523.25]
    bass_freq = 130.81  # C3

    signal = _np.zeros(n)
    for f in chord_freqs:
        signal += _np.sin(2 * _np.pi * f * t) * (0.6 / len(chord_freqs))
    signal += _np.sin(2 * _np.pi * bass_freq * t) * 0.25

    # 0.5 Hz LFO 讓音量緩慢起伏，更有韻律感
    lfo = 0.75 + 0.25 * _np.sin(2 * _np.pi * 0.5 * t)
    signal *= lfo * volume

    # 淡入淡出（各 0.4 秒）
    fade = min(int(0.4 * sample_rate), n // 4)
    signal[:fade] *= _np.linspace(0, 1, fade)
    signal[-fade:] *= _np.linspace(1, 0, fade)

    signal = _np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(_np.int16)

    tmp = _tmp.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with _wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return tmp.name


def _mix_bgm_into_video(video_bytes: bytes, bgm_volume: float) -> bytes:
    """用 moviepy 將程式生成的 BGM 疊加到影片中，原有音效保留。"""
    import tempfile as _tmp
    import os as _os

    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    except ImportError:
        return video_bytes  # moviepy 未安裝時原樣返回

    vid_path = _tmp.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out_path = _tmp.NamedTemporaryFile(suffix=".mp4", delete=False).name
    bgm_path = None

    try:
        with open(vid_path, "wb") as f:
            f.write(video_bytes)

        video = VideoFileClip(vid_path)
        bgm_path = _generate_bgm_wav(video.duration, volume=bgm_volume)
        bgm_audio = AudioFileClip(bgm_path)

        if video.audio is not None:
            mixed = CompositeAudioClip([video.audio, bgm_audio])
            final = video.set_audio(mixed)
        else:
            final = video.set_audio(bgm_audio)

        final.write_videofile(
            out_path, codec="libx264", audio_codec="aac",
            logger=None, verbose=False,
        )
        with open(out_path, "rb") as f:
            result = f.read()
        return result
    except Exception:
        return video_bytes  # 失敗時原樣返回，不中斷主流程
    finally:
        for p in [vid_path, out_path]:
            try:
                _os.unlink(p)
            except Exception:
                pass
        if bgm_path:
            try:
                _os.unlink(bgm_path)
            except Exception:
                pass


# ─────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI 電商素材生成器",
    page_icon="🧦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 自訂 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .step-header {
        background: linear-gradient(90deg, #f8a5c2, #f3d9e7);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        color: #4a2040;
    }
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #276749;
    }
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #4299e1;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #2c5282;
        font-size: 0.9rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #f06292, #e91e8c);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar：API Key 設定 + 說明
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://ai.google.dev/static/site-assets/images/share.png", width=160)
    st.markdown("## ⚙️ 設定")

    # Gemini API Key（用於 Step 3 圖片生成）
    env_gemini = os.environ.get("GEMINI_API_KEY", "")
    if "api_key_cache" not in st.session_state:
        st.session_state.api_key_cache = env_gemini

    api_key = st.text_input(
        "Gemini API Key（圖片生成用）",
        type="password",
        value=st.session_state.api_key_cache,
        placeholder="AIzaSy...",
        help="前往 https://aistudio.google.com/apikey 取得免費 API Key"
    )
    if api_key:
        st.session_state.api_key_cache = api_key

    # Anthropic API Key（用於 Step 2 分析 + Step 4 文案）
    env_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
    if "anthropic_key_cache" not in st.session_state:
        st.session_state.anthropic_key_cache = env_anthropic

    anthropic_key = st.text_input(
        "Anthropic API Key（Claude 分析 & 文案用）",
        type="password",
        value=st.session_state.anthropic_key_cache,
        placeholder="sk-ant-...",
        help="前往 https://console.anthropic.com/ 取得 API Key"
    )
    if anthropic_key:
        st.session_state.anthropic_key_cache = anthropic_key

    # Kling AI API Key（用於 Step 5 影片生成）
    env_kling_ak = os.environ.get("KLING_ACCESS_KEY", "")
    env_kling_sk = os.environ.get("KLING_SECRET_KEY", "")
    if "kling_ak_cache" not in st.session_state:
        st.session_state.kling_ak_cache = env_kling_ak
    if "kling_sk_cache" not in st.session_state:
        st.session_state.kling_sk_cache = env_kling_sk

    with st.expander("🎬 Kling AI Key（影片生成用）", expanded=False):
        kling_ak_input = st.text_input(
            "Access Key",
            type="password",
            value=st.session_state.kling_ak_cache,
            placeholder="ak-...",
            help="前往 https://kling.ai/dev/resource-pack-manage 取得 API Key"
        )
        kling_sk_input = st.text_input(
            "Secret Key",
            type="password",
            value=st.session_state.kling_sk_cache,
            placeholder="sk-...",
        )
        if kling_ak_input:
            st.session_state.kling_ak_cache = kling_ak_input
        if kling_sk_input:
            st.session_state.kling_sk_cache = kling_sk_input

    # 優先從 session cache 讀取（包含環境變數和手動輸入）
    kling_ak = st.session_state.kling_ak_cache or env_kling_ak
    kling_sk = st.session_state.kling_sk_cache or env_kling_sk

    if api_key and anthropic_key:
        st.markdown('<div class="success-box">✅ Gemini + Anthropic API Key 已設定</div>', unsafe_allow_html=True)
    else:
        missing = []
        if not api_key: missing.append("Gemini")
        if not anthropic_key: missing.append("Anthropic")
        st.markdown(
            f'<div class="info-box">⚠️ 尚未設定：{", ".join(missing)} API Key</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 累計花費 ──
    _total_cost = (
        st.session_state.get("cost_step2", 0.0) +
        st.session_state.get("cost_step3_total", 0.0) +
        st.session_state.get("cost_step4", 0.0) +
        st.session_state.get("cost_step5", 0.0) +
        st.session_state.get("cost_step6", 0.0)
    )
    st.metric("💰 本次累計花費", f"${_total_cost:.4f}")

    # ── 帳戶狀態 ──
    with st.expander("📊 服務帳戶狀態", expanded=False):
        _anthr_ok = bool(anthropic_key)
        st.markdown(f"{'✅' if _anthr_ok else '❌'} **Anthropic** {'已連結' if _anthr_ok else '尚未連結'}")
        if _anthr_ok:
            st.caption("餘額需手動查詢")
            st.markdown("[🔗 console.anthropic.com](https://console.anthropic.com/settings/billing)")

        _google_ok = bool(api_key)
        st.markdown(f"{'✅' if _google_ok else '❌'} **Google AI** {'已連結' if _google_ok else '尚未連結'}")
        if _google_ok:
            st.caption("Google AI Studio 按用量計費，需手動查詢")
            st.markdown("[🔗 aistudio.google.com](https://aistudio.google.com/apikey)")

        _kling_ok = bool(kling_ak and kling_sk)
        st.markdown(f"{'✅' if _kling_ok else '❌'} **Kling AI** {'已連結' if _kling_ok else '尚未連結'}")
        if _kling_ok:
            # 嘗試查詢 Kling 積分（快取於 session state）
            if "kling_balance_display" in st.session_state:
                st.caption(f"積分：{st.session_state.kling_balance_display}")
            if st.button("🔄 查詢 Kling 積分", key="btn_kling_balance"):
                try:
                    import jwt as _pyjwt_b
                    import requests as _req_b
                    import time as _tb
                    _tok_b = _pyjwt_b.encode(
                        {"iss": kling_ak, "exp": int(_tb.time()) + 300, "nbf": int(_tb.time()) - 5},
                        kling_sk, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"}
                    )
                    _rb = _req_b.get(
                        "https://api-global.klingai.com/v1/account/resource",
                        headers={"Authorization": f"Bearer {_tok_b}"},
                        timeout=5,
                    )
                    _db = _rb.json()
                    if _db.get("code") == 0 and _db.get("data"):
                        _credits = _db["data"].get("credits", {})
                        _avail = _credits.get("available", "?")
                        st.session_state.kling_balance_display = f"{_avail} 積分"
                    else:
                        st.session_state.kling_balance_display = "查詢失敗"
                except Exception:
                    st.session_state.kling_balance_display = "查詢失敗"
            st.markdown("[🔗 kling.ai/dev/resource-pack-manage](https://kling.ai/dev/resource-pack-manage)")

    st.markdown("---")

    # ── 側邊導航 ──
    st.markdown("### 🧭 快速導航")
    nav_mode = st.radio(
        "選擇執行模式",
        [
            "🔄 完整流程（Step 1→6）",
            "✍️ Step 4 · 社群文案",
            "🎬 Step 5 · 穿搭影音",
            "🏷️ Step 6 · 電商首圖",
        ],
        key="nav_mode",
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("### 📋 流程說明")
    st.markdown("""
1. 🖼️ **上傳**平拍照
2. 🔍 **分析**自動產出提示詞
3. 🎨 **生成**模特兒實穿照
4. ✍️ **生成**社群貼文文案
5. 🎬 **生成**穿搭短影音
6. 🏷️ **製作**電商首圖
""")
    st.markdown("---")
    st.markdown("### 🤖 使用模型")
    st.markdown("""
- **分析 & 文案**：`claude-sonnet-4-6` (Anthropic)
- **圖片生成**：`gemini-3.1-flash-image-preview` (Google)
""")
    st.caption("Powered by Anthropic Claude + Google Gemini")

# ─────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────
for key in ["prompts", "model_image_bytes", "model_images", "captions", "upload_mime", "selected_scene", "video_bytes", "video_generating", "hero_image", "hero_generated", "remaining_generated", "selected_outfit", "outfit_desc_en"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "model_images" not in st.session_state or st.session_state.model_images is None:
    st.session_state.model_images = []
for key in ["cost_step2", "cost_step3_total", "cost_step4", "cost_step5"]:
    if key not in st.session_state:
        st.session_state[key] = 0.0
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# ─────────────────────────────────────────
# 導航模式判斷
# ─────────────────────────────────────────
_nav = st.session_state.get("nav_mode", "🔄 完整流程（Step 1→6）")
_is_full = _nav.startswith("🔄")
show_step1 = _is_full
show_step2 = _is_full
show_step3 = _is_full
show_step4 = _is_full or "Step 4" in _nav
show_step5 = _is_full or "Step 5" in _nav
show_step6 = _is_full or "Step 6" in _nav
show_batch = _is_full

# ─────────────────────────────────────────
# 主標題
# ─────────────────────────────────────────
st.title("🧦 AI 電商素材生成器")
st.caption("以 Google AI Studio (Gemini) 為核心，從平拍照一鍵產出模特兒實穿照與社群文案")
st.divider()

# ─────────────────────────────────────────
# STEP 1：上傳平拍照
# ─────────────────────────────────────────

# 範例圖片檔案路徑（全域）
SAMPLE_IMAGE_PATH = pathlib.Path(__file__).parent / "test_product.jpg"

# MockUploadedFile：模擬 Streamlit UploadedFile 介面（全域）
class _MockFile:
    def __init__(self, data, name, mime_type):
        self._data = data
        self.name = name
        self.type = mime_type
        self.size = len(data)
    def read(self): return self._data
    def getvalue(self): return self._data
    def seek(self, pos): pass

uploaded_file = None
product_notes = ""

if show_step1:
    st.markdown('<div class="step-header">Step 1 · 🖼️ 上傳商品平拍照</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "選擇商品平拍圖片（支援 JPG / PNG / WEBP，可多選）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="建議使用清晰、白底或淺色背景的商品平拍照；可一次上傳多張，再用下方選擇器挑選要處理的一張"
    )

    # 多張上傳：縮圖網格 + checkbox 多選
    if uploaded_files:
        n_uploaded = len(uploaded_files)
        if n_uploaded == 1:
            uploaded_file = uploaded_files[0]
            st.session_state.selected_files = [uploaded_files[0]]
        else:
            st.markdown(f"**已上傳 {n_uploaded} 張圖片，請勾選要處理的圖片：**")

            # 當上傳的檔案清單改變時，重置選取狀態
            _files_key = tuple(f.name for f in uploaded_files)
            if st.session_state.get("_uploaded_files_key") != _files_key:
                st.session_state["_uploaded_files_key"] = _files_key
                for _ci in range(n_uploaded):
                    st.session_state[f"check_img_{_ci}"] = True
                st.session_state["_select_all_cb"] = True

            # 全選 callback
            def _on_select_all_change():
                _val = st.session_state.get("_select_all_cb", True)
                for _j in range(n_uploaded):
                    st.session_state[f"check_img_{_j}"] = _val

            st.checkbox("✅ 全選", key="_select_all_cb", on_change=_on_select_all_change)

            # 縮圖網格 + 個別 checkbox
            _cols_n = min(n_uploaded, 4)
            _thumb_cols = st.columns(_cols_n)
            _selected_files = []
            for _ci, _cf in enumerate(uploaded_files):
                with _thumb_cols[_ci % _cols_n]:
                    st.image(_cf.getvalue(), use_container_width=True)
                    if st.checkbox(_cf.name, key=f"check_img_{_ci}"):
                        _selected_files.append(_cf)

            st.session_state.selected_files = _selected_files

            if _selected_files:
                _n_sel = len(_selected_files)
                _mode = "批次處理模式" if _n_sel > 1 else "單張處理模式"
                st.markdown(f"**已選 {_n_sel}/{n_uploaded} 張**（{_mode}）")
                uploaded_file = _selected_files[0]
            else:
                st.warning("⚠️ 請至少選擇一張圖片")

    # 若未上傳，提供範例圖片按鈕
    if not uploaded_file:
        if SAMPLE_IMAGE_PATH.exists():
            if st.button("🧦 使用範例圖片（襪子平拍照）", type="secondary"):
                st.session_state["sample_bytes"] = SAMPLE_IMAGE_PATH.read_bytes()
                st.rerun()
        else:
            st.info("💡 請上傳一張襪子商品平拍照開始使用")

    # 若 session 中有範例圖
    if not uploaded_file and st.session_state.get("sample_bytes"):
        uploaded_file = _MockFile(st.session_state["sample_bytes"], "sample_sock.jpg", "image/jpeg")

    if uploaded_file:
        st.session_state.upload_mime = uploaded_file.type or "image/jpeg"
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(uploaded_file.getvalue(), caption="📸 已選擇的圖片", use_container_width=True)
        with col_info:
            st.markdown('<div class="success-box">✅ 圖片已選擇，可繼續下一步</div>', unsafe_allow_html=True)
            st.markdown(f"- **檔名**：`{uploaded_file.name}`")
            st.markdown(f"- **大小**：`{uploaded_file.size / 1024:.1f} KB`")
            st.markdown(f"- **格式**：`{uploaded_file.type}`")

        product_notes = st.text_area(
            "📝 商品注意事項（選填）",
            placeholder="例如：這款襪子花色偏淡，場景不要太暗；襪口有蕾絲邊設計請特別展示；此為兒童襪請用年輕活潑風格…",
            help="輸入任何關於這款襪子的特殊說明，Claude 會在 Step 2 自動將這些注意事項融入提示詞中",
            key="product_notes",
        )

    st.divider()

# ─────────────────────────────────────────
# STEP 2：自動產出提示詞
# ─────────────────────────────────────────
if show_step2:
    st.markdown('<div class="step-header">Step 2 · 🔍 自動分析並產出提示詞</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.info("請先完成 Step 1 上傳圖片")
    elif not anthropic_key:
        st.warning("請先在左側 Sidebar 輸入 Anthropic API Key")
    else:
        # ── 場景選擇（7 種場景，每種有 2 組隨機腳本） ──
        scene_options = SCENE_CONFIG
        selected_scene = st.selectbox("🏠 選擇場景", list(scene_options.keys()))
        st.session_state.selected_scene = selected_scene

        # ── 穿搭風格選擇 ──
        OUTFIT_STYLES = {
            "T恤 + 百褶短裙（甜美韓系）": (
                "wearing a soft pastel short-sleeve T-shirt tucked into a pleated mini skirt, "
                "white canvas sneakers, sweet Korean girl-next-door style"
            ),
            "針織衫 + A字裙（溫柔氣質）": (
                "wearing a delicate knit cardigan or short-sleeve knit top with an A-line midi skirt, "
                "Mary Jane shoes or loafers, elegant and feminine Korean style"
            ),
            "衛衣 + 寬褲（休閒街頭）": (
                "wearing an oversized cropped hoodie or sweatshirt with wide-leg pants or joggers, "
                "chunky sneakers or platform shoes, casual Korean streetwear style"
            ),
            "襯衫 + 牛仔短褲（清新日常）": (
                "wearing a crisp button-down shirt (tucked in or tied at waist) with denim shorts, "
                "white sneakers or slip-on shoes, fresh and casual everyday Korean look"
            ),
            "背心洋裝（簡約一件式）": (
                "wearing a sleeveless mini dress or pinafore dress over a simple inner top, "
                "flat sandals or canvas shoes, minimalist one-piece Korean outfit"
            ),
            "運動套裝（活力元氣）": (
                "wearing a sporty cropped zip-up jacket or sports bra top with bike shorts or track pants, "
                "athletic running shoes, energetic Korean athleisure style"
            ),
        }
        selected_outfit = st.selectbox("👗 穿搭風格", list(OUTFIT_STYLES.keys()), key="outfit_style")
        outfit_desc_en = OUTFIT_STYLES[selected_outfit]
        st.session_state.selected_outfit = selected_outfit
        st.session_state.outfit_desc_en = outfit_desc_en

        # ── 襪子資訊欄位 ──
        col_sock1, col_sock2 = st.columns(2)
        with col_sock1:
            sock_type = st.selectbox(
                "🧦 襪子類型",
                [
                    "長襪 — 小腿肚以上",
                    "中筒襪 — 腳踝以上，小腿肚以下",
                    "短襪 — 剛好蓋過腳踝",
                    "隱形襪 — 露出腳踝",
                ],
            )
        with col_sock2:
            sock_length = st.text_input(
                "📏 襪筒長度",
                placeholder="例：25cm、膝下10cm",
                help="輸入襪子的實際長度或相對位置描述"
            )

        # ── 襪子資訊組合 ──
        sock_type_en_map = {
            "長襪 — 小腿肚以上": "knee-high socks (above calf)",
            "中筒襪 — 腳踝以上，小腿肚以下": "crew socks / mid-calf socks (above ankle, below calf)",
            "短襪 — 剛好蓋過腳踝": "ankle socks (just covering the ankle)",
            "隱形襪 — 露出腳踝": "no-show socks / invisible socks (ankle exposed)",
        }
        sock_info_en = sock_type_en_map.get(sock_type, "socks")
        sock_length_desc = f", sock tube length approximately {sock_length}" if sock_length else ""
        sock_length_zh = f"，襪筒長度約 {sock_length}" if sock_length else ""
        scene_desc = random.choice(scene_options[selected_scene])
        sock_type_zh = sock_type.split(" — ")[0]

        # 儲存批次處理所需的分析參數（供 Step 4 後的批次處理區塊使用）
        st.session_state["_batch_params"] = {
            "sock_info_en": sock_info_en,
            "sock_length_desc": sock_length_desc,
            "sock_type_zh": sock_type_zh,
            "sock_length_zh": sock_length_zh,
            "sock_length": sock_length,
            "selected_scene": selected_scene,
            "selected_outfit": selected_outfit,
            "outfit_desc_en": outfit_desc_en,
        }

        if st.button("🔍 分析圖片並自動產出提示詞", type="primary", use_container_width=False):
            img_bytes = uploaded_file.getvalue()
            img_base64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            mime_type = st.session_state.upload_mime or "image/jpeg"

            _notes_block = ""
            if product_notes:
                _notes_block = f"\nSPECIAL NOTES FROM USER (you MUST incorporate these into your prompts):\n{product_notes}\n"

            analysis_prompt = f"""You are a professional e-commerce fashion photographer and AI image prompt engineer specializing in Korean style.

    Analyze this product flat lay image carefully and generate AI image generation prompts for a Korean female model wearing this product.

    PRODUCT INFO (MUST appear in the generated prompts):
    - Sock type: {sock_info_en}{sock_length_desc}
    - Scene / Background: {scene_desc}
    - Outfit style: {outfit_desc_en}
    {_notes_block}
    IMPORTANT RULES:
    - The generated positive_en prompt MUST explicitly contain these exact details:
      1. The sock type: "{sock_info_en}"
      2. The sock length: "{sock_length if sock_length else 'not specified'}" (include exact measurement if provided)
      3. The scene description keywords from: "{scene_desc}"
      4. The outfit description: "{outfit_desc_en}"
    - If user provided special notes above, incorporate them naturally into the positive_en prompt and reflect any constraints in negative_en
    - Do NOT describe the specific pattern, color, or design details of the product itself (the reference image will be provided separately to the image generation model)
    - Focus on the MODEL SCENE: pose, angle, background, lighting, styling
    - The sock type is "{sock_info_en}" — make sure the pose and camera angle clearly showcase socks at the correct height on the leg
    - Shot must be LOWER BODY only (waist down, include waist)
    - Korean female model aesthetic, slim legs
    - E-commerce commercial quality
    - Slight side angle to showcase the product
    - The negative prompt should prevent common AI image generation errors

    Return ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
    {{
      "positive_en": "Korean female model, slim legs, wearing {sock_info_en}{sock_length_desc}, {outfit_desc_en}, [pose details], lower body shot from waist down, {scene_desc}, [lighting], [photography quality]",
      "positive_zh": "韓系女性模特兒，穿著{sock_type_zh}{sock_length_zh}，{selected_outfit}，[姿勢細節]，腰部以下畫面，[場景]，[光線]，[攝影質感]",
      "negative_en": "full body, face visible, upper body dominant, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, missing product, duplicate body parts, bad anatomy, extra fingers, nsfw, wrong sock length, barefoot"
    }}"""

            # ── 嘗試 Claude，失敗則 fallback 到 Gemini ──
            analysis_text = None
            used_engine = None
            cost_info = None

            # 1) 先嘗試 Claude
            if anthropic_key:
                with st.spinner("Claude 正在分析商品圖片…"):
                    try:
                        claude_client = anthropic.Anthropic(api_key=anthropic_key)
                        response = retry_api_call(
                            claude_client.messages.create,
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime_type,
                                            "data": img_base64,
                                        },
                                    },
                                    {"type": "text", "text": analysis_prompt},
                                ],
                            }],
                        )
                        analysis_text = response.content[0].text.strip()
                        used_engine = "claude"
                        _inp2 = response.usage.input_tokens
                        _out2 = response.usage.output_tokens
                        _c2 = _cost_claude(_inp2, _out2)
                        cost_info = f"💰 本步驟花費：${_c2:.4f}（Input: {_inp2:,} tokens / Output: {_out2:,} tokens）"
                        st.session_state.cost_step2 = _c2
                    except Exception as e:
                        st.warning(f"⚠️ Claude API 失敗（{e}），嘗試使用 Gemini 分析…")

            # 2) Claude 失敗或無 key → fallback 到 Gemini
            if analysis_text is None and api_key:
                with st.spinner("Gemini 正在分析商品圖片…"):
                    try:
                        gemini_client = genai.Client(api_key=api_key)
                        gemini_response = retry_api_call(
                            gemini_client.models.generate_content,
                            model="gemini-2.5-flash",
                            contents=[
                                types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                                analysis_prompt,
                            ],
                        )
                        analysis_text = gemini_response.text.strip()
                        used_engine = "gemini"
                    except Exception as e2:
                        st.error(f"❌ Gemini 分析也失敗：{e2}")

            # 3) 無任何可用 API
            if analysis_text is None and not api_key and not anthropic_key:
                st.error("❌ 請先設定 Anthropic 或 Gemini API Key")

            # ── 解析結果 ──
            if analysis_text:
                try:
                    text = analysis_text
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()

                    st.session_state.prompts = json.loads(text)
                    engine_label = "Claude" if used_engine == "claude" else "Gemini"
                    st.success(f"✅ 提示詞已自動生成（使用 {engine_label}）！可在下方編輯後再生成圖片。")
                    if cost_info:
                        st.info(cost_info)
                except json.JSONDecodeError:
                    st.session_state.prompts = {
                        "positive_en": analysis_text,
                        "positive_zh": "",
                        "negative_en": "full body, face visible, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, duplicate body parts, bad anatomy, extra fingers",
                    }
                    st.warning("⚠️ JSON 解析失敗，原始回覆已填入正向提示詞欄位，請手動調整。")

    # 顯示並允許編輯提示詞
    if st.session_state.prompts:
        st.markdown("#### ✏️ 提示詞預覽（可直接編輯）")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pos_en = st.text_area(
                "正向提示詞（English）",
                value=st.session_state.prompts.get("positive_en", ""),
                height=130,
                key="pos_en_input",
            )
        with col_p2:
            pos_zh = st.text_area(
                "正向提示詞（中文版）",
                value=st.session_state.prompts.get("positive_zh", ""),
                height=130,
                key="pos_zh_input",
            )
        neg_en = st.text_area(
            "反向提示詞（Negative Prompts）",
            value=st.session_state.prompts.get("negative_en", ""),
            height=90,
            key="neg_en_input",
        )
        st.session_state.prompts["positive_en"] = pos_en
        st.session_state.prompts["positive_zh"] = pos_zh
        st.session_state.prompts["negative_en"] = neg_en

    st.divider()

# ─────────────────────────────────────────
# 單張圖片生成函式（供批次生成 & 個別重新生成共用）
# ─────────────────────────────────────────
def generate_single_photo(api_key_val, shot_config, base_prompt, neg_prompt, scene_desc, ref_part, hero_ref_part=None):
    """生成單張模特兒實穿照，回傳 dict: {label, bytes, error?}
    hero_ref_part: 第一張照片的 Part 物件，供後續照片參考服裝/背景一致性
    """
    client = genai.Client(api_key=api_key_val)

    # 如果有第一張照片作為參考，加入一致性指令
    consistency_block = ""
    if hero_ref_part:
        consistency_block = (
            f"[VISUAL CONSISTENCY REFERENCE — HIGHEST PRIORITY]\n"
            f"A 'hero reference photo' is attached (the second image). "
            f"You MUST match the following elements from that hero photo EXACTLY:\n"
            f"  • SAME outfit: identical clothing pieces, shoes (color, style, material)\n"
            f"  • SAME model appearance: same face, hair style, hair color, body type\n"
            f"  • SAME background/scene: identical location, props, and lighting mood\n"
            f"  • SAME color temperature and photography style\n"
            f"The ONLY change is the POSE described below — everything else must be identical.\n\n"
        )

    generation_prompt = (
        f"[CRITICAL INSTRUCTION - MUST FOLLOW EXACTLY]\n"
        f"Using the sock/stocking design shown in the reference image, "
        f"generate a photorealistic e-commerce model photo.\n\n"
        f"{consistency_block}"
        f"[SHOT TYPE - THIS IS THE MOST IMPORTANT REQUIREMENT]\n"
        f"{shot_config['shot_desc']}\n\n"
        f"[SCENE & STYLE]\n"
        f"A Korean female model wearing these exact socks with the same pattern, color, and design. "
        f"Scene: {scene_desc}. "
        f"Style: {base_prompt}, "
        f"photorealistic, commercial e-commerce photography, 8K resolution, "
        f"sharp fabric texture, feminine and elegant, editorial fashion quality.\n\n"
        f"[OUTFIT & VISUAL CONSISTENCY - CRITICAL]\n"
        f"ALL 5 photos in this set MUST show the EXACT SAME outfit: same clothing pieces, same colors, same shoes. "
        f"The ONLY product being showcased is the socks — everything else stays identical. "
        f"Same model (same face, same body type, same hair style and color). "
        f"Same lighting mood and color temperature across all shots. "
        f"The socks must faithfully reproduce the pattern from the reference image.\n\n"
        f"[ANATOMY & BODY CONSISTENCY]\n"
        f"The ENTIRE body must face the same direction — upper body and lower body must be aligned. "
        f"DO NOT twist the torso so that the chest faces a different direction than the legs. "
        f"Human feet must have exactly 5 toes each, natural proportions, no extra or merged toes. "
        f"Legs must have natural anatomy with proper knee and ankle joints. "
        f"No floating limbs, no extra legs, no distorted body parts.\n\n"
        f"[AVOID]\n"
        f"twisted torso, body facing different directions, upper body reversed, "
        f"inconsistent body direction, contorted pose, unnatural twist, "
        f"{neg_prompt}"
    )

    content_parts = []
    if ref_part:
        content_parts.append(ref_part)
    # 附上第一張照片作為視覺一致性參考
    if hero_ref_part:
        content_parts.append(hero_ref_part)
    content_parts.append(generation_prompt)

    try:
        response = retry_api_call(
            client.models.generate_content,
            model="gemini-3.1-flash-image-preview",
            contents=content_parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size="1K",
                ),
            ),
        )
        image_bytes = None
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for part in parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_bytes = part.inline_data.data
                        break
        if not image_bytes and hero_ref_part:
            # 帶 hero 參考圖失敗時，重試一次不帶 hero（fallback）
            content_parts_retry = []
            if ref_part:
                content_parts_retry.append(ref_part)
            content_parts_retry.append(generation_prompt)
            response2 = retry_api_call(
                client.models.generate_content,
                model="gemini-3.1-flash-image-preview",
                contents=content_parts_retry,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(image_size="1K"),
                ),
            )
            candidates2 = getattr(response2, "candidates", None)
            if candidates2 and len(candidates2) > 0:
                content2 = getattr(candidates2[0], "content", None)
                parts2 = getattr(content2, "parts", None) if content2 else None
                if parts2:
                    for part in parts2:
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_bytes = part.inline_data.data
                            break
        if image_bytes:
            # 強制縮放到 1024×1024（API 可能忽略 image_size 參數）
            img = Image.open(io.BytesIO(image_bytes))
            if img.width != 1024 or img.height != 1024:
                img = img.resize((1024, 1024), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            return {"label": shot_config["label"], "bytes": image_bytes}
        else:
            return {"label": shot_config["label"], "bytes": None, "error": "未收到圖片"}
    except Exception as e:
        return {"label": shot_config["label"], "bytes": None, "error": str(e)}

# ─────────────────────────────────────────
# STEP 3：生成模特兒實穿照（5 張照片組）
if show_step3:
    # ─────────────────────────────────────────
    st.markdown('<div class="step-header">Step 3 · 🎨 生成模特兒實穿照組（8 張）</div>', unsafe_allow_html=True)

    # ── 動作類型池（依拍攝範圍分類） ──
    # 全身照動作池（A / C / D）
    FULLBODY_POSES = [
        {
            "id": "A", "name": "高處坐姿",
            "shot_desc": (
                "FULL BODY photo from head to shoes. "
                "Model sitting on an elevated surface (bench, ledge, slide, or railing), legs dangling or stretched outward. "
                "Relaxed playful posture, one hand resting on the surface, the other touching hair or holding a prop. "
                "FACE FULLY VISIBLE: natural light makeup, gentle smile, looking at camera. "
                "Korean young woman, long hair. Socks and shoes clearly visible on dangling feet. "
                "Complete outfit visible: clothing, socks, and shoes all in frame."
            ),
        },
        {
            "id": "C", "name": "站姿俏皮",
            "shot_desc": (
                "FULL BODY photo from head to shoes. "
                "Model standing with a playful pose: one foot slightly lifted or on tiptoe, body tilted, "
                "one hand touching hat/hair or raised cheerfully. Dynamic and youthful energy. "
                "FACE FULLY VISIBLE: bright smile, looking at camera, natural light makeup. "
                "Korean young woman, long hair. Complete outfit visible: clothing, socks, and shoes."
            ),
        },
        {
            "id": "D", "name": "躺靠伸腳",
            "shot_desc": (
                "FULL BODY photo from head to shoes. "
                "Model leaning back casually on a railing, bench, or chair, legs stretched forward and slightly raised. "
                "One hand holding a drink or prop, relaxed happy expression. Socks are prominently displayed on the raised feet. "
                "FACE FULLY VISIBLE: natural smile, looking at camera or upward. "
                "Korean young woman, long hair. Complete outfit visible: clothing, socks, and shoes."
            ),
        },
        {
            "id": "I", "name": "活力前傾站姿",
            "shot_desc": (
                "FULL BODY photo from head to shoes. "
                "Model standing with a slight forward lean, one hand casually gripping a metal railing or fence, body tilted playfully forward, legs slightly apart showing socks clearly, energetic and dynamic full-body shot, natural outdoor playground or park setting. "
                "FACE FULLY VISIBLE: bright smile, looking at camera, natural light makeup. "
                "Korean young woman, long hair. Complete outfit visible: clothing, socks, and shoes."
            ),
        },
    ]

    # 下半身動作池（B / F / H）
    LOWERBODY_POSES = [
        {
            "id": "B", "name": "地面坐姿",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model sitting on the ground with legs extended forward or one leg bent. "
                "Camera at low angle, shooting toward the feet. Both socks clearly visible. "
                "Hands may rest on knees (only hands visible, no arms above elbow). "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "F", "name": "道具互動",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model's legs and feet with a lifestyle prop nearby: a basketball, coffee cup, tote bag, or book on the ground. "
                "Casual seated or standing pose, socks are the hero of the composition. "
                "Prop adds context and visual interest without stealing focus from socks. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "H", "name": "蹲姿抱膝",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model squatting or crouching on a low wall, stone, or curb, arms wrapped around knees (only hands visible). "
                "Camera at ground level, shooting straight at the feet and socks. "
                "Socks prominent and centered in composition. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "J", "name": "🦶 坐地放鬆",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model sitting on the ground with knees slightly bent upward, both feet flat on the ground and slightly apart with toes pointing slightly outward, lower body close-up shot from a low angle, socks clearly visible on both feet showing front pattern details, legs naturally relaxed, casual and effortless pose. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "K", "name": "側躺伸腿",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model half-reclining on the ground leaning on one arm, legs extended and slightly crossed, lower body close-up shot, side view of socks clearly showing pattern details, relaxed laid-back pose, outdoor ground-level perspective. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "L", "name": "💺 高台垂腳",
            "shot_desc": (
                "LOWER BODY ONLY from waist down. NO face, NO chest visible. "
                "Model sitting on an elevated concrete ledge or bench, one foot planted on the ground while the other foot hangs naturally off the edge, legs staggered front and back (not crossed or overlapping), hands clasped together resting above the front knee, lower body close-up shot, sock pattern on the front-facing leg prominently displayed, clean minimalist background with natural sunlight and shadows. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
        {
            "id": "M", "name": "🪜 階梯踩踏",
            "shot_desc": (
                "Model standing on stone or concrete steps with one foot on a higher step and the other foot on a lower step, side angle shot focusing on lower body, legs in a natural climbing pose showing sock details clearly on both feet, outdoor park or street stairway setting with railing. "
                "Correct anatomy: 5 toes per foot, natural proportions."
            ),
        },
    ]

    # 腳部特寫動作池（E / G）
    FEET_POSES = [
        {
            "id": "E", "name": "階梯踩踏",
            "shot_desc": (
                "CLOSE-UP of feet and ankles ONLY, from mid-calf down. NO knee, NO thigh. "
                "Model's feet stepping on stairs or a raised surface, shot from low angle looking upward. "
                "One foot on a higher step, one on a lower step, showing sock height and detail. "
                "Sharp focus on sock fabric texture with blurred stairway background. "
                "Correct foot anatomy: 5 toes, natural bone structure, no deformation."
            ),
        },
        {
            "id": "G", "name": "屈膝坐姿側面",
            "shot_desc": (
                "Model sitting down on the flat ground with buttocks touching the ground, both knees bent and pulled up toward chest, arms gently hugging or resting on the knees, shot from a slight side angle at ground level, the model is NOT standing, she is fully seated on the ground in a relaxed crouching sit position, socks visible on both feet which are flat on the ground, casual and youthful vibe. "
                "Sock side profile prominently displayed — pattern, color bands, and texture clearly visible. "
                "Outdoor setting, natural lighting. Correct anatomy throughout."
            ),
        },
    ]

    def build_shot_configs():
        """每次生成時隨機組合 8 張照片的動作：3 全身 + 4 下半身 + 1 屈膝坐姿"""
        full = random.sample(FULLBODY_POSES, 3)
        lower = random.sample(LOWERBODY_POSES, 4)
        feet = random.sample(FEET_POSES, 1)

        return [
            {"label": f"📷 全身照 ①（{full[0]['name']}）", "shot_desc": full[0]["shot_desc"]},
            {"label": f"📷 全身照 ②（{full[1]['name']}）", "shot_desc": full[1]["shot_desc"]},
            {"label": f"📷 全身照 ③（{full[2]['name']}）", "shot_desc": full[2]["shot_desc"]},
            {"label": f"🦵 下半身特寫 ①（{lower[0]['name']}）", "shot_desc": lower[0]["shot_desc"]},
            {"label": f"🦵 下半身特寫 ②（{lower[1]['name']}）", "shot_desc": lower[1]["shot_desc"]},
            {"label": f"🦵 下半身特寫 ③（{lower[2]['name']}）", "shot_desc": lower[2]["shot_desc"]},
            {"label": f"🦵 下半身特寫 ④（{lower[3]['name']}）", "shot_desc": lower[3]["shot_desc"]},
            {"label": f"🧎 屈膝坐姿（{feet[0]['name']}）", "shot_desc": feet[0]["shot_desc"]},
        ]

    # 初始化或使用已生成的 SHOT_CONFIGS（避免 rerun 時重新隨機）
    if "current_shot_configs" not in st.session_state:
        st.session_state.current_shot_configs = build_shot_configs()
    SHOT_CONFIGS = st.session_state.current_shot_configs

    if not st.session_state.prompts:
        st.info("請先完成 Step 2 產出提示詞")
    elif not api_key:
        st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
    else:
        # 顯示 Step 2 選擇的場景（唯讀提示）
        if st.session_state.selected_scene:
            st.info(f"🏠 使用場景：**{st.session_state.selected_scene}**（可在 Step 2 更換）")

        # 顯示 Step 2 選擇的穿搭風格（唯讀提示）
        if st.session_state.get("selected_outfit"):
            st.info(f"👗 穿搭風格：**{st.session_state.selected_outfit}**（可在 Step 2 更換）")

        st.markdown("**Step 3a**: 先生成第 1 張基準照 → **Step 3b**: 確認後再生成其餘 7 張")

        if st.button("🎨 生成第 1 張基準實穿照", type="primary", use_container_width=False):
            # 每次生成重新隨機組合動作
            st.session_state.current_shot_configs = build_shot_configs()
            SHOT_CONFIGS = st.session_state.current_shot_configs

            client = genai.Client(api_key=api_key)

            # 從 Step 2 取場景描述（每場景隨機選 1 組腳本）
            scene_variants = SCENE_CONFIG.get(
                st.session_state.selected_scene or "清爽白背景（電商主圖）",
                ["pure white studio background, soft diffused studio lighting"]
            )
            scene_desc = random.choice(scene_variants)

            base_prompt = st.session_state.prompts["positive_en"] + f", {st.session_state.get('outfit_desc_en', '')}"
            neg_prompt = st.session_state.prompts["negative_en"]

            # 準備上傳的原始商品圖片
            ref_part = None
            if uploaded_file:
                img_bytes = uploaded_file.getvalue()
                mime_type = st.session_state.upload_mime or "image/jpeg"
                ref_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

            # ── Step 3a: 只生成第 1 張基準照 ──
            hero_shot = SHOT_CONFIGS[0]
            with st.spinner(f"正在生成基準照 {hero_shot['label']}…約需 30～60 秒"):
                result = generate_single_photo(api_key, hero_shot, base_prompt, neg_prompt, scene_desc, ref_part)

            if result.get("bytes"):
                st.session_state.hero_image = result
                st.session_state.hero_generated = True
                st.session_state.remaining_generated = False
                # 初始化 model_images（先放第一張）
                st.session_state.model_images = [result]
                st.session_state.model_image_bytes = result["bytes"]
                # 儲存生成參數供 Step 3b 使用
                st.session_state["_gen_params"] = {
                    "base_prompt": base_prompt,
                    "neg_prompt": neg_prompt,
                    "scene_desc": scene_desc,
                }
                st.success("✅ 基準照生成完成！確認滿意後，按下方按鈕生成其餘 7 張。")
                _c3a = _cost_gemini_images(1)
                st.session_state.cost_step3_total = st.session_state.get("cost_step3_total", 0.0) + _c3a
                st.info(f"💰 本步驟花費：${_c3a:.4f}（Gemini 圖片生成 1 張，預估值）")
            else:
                st.error(f"❌ 基準照生成失敗：{result.get('error', '未知錯誤')}")

        # ── 顯示基準照 & Step 3b 按鈕 ──
        if st.session_state.hero_generated and st.session_state.hero_image and st.session_state.hero_image.get("bytes"):
            hero = st.session_state.hero_image
            st.markdown("#### 📸 基準照預覽")
            hero_img = Image.open(io.BytesIO(hero["bytes"]))
            st.image(hero_img, caption=hero["label"], width=400)

            if not st.session_state.remaining_generated:
                if st.button("✅ 基準照 OK，生成其餘 7 張", type="primary", use_container_width=False):
                    params = st.session_state.get("_gen_params", {})
                    base_prompt = params.get("base_prompt", st.session_state.prompts.get("positive_en", ""))
                    neg_prompt = params.get("neg_prompt", st.session_state.prompts.get("negative_en", ""))
                    scene_desc = params.get("scene_desc", "")

                    ref_part = None
                    if uploaded_file:
                        ref_part = types.Part.from_bytes(
                            data=uploaded_file.getvalue(),
                            mime_type=st.session_state.upload_mime or "image/jpeg"
                        )

                    hero_ref_part = types.Part.from_bytes(
                        data=hero["bytes"], mime_type="image/png"
                    )

                    remaining_images = []
                    progress_bar = st.progress(0, text="準備生成其餘照片…")

                    remaining_shots = SHOT_CONFIGS[1:]
                    for idx, shot in enumerate(remaining_shots):
                        progress_bar.progress(
                            idx / len(remaining_shots),
                            text=f"正在生成 {shot['label']}（{idx+2}/8 · 參考基準照）…約需 30～60 秒"
                        )
                        result = generate_single_photo(
                            api_key, shot, base_prompt, neg_prompt, scene_desc,
                            ref_part, hero_ref_part=hero_ref_part,
                        )
                        remaining_images.append(result)

                    progress_bar.progress(1.0, text="✅ 照片組生成完成！")

                    # 合併所有照片
                    all_images = [st.session_state.hero_image] + remaining_images
                    st.session_state.model_images = all_images
                    st.session_state.remaining_generated = True
                    for img in all_images:
                        if img.get("bytes"):
                            st.session_state.model_image_bytes = img["bytes"]
                            break
                    success_count = sum(1 for i in all_images if i.get("bytes"))
                    st.success(f"✅ 成功生成 {success_count} / 8 張照片！")
                    _remaining_ok = sum(1 for r in remaining_images if r.get("bytes"))
                    _c3b = _cost_gemini_images(_remaining_ok)
                    st.session_state.cost_step3_total = st.session_state.get("cost_step3_total", 0.0) + _c3b
                    st.info(f"💰 本步驟花費：${_c3b:.4f}（Gemini 圖片生成 {_remaining_ok} 張，預估值）")

    # ── 個別重新生成處理 ──
    def _get_regen_params():
        """取得重新生成所需的共用參數"""
        scene_variants = SCENE_CONFIG.get(
            st.session_state.selected_scene or "清爽白背景（電商主圖）",
            ["pure white studio background, soft diffused studio lighting"]
        )
        sd = random.choice(scene_variants)
        bp = st.session_state.prompts.get("positive_en", "") if st.session_state.prompts else ""
        np_ = st.session_state.prompts.get("negative_en", "") if st.session_state.prompts else ""
        rp = None
        if uploaded_file:
            rp = types.Part.from_bytes(
                data=uploaded_file.getvalue(),
                mime_type=st.session_state.upload_mime or "image/jpeg"
            )
        return bp, np_, sd, rp

    # 處理重新生成請求（在顯示之前處理，避免 rerun 問題）
    for regen_idx in range(8):
        regen_key = f"regen_photo_{regen_idx}"
        if st.session_state.get(regen_key):
            st.session_state[regen_key] = False
            if api_key and st.session_state.prompts and st.session_state.model_images:
                bp, np_, sd, rp = _get_regen_params()
                with st.spinner(f"🔄 正在重新生成第 {regen_idx+1} 張照片…"):
                    result = generate_single_photo(api_key, SHOT_CONFIGS[regen_idx], bp, np_, sd, rp)
                    st.session_state.model_images[regen_idx] = result
                    # 更新 model_image_bytes
                    for img in st.session_state.model_images:
                        if img.get("bytes"):
                            st.session_state.model_image_bytes = img["bytes"]
                            break
                if result.get("bytes"):
                    st.success(f"✅ 第 {regen_idx+1} 張照片重新生成成功！")
                    _c_regen = _cost_gemini_images(1)
                    st.session_state.cost_step3_total = st.session_state.get("cost_step3_total", 0.0) + _c_regen
                    st.info(f"💰 重新生成花費：${_c_regen:.4f}（Gemini 圖片生成 1 張，預估值）")
                else:
                    st.error(f"❌ 重新生成失敗：{result.get('error', '未知錯誤')}")

    # 顯示生成的照片組
    if st.session_state.model_images:
        st.markdown("### 📸 實穿照片組")

        # 通用照片顯示函式
        def _show_photo(img_data, idx, file_prefix):
            if img_data.get("bytes"):
                gen_img = Image.open(io.BytesIO(img_data["bytes"]))
                st.image(gen_img, caption=img_data["label"], use_container_width=True)
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="💾 下載",
                        data=img_data["bytes"],
                        file_name=f"{file_prefix}_{idx+1}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{file_prefix}_{idx}",
                    )
                with btn_col2:
                    if st.button("🔄 重新生成", key=f"btn_regen_{file_prefix}_{idx}", use_container_width=True):
                        st.session_state[f"regen_photo_{idx}"] = True
                        st.rerun()
            else:
                st.error(f"❌ {img_data['label']}：{img_data.get('error', '生成失敗')}")
                if st.button("🔄 重新生成", key=f"btn_regen_{file_prefix}_err_{idx}", use_container_width=True):
                    st.session_state[f"regen_photo_{idx}"] = True
                    st.rerun()

        total_imgs = len(st.session_state.model_images)

        # 第一行：3 張全身照（indices 0, 1, 2）
        st.markdown("**👗 全身照**")
        full_cols = st.columns(3)
        for i in range(3):
            with full_cols[i]:
                if i < total_imgs:
                    _show_photo(st.session_state.model_images[i], i, "fullbody")

        # 第二行：下半身特寫前 2 張（indices 3, 4）
        if total_imgs > 3:
            st.markdown("**🦵 下半身特寫**")
            lower_cols1 = st.columns(2)
            for i in range(2):
                real_idx = i + 3
                with lower_cols1[i]:
                    if real_idx < total_imgs:
                        _show_photo(st.session_state.model_images[real_idx], real_idx, "lower")

        # 第三行：下半身特寫後 2 張（indices 5, 6）
        if total_imgs > 5:
            lower_cols2 = st.columns(2)
            for i in range(2):
                real_idx = i + 5
                with lower_cols2[i]:
                    if real_idx < total_imgs:
                        _show_photo(st.session_state.model_images[real_idx], real_idx, "lower2")

        # 第四行：1 張屈膝坐姿（index 7）
        if total_imgs > 7:
            st.markdown("**🧎 屈膝坐姿**")
            feet_col, _ = st.columns([1, 2])
            with feet_col:
                _show_photo(st.session_state.model_images[7], 7, "feet")

        # 圖片資訊
        successful = [i for i in st.session_state.model_images if i.get("bytes")]
        if successful:
            st.markdown("---")
            st.markdown(f"📊 **圖片資訊**：共 {len(successful)} / 8 張成功")
            for img_data in successful:
                img_info = Image.open(io.BytesIO(img_data["bytes"]))
                st.caption(f"  {img_data['label']}：{img_info.width}×{img_info.height}，{len(img_data['bytes'])/1024:.0f} KB")

        # 批次下載（全部 8 張都成功才顯示）
        all_successful = [i for i in st.session_state.model_images if i.get("bytes")]
        if len(st.session_state.model_images) >= 8 and len(all_successful) == 8:
            st.markdown("---")
            product_name = (
                pathlib.Path(uploaded_file.name).stem if uploaded_file else "商品"
            )
            from datetime import datetime as _dt
            zip_filename = f"實穿照_{product_name}_{_dt.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for idx, img_data in enumerate(st.session_state.model_images):
                    if img_data.get("bytes"):
                        entry_name = f"{idx+1}_{img_data['label']}.png"
                        zf.writestr(entry_name, img_data["bytes"])
            st.download_button(
                label="📦 批次下載全部照片（ZIP）",
                data=zip_buf.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=False,
            )

if _is_full:
    st.divider()

# ─────────────────────────────────────────
# STEP 4：生成社群貼文文案
# ─────────────────────────────────────────
if show_step4:
  st.markdown('<div class="step-header">Step 4 · ✍️ 生成 Instagram 社群貼文文案</div>', unsafe_allow_html=True)

  if not anthropic_key:
    st.warning("請先在左側 Sidebar 輸入 Anthropic API Key")
  else:
    # ── 雙模式選擇器 ──
    _s4_pipeline_imgs = [i for i in (st.session_state.model_images or []) if i.get("bytes")]
    _s4_has_pipeline = len(_s4_pipeline_imgs) > 0 or st.session_state.model_image_bytes is not None

    if _s4_has_pipeline and _is_full:
        _s4_source = st.radio(
            "📷 圖片來源",
            ["使用流程中的實穿照", "直接上傳圖片"],
            key="s4_img_source",
            horizontal=True,
        )
    elif not _is_full:
        _s4_source = "直接上傳圖片"
        st.info("📷 獨立模式：請上傳要用於生成文案的商品照片。")
    else:
        _s4_source = "直接上傳圖片"

    # 準備圖片列表
    _s4_images = []  # list of (bytes, mime_type)

    if _s4_source == "使用流程中的實穿照":
        for si in _s4_pipeline_imgs[:8]:
            _s4_images.append((si["bytes"], "image/png"))
        if not _s4_images and st.session_state.model_image_bytes:
            _s4_images.append((st.session_state.model_image_bytes, "image/png"))
    else:
        _s4_uploads = st.file_uploader(
            "上傳商品/實穿照片（可多選，最多 8 張）",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="s4_upload_imgs",
        )
        if _s4_uploads:
            for f in _s4_uploads[:8]:
                f.seek(0)
                _s4_images.append((f.read(), f.type or "image/jpeg"))
            # 預覽
            _s4_pcols = st.columns(min(len(_s4_uploads), 4))
            for idx, f in enumerate(_s4_uploads[:8]):
                with _s4_pcols[idx % len(_s4_pcols)]:
                    st.image(f, width=140)

    col_style, col_lang = st.columns(2)
    with col_style:
        caption_style = st.selectbox(
            "✨ 文案風格",
            [
                "韓系少女活潑風（可愛、輕鬆、有個性）",
                "簡約質感電商風（專業、俐落、有品味）",
                "溫柔居家療癒風（溫暖、舒適、生活感）",
                "時髦潮流街頭風（大膽、時尚、話題性）",
            ],
        )
    with col_lang:
        caption_lang = st.selectbox("🌐 語言", ["繁體中文", "繁體中文 + English", "English only"])

    product_desc = st.text_input(
        "📦 補充商品資訊（選填）",
        placeholder="例：純棉材質、日本製、限定色、特價 NT$199",
        help="可補充材質、特色、價格等，讓文案更精準"
    )

    # 獨立模式時可手動輸入場景資訊
    if _s4_source == "直接上傳圖片":
        _s4_scene_manual = st.text_input(
            "🏠 場景描述（選填）",
            placeholder="例：咖啡廳午後、日系街道散步、白色簡約背景…",
            help="補充拍攝場景資訊讓文案更精準；留空時 AI 會自動從照片判斷",
            key="s4_scene_manual",
        )
    else:
        _s4_scene_manual = ""

    has_model_image = len(_s4_images) > 0
    if not has_model_image:
        st.info("💡 請上傳圖片或先完成 Step 3 生成實穿照，文案將根據照片的場景情境撰寫更精準的內容。")

    if st.button("✍️ 生成社群貼文文案", type="primary", use_container_width=False):
        with st.spinner("Claude 正在根據照片撰寫情境文案…" if has_model_image else "Claude 正在撰寫文案…"):
            try:
                claude_client = anthropic.Anthropic(api_key=anthropic_key)

                lang_instruction = {
                    "繁體中文": "請全程使用繁體中文（台灣用語）撰寫，包含標題、內文與 hashtag。",
                    "繁體中文 + English": "標題與內文使用繁體中文，hashtag 中英混合。",
                    "English only": "Please write entirely in English including all hashtags.",
                }[caption_lang]

                extra = f"\n- 商品特色補充：{product_desc}" if product_desc else ""

                # 場景資訊：優先使用手動輸入，否則取 Step 2
                scene_name = _s4_scene_manual if _s4_scene_manual else (st.session_state.selected_scene or "未指定（請根據照片自行判斷）")
                prompt_en = st.session_state.prompts.get("positive_en", "") if st.session_state.prompts else ""

                caption_prompt = f"""你是一位專業的電商社群媒體文案師，擅長韓系時尚品牌的 Instagram 行銷。

請根據「附圖中的模特兒實穿照」撰寫一篇高互動率的 Instagram 貼文文案。
文案必須與照片中的場景、氛圍、穿搭情境完全吻合。

【照片場景資訊】
- 拍攝場景：{scene_name}
- 照片中的穿搭提示詞：{prompt_en}

【商品與風格】
- 商品類型：韓系襪子（電商商品）
- 文案風格：{caption_style}{extra}
- 語言規範：{lang_instruction}

【撰寫要求】
1. 仔細觀察附圖中模特兒的姿勢、場景、光線、穿搭搭配
2. 文案需描述照片中的情境（例如：咖啡廳的午後、室內的慵懶時光等）
3. 將襪子自然融入穿搭場景的敘事中，不要只是單純介紹商品規格
4. 讓讀者看到文案就能聯想到照片中的畫面

請依照以下格式輸出，不要加其他說明：

【標題】
（1行，吸睛有力，與照片場景呼應，可含表情符號）

【貼文內容】
（4～6行，以照片場景為背景，自然口語化地帶出穿搭情境與商品亮點，含適量表情符號）

【Call to Action】
（1行，引導互動或購買）

【Hashtags】
（20～25個，分行整理，涵蓋：商品、穿搭、韓系、場景情境、季節、品味生活 等主題）
"""

                # 組合訊息內容
                message_content = []
                for _s4_img_bytes, _s4_img_mime in _s4_images:
                    img_b64 = base64.standard_b64encode(_s4_img_bytes).decode("utf-8")
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _s4_img_mime,
                            "data": img_b64,
                        },
                    })
                message_content.append({"type": "text", "text": caption_prompt})

                response = retry_api_call(
                    claude_client.messages.create,
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": message_content}],
                )
                st.session_state.captions = response.content[0].text
                st.success("✅ 文案生成完成！（已參考照片場景）" if has_model_image else "✅ 文案生成完成！")
                _inp4 = response.usage.input_tokens
                _out4 = response.usage.output_tokens
                _c4 = _cost_claude(_inp4, _out4)
                st.session_state.cost_step4 = _c4
                st.info(f"💰 本步驟花費：${_c4:.4f}（Input: {_inp4:,} tokens / Output: {_out4:,} tokens）")

            except Exception as e:
                st.error(f"❌ 文案生成失敗：{e}")

if show_step4 and st.session_state.captions:
    edited_caption = st.text_area(
        "📝 貼文文案（可直接編輯後複製使用）",
        value=st.session_state.captions,
        height=420,
        key="caption_output",
    )

    col_copy, col_dl2 = st.columns(2)
    with col_copy:
        st.download_button(
            label="📋 下載文案 .txt",
            data=edited_caption.encode("utf-8"),
            file_name="instagram_caption.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_dl2:
        st.markdown(
            '<div class="info-box">💡 下一步：將圖片與文案貼至 <strong>Instagram</strong>、<strong>Buffer</strong> 或 <strong>Later</strong> 排程發布！</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────
# 批次處理：多張圖片同時執行 Step 2-4
# ─────────────────────────────────────────
_sel_batch = st.session_state.get("selected_files", [])
_bp = st.session_state.get("_batch_params")

if show_batch and len(_sel_batch) > 1 and _bp:
    st.markdown('<div class="step-header">批次處理 · 🚀 對全部選中圖片執行完整流程</div>', unsafe_allow_html=True)
    st.markdown(
        f"已選取 **{len(_sel_batch)}** 張圖片。點擊下方按鈕，系統將依序對每張圖片執行"
        f"分析提示詞 → 生成 8 張實穿照 → 生成文案，完成後統一顯示結果。"
    )
    st.warning("⚠️ 批次處理需要較長時間（每張圖片約 3～5 分鐘），請耐心等候。")

    _bc1, _bc2, _bc3 = st.columns(3)
    with _bc1:
        _batch_style = st.selectbox(
            "✨ 文案風格",
            ["韓系少女活潑風（可愛、輕鬆、有個性）", "簡約質感電商風（專業、俐落、有品味）",
             "溫柔居家療癒風（溫暖、舒適、生活感）", "時髦潮流街頭風（大膽、時尚、話題性）"],
            key="batch_caption_style",
        )
    with _bc2:
        _batch_lang = st.selectbox("🌐 語言", ["繁體中文", "繁體中文 + English", "English only"], key="batch_caption_lang")
    with _bc3:
        _batch_product_desc = st.text_input("📦 商品補充（選填）", placeholder="例：純棉材質、特價 NT$199", key="batch_product_desc")

    if st.button(f"🚀 批次處理全部 {len(_sel_batch)} 張圖片", type="primary", key="btn_batch_all", use_container_width=False):
        _batch_results_new = []
        _total_b = len(_sel_batch)
        _prog_b = st.progress(0, text="正在準備批次處理…")
        _status_b = st.empty()

        _b_ak = st.session_state.get("anthropic_key_cache", "")
        _b_gk = st.session_state.get("api_key_cache", "")

        _b_lang_map = {
            "繁體中文": "請全程使用繁體中文（台灣用語）撰寫，包含標題、內文與 hashtag。",
            "繁體中文 + English": "標題與內文使用繁體中文，hashtag 中英混合。",
            "English only": "Please write entirely in English including all hashtags.",
        }

        for _bidx, _bfile in enumerate(_sel_batch):
            _prog_b.progress(_bidx / _total_b, text=f"正在處理第 {_bidx+1}/{_total_b} 張：{_bfile.name}")
            _status_b.markdown(f"**正在處理第 {_bidx+1}/{_total_b} 張：** `{_bfile.name}`")
            _bres = {"filename": _bfile.name, "error": None, "prompts": None, "images": [], "caption": ""}

            try:
                _b_bytes = _bfile.getvalue()
                _b_b64 = base64.standard_b64encode(_b_bytes).decode("utf-8")
                _b_mime = _bfile.type or "image/jpeg"
                _b_scene = random.choice(SCENE_CONFIG.get(_bp["selected_scene"], ["pure white studio background, soft diffused studio lighting"]))

                # ── Step 2：分析圖片產出提示詞 ──
                _b_notes_block = ""
                if product_notes:
                    _b_notes_block = f"\nSPECIAL NOTES FROM USER (you MUST incorporate these into your prompts):\n{product_notes}\n"

                _b_analysis_prompt = f"""You are a professional e-commerce fashion photographer and AI image prompt engineer specializing in Korean style.

Analyze this product flat lay image carefully and generate AI image generation prompts for a Korean female model wearing this product.

PRODUCT INFO (MUST appear in the generated prompts):
- Sock type: {_bp['sock_info_en']}{_bp['sock_length_desc']}
- Scene / Background: {_b_scene}
- Outfit style: {_bp['outfit_desc_en']}
{_b_notes_block}
IMPORTANT RULES:
- The generated positive_en prompt MUST explicitly contain these exact details:
  1. The sock type: "{_bp['sock_info_en']}"
  2. The sock length: "{_bp['sock_length'] if _bp['sock_length'] else 'not specified'}" (include exact measurement if provided)
  3. The scene description keywords from: "{_b_scene}"
  4. The outfit description: "{_bp['outfit_desc_en']}"
- If user provided special notes above, incorporate them naturally into the positive_en prompt and reflect any constraints in negative_en
- Do NOT describe the specific pattern, color, or design details of the product itself (the reference image will be provided separately to the image generation model)
- Focus on the MODEL SCENE: pose, angle, background, lighting, styling
- The sock type is "{_bp['sock_info_en']}" — make sure the pose and camera angle clearly showcase socks at the correct height on the leg
- Shot must be LOWER BODY only (waist down, include waist)
- Korean female model aesthetic, slim legs
- E-commerce commercial quality
- Slight side angle to showcase the product
- The negative prompt should prevent common AI image generation errors

Return ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
{{
  "positive_en": "Korean female model, slim legs, wearing {_bp['sock_info_en']}{_bp['sock_length_desc']}, {_bp['outfit_desc_en']}, [pose details], lower body shot from waist down, {_b_scene}, [lighting], [photography quality]",
  "positive_zh": "韓系女性模特兒，穿著{_bp['sock_type_zh']}{_bp['sock_length_zh']}，{_bp['selected_outfit']}，[姿勢細節]，腰部以下畫面，[場景]，[光線]，[攝影質感]",
  "negative_en": "full body, face visible, upper body dominant, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, missing product, duplicate body parts, bad anatomy, extra fingers, nsfw, wrong sock length, barefoot"
}}"""

                # ── 嘗試 Claude，失敗則 fallback 到 Gemini ──
                _b_analysis_text = None
                if _b_ak:
                    try:
                        _b_claude = anthropic.Anthropic(api_key=_b_ak)
                        _b_resp = retry_api_call(
                            _b_claude.messages.create,
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            messages=[{"role": "user", "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": _b_mime, "data": _b_b64}},
                                {"type": "text", "text": _b_analysis_prompt},
                            ]}],
                        )
                        _b_analysis_text = _b_resp.content[0].text.strip()
                    except Exception as _be:
                        _status_b.warning(f"⚠️ Claude 失敗，切換 Gemini…（{_be}）")

                if _b_analysis_text is None and _b_gk:
                    _b_gemini = genai.Client(api_key=_b_gk)
                    _b_gem_resp = retry_api_call(
                        _b_gemini.models.generate_content,
                        model="gemini-2.5-flash",
                        contents=[
                            types.Part.from_bytes(data=_b_bytes, mime_type=_b_mime),
                            _b_analysis_prompt,
                        ],
                    )
                    _b_analysis_text = _b_gem_resp.text.strip()

                if _b_analysis_text is None:
                    raise Exception("Claude 與 Gemini 皆無法分析圖片")

                _btext = _b_analysis_text
                if "```json" in _btext:
                    _btext = _btext.split("```json")[1].split("```")[0].strip()
                elif "```" in _btext:
                    _btext = _btext.split("```")[1].split("```")[0].strip()
                _bres["prompts"] = json.loads(_btext)

                # ── Step 3：生成 8 張實穿照 ──
                _b_ref = types.Part.from_bytes(data=_b_bytes, mime_type=_b_mime)
                _b_base_prompt = _bres["prompts"]["positive_en"] + f", {_bp['outfit_desc_en']}"
                _b_neg = _bres["prompts"]["negative_en"]
                _b_shots = build_shot_configs()
                _b_images = []
                for _si, _bshot in enumerate(_b_shots):
                    _status_b.markdown(
                        f"**第 {_bidx+1}/{_total_b} 張** `{_bfile.name}` — 生成照片 {_si+1}/8…"
                    )
                    _b_hero_ref = None
                    if _b_images and _b_images[0].get("bytes"):
                        _b_hero_ref = types.Part.from_bytes(data=_b_images[0]["bytes"], mime_type="image/png")
                    _img_r = generate_single_photo(
                        _b_gk, _bshot, _b_base_prompt, _b_neg, _b_scene, _b_ref, _b_hero_ref
                    )
                    _b_images.append(_img_r)
                _bres["images"] = [_im for _im in _b_images if _im.get("bytes")]

                # ── Step 4：生成文案 ──
                _b_cap_content = []
                for _bim in _bres["images"][:8]:
                    _b_cap_content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png",
                                   "data": base64.standard_b64encode(_bim["bytes"]).decode("utf-8")},
                    })
                _b_extra = f"\n- 商品特色補充：{_batch_product_desc}" if _batch_product_desc else ""
                _b_cap_prompt = f"""你是一位專業的電商社群媒體文案師，擅長韓系時尚品牌的 Instagram 行銷。

請根據「附圖中的模特兒實穿照」撰寫一篇高互動率的 Instagram 貼文文案。
文案必須與照片中的場景、氛圍、穿搭情境完全吻合。

【照片場景資訊】
- 拍攝場景：{_bp['selected_scene']}
- 照片中的穿搭提示詞：{_bres['prompts'].get('positive_en', '')}

【商品與風格】
- 商品類型：韓系襪子（電商商品）
- 文案風格：{_batch_style}{_b_extra}
- 語言規範：{_b_lang_map.get(_batch_lang, _b_lang_map['繁體中文'])}

【撰寫要求】
1. 仔細觀察附圖中模特兒的姿勢、場景、光線、穿搭搭配
2. 文案需描述照片中的情境（例如：咖啡廳的午後、室內的慵懶時光等）
3. 將襪子自然融入穿搭場景的敘事中，不要只是單純介紹商品規格
4. 讓讀者看到文案就能聯想到照片中的畫面

請依照以下格式輸出，不要加其他說明：

【標題】
（1行，吸睛有力，與照片場景呼應，可含表情符號）

【貼文內容】
（4～6行，以照片場景為背景，自然口語化地帶出穿搭情境與商品亮點，含適量表情符號）

【Call to Action】
（1行，引導互動或購買）

【Hashtags】
（20～25個，分行整理，涵蓋：商品、穿搭、韓系、場景情境、季節、品味生活 等主題）
"""
                _b_cap_content.append({"type": "text", "text": _b_cap_prompt})
                _b_cap_resp = retry_api_call(
                    _b_claude.messages.create,
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": _b_cap_content}],
                )
                _bres["caption"] = _b_cap_resp.content[0].text

            except Exception as _be:
                _bres["error"] = str(_be)

            _batch_results_new.append(_bres)
            _prog_b.progress((_bidx + 1) / _total_b, text=f"第 {_bidx+1}/{_total_b} 張完成")

        _prog_b.progress(1.0, text="✅ 批次處理完成！")
        _status_b.markdown("✅ **批次處理全部完成！**")
        st.session_state.batch_results = _batch_results_new
        st.rerun()

    # 顯示批次結果
    if st.session_state.batch_results:
        st.markdown("### 📦 批次處理結果")
        for _bres in st.session_state.batch_results:
            _icon = "✅" if not _bres.get("error") else "❌"
            with st.expander(f"{_icon} {_bres['filename']}", expanded=False):
                if _bres.get("error"):
                    st.error(f"處理失敗：{_bres['error']}")
                else:
                    if _bres.get("images"):
                        _bi_cols = st.columns(min(len(_bres["images"]), 4))
                        for _ii, _bim in enumerate(_bres["images"]):
                            if _bim.get("bytes"):
                                with _bi_cols[_ii % 4]:
                                    st.image(_bim["bytes"], caption=_bim["label"], use_container_width=True)
                                    st.download_button(
                                        "💾 下載",
                                        data=_bim["bytes"],
                                        file_name=f"{_bres['filename'].rsplit('.', 1)[0]}_{_ii+1}.png",
                                        mime="image/png",
                                        key=f"dl_batch_{_bres['filename']}_{_ii}",
                                        use_container_width=True,
                                    )
                    if _bres.get("caption"):
                        st.text_area(
                            "📝 文案",
                            value=_bres["caption"],
                            height=200,
                            key=f"cap_batch_{_bres['filename']}",
                        )
                        st.download_button(
                            "📋 下載文案 .txt",
                            data=_bres["caption"].encode("utf-8"),
                            file_name=f"{_bres['filename'].rsplit('.', 1)[0]}_caption.txt",
                            mime="text/plain",
                            key=f"dl_cap_batch_{_bres['filename']}",
                        )

    st.divider()

# ─────────────────────────────────────────
# STEP 5：生成穿搭短影音（Kling 3.0）
# ─────────────────────────────────────────
def _kling_jwt(ak, sk):
    """生成 Kling AI JWT Token"""
    import jwt as pyjwt
    import time as _t
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(_t.time()) + 1800,
        "nbf": int(_t.time()) - 5,
    }
    return pyjwt.encode(payload, sk, algorithm="HS256", headers=headers)

if show_step5:
  st.markdown('<div class="step-header">Step 5 · 🎬 生成穿搭短影音（Kling 3.0）</div>', unsafe_allow_html=True)

  if not kling_ak or not kling_sk:
    st.warning("請先在左側 Sidebar 展開「🎬 Kling AI Key」輸入 Access Key 和 Secret Key")
    st.markdown(
        '<div class="info-box">💡 前往 <a href="https://kling.ai/dev/resource-pack-manage" target="_blank">kling.ai/dev/resource-pack-manage</a> 取得 API Key</div>',
        unsafe_allow_html=True,
    )
  else:
    # ── 雙模式選擇器 ──
    _s5_pipeline_imgs = [i for i in (st.session_state.model_images or []) if i.get("bytes")]
    _s5_has_pipeline = len(_s5_pipeline_imgs) > 0

    if _s5_has_pipeline and _is_full:
        _s5_source = st.radio(
            "📷 圖片來源",
            ["使用流程中的實穿照", "直接上傳圖片"],
            key="s5_img_source",
            horizontal=True,
        )
    elif not _is_full:
        _s5_source = "直接上傳圖片"
        st.info("📷 獨立模式：請上傳要作為影片起始畫面的照片。")
    else:
        _s5_source = "直接上傳圖片"

    # 準備圖片
    successful_imgs = []
    selected_img_data = None

    if _s5_source == "使用流程中的實穿照":
        successful_imgs = _s5_pipeline_imgs
        if not successful_imgs:
            st.warning("沒有可用的實穿照片，請先在 Step 3 生成或切換為「直接上傳圖片」")
        else:
            st.markdown("從實穿照片中選擇**起始畫面**及**參考照片**，Kling 3.0 會自動生成動態影片。")
            img_labels = [img["label"] for img in successful_imgs]
            selected_img_label = st.selectbox("📷 選擇起始畫面（API 主參考圖）", img_labels, key="video_source_img")
            selected_img_data = next(i for i in successful_imgs if i["label"] == selected_img_label)
            preview_img = Image.open(io.BytesIO(selected_img_data["bytes"]))
            st.image(preview_img, caption=f"起始畫面：{selected_img_label}", width=300)

            st.markdown("**參考照片**：選擇要加入 prompt 強調的照片（Kling API 目前僅支援單張主圖，其餘以 prompt 描述補強）")
            ref_img_labels = st.multiselect(
                "📸 選擇參考照片（可多選，預設全選）",
                img_labels,
                default=img_labels,
                key="video_ref_imgs",
                help="所選照片的數量與樣式特徵會加入 prompt，幫助 Kling 保持更準確的襪子圖案。",
            )
    else:
        _s5_upload = st.file_uploader(
            "上傳起始畫面照片（單張）",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="s5_upload_img",
        )
        if _s5_upload:
            _s5_upload.seek(0)
            _s5_upload_bytes = _s5_upload.read()
            selected_img_data = {"bytes": _s5_upload_bytes, "label": _s5_upload.name}
            successful_imgs = [selected_img_data]
            preview_img = Image.open(io.BytesIO(_s5_upload_bytes))
            st.image(preview_img, caption=f"起始畫面：{_s5_upload.name}", width=300)
            ref_img_labels = []

    try:
        ref_img_count = len(ref_img_labels)
    except NameError:
        ref_img_count = 0

    if selected_img_data:
      # 影片設定
      col_ratio, col_dur = st.columns(2)
      with col_ratio:
          video_ratio = st.selectbox("📐 影片比例", ["9:16（直式 Reels）", "16:9（橫式）", "1:1（正方形）"], key="video_ratio")
      with col_dur:
          video_duration = st.selectbox("⏱️ 影片長度", ["5 秒", "10 秒", "15 秒"], key="video_duration")

      col_mode, col_sound = st.columns(2)
      with col_mode:
          video_mode = st.selectbox("🎬 畫質模式", ["std（標準）", "pro（專業）"], key="video_mode")
      with col_sound:
          video_sound = st.selectbox("🔊 音效", ["on（開啟）", "off（關閉）"], key="video_sound")

      col_bgm, col_vol = st.columns(2)
      with col_bgm:
          add_bgm = st.checkbox(
              "🎵 疊加背景音樂（BGM）",
              value=True,
              key="add_bgm",
              help="影片生成後，用程式合成 C 大調和弦環境音樂疊入影片（需安裝 moviepy）。",
          )
      with col_vol:
          bgm_volume = st.slider(
              "🔉 BGM 音量",
              min_value=0.05,
              max_value=1.0,
              value=0.25,
              step=0.05,
              key="bgm_volume",
              disabled=not add_bgm,
          )

      # 從 Step 2 分析結果擷取襪子圖案描述
      _sock_desc_hint = ""
      if st.session_state.get("prompts", {}).get("positive_en"):
          _sock_desc_hint = (
              f" The socks in this video have the following characteristics: "
              f"{st.session_state.prompts['positive_en'][:300]}."
          )

      # 多照片參考備注
      _ref_note = (
          f" This video references {ref_img_count} styled photos — ensure the sock pattern is consistent across all of them."
          if ref_img_count > 1 else ""
      )

      # ── 影片風格選擇器 ──
      _style_map = {
          "自動匹配": None,
          "日系甜美": "A",
          "韓系街拍": "B",
          "咖啡廳慵懶": "C",
          "校園青春": "D",
      }
      selected_style_label = st.selectbox(
          "🎬 影片風格",
          list(_style_map.keys()),
          key="video_style_select",
          help="自動匹配會根據 Step 2 選擇的場景決定風格；也可手動指定。",
      )
      _chosen_tmpl_key = _style_map[selected_style_label]
      if _chosen_tmpl_key is None:
          _scene_key = st.session_state.get("selected_scene") or ""
          _chosen_tmpl_key = _match_video_prompt_template(_scene_key)
          _style_display = f"自動匹配 → {VIDEO_PROMPT_TEMPLATES[_chosen_tmpl_key]['name']}"
      else:
          _style_display = VIDEO_PROMPT_TEMPLATES[_chosen_tmpl_key]["name"]
      st.caption(f"目前風格：**{_style_display}**")

      # 組合 prompt：模板動態段 + 固定準確性後綴 + 圖案描述 + 多照參考備注
      video_prompt_default = (
          VIDEO_PROMPT_TEMPLATES[_chosen_tmpl_key]["prompt"]
          + _VIDEO_PROMPT_FIXED_SUFFIX
          + _sock_desc_hint
          + _ref_note
      )

      # 當風格變更時，自動更新 text_area 內容（覆蓋手動編輯）
      _prev_tmpl_key = st.session_state.get("_prev_video_tmpl_key")
      if _prev_tmpl_key != _chosen_tmpl_key:
          st.session_state["video_prompt_input"] = video_prompt_default
          st.session_state["_prev_video_tmpl_key"] = _chosen_tmpl_key

      video_prompt = st.text_area(
          "🎬 影片動態描述（可自訂）",
          value=video_prompt_default,
          height=140,
          key="video_prompt_input",
          help="根據選擇的風格自動填入，仍可手動編輯。影片會以選擇的照片為起始畫面。"
      )

      if st.button("🎬 生成穿搭短影音", type="primary", use_container_width=False):
          with st.spinner("🎬 Kling 3.0 正在生成影片，約需 2～5 分鐘，請耐心等待…"):
              try:
                  import time as _time
                  import requests as _requests

                  # 解析設定
                  aspect = "9:16" if "9:16" in video_ratio else ("1:1" if "1:1" in video_ratio else "16:9")
                  duration = 5 if video_duration.startswith("5") else (15 if video_duration.startswith("15") else 10)
                  mode = "std" if "std" in video_mode else "pro"
                  sound = "on" if "on" in video_sound else "off"

                  # 將圖片轉為 Base64
                  img_b64 = base64.standard_b64encode(selected_img_data["bytes"]).decode("utf-8")

                  # 生成 JWT Token
                  token = _kling_jwt(kling_ak, kling_sk)

                  # 發起 image-to-video 任務（依序嘗試多個區域端點）
                  _kling_endpoints = [
                      "https://api-global.klingai.com",
                      "https://api-singapore.klingai.com",
                      "https://api-beijing.klingai.com",
                  ]
                  create_data = None
                  _kling_last_err = ""
                  for _ep in _kling_endpoints:
                      try:
                          create_resp = _requests.post(
                              f"{_ep}/v1/videos/image2video",
                              headers={
                                  "Authorization": f"Bearer {token}",
                                  "Content-Type": "application/json",
                              },
                              json={
                                  "model_name": "kling-v3",
                                  "mode": mode,
                                  "duration": str(duration),
                                  "aspect_ratio": aspect,
                                  "image": img_b64,
                                  "prompt": video_prompt,
                                  "sound": sound,
                              },
                              timeout=30,
                          )
                          if create_resp.status_code == 200 and create_resp.text.strip():
                              create_data = create_resp.json()
                              KLING_BASE = _ep
                              break
                          else:
                              _kling_last_err = f"{_ep} → HTTP {create_resp.status_code}, body={create_resp.text[:200]}"
                      except Exception as _ep_err:
                          _kling_last_err = f"{_ep} → {_ep_err}"
                          continue

                  if create_data is None:
                      st.error(f"❌ 所有 Kling API 端點均無法連線：{_kling_last_err}")
                  elif create_data.get("code") != 0:
                      st.error(f"❌ 任務建立失敗：{create_data.get('message', create_data)}")
                  else:
                      task_id = create_data["data"]["task_id"]
                      st.info(f"📋 任務已建立，Task ID: `{task_id}`")

                      # 輪詢等待完成
                      progress = st.progress(0, text="影片生成中…")
                      poll_count = 0
                      max_polls = 60  # 最多等 10 分鐘

                      video_url = None
                      while poll_count < max_polls:
                          poll_count += 1
                          progress.progress(
                              min(poll_count / max_polls, 0.95),
                              text=f"影片生成中… 已等待 {poll_count * 10} 秒"
                          )
                          _time.sleep(10)

                          # 重新生成 token（避免過期）
                          token = _kling_jwt(kling_ak, kling_sk)
                          query_resp = _requests.get(
                              f"{KLING_BASE}/v1/videos/image2video/{task_id}",
                              headers={"Authorization": f"Bearer {token}"},
                              timeout=30,
                          )
                          if not query_resp.text.strip():
                              continue  # 空回應，重試
                          query_data = query_resp.json()

                          if query_data.get("code") != 0:
                              st.error(f"❌ 查詢失敗：{query_data.get('message', query_data)}")
                              break

                          task_status = query_data["data"]["task_status"]
                          if task_status == "succeed":
                              videos = query_data["data"].get("task_result", {}).get("videos", [])
                              if videos:
                                  video_url = videos[0].get("url")
                              break
                          elif task_status == "failed":
                              fail_reason = query_data["data"].get("task_status_msg", "未知原因")
                              st.error(f"❌ 影片生成失敗：{fail_reason}")
                              break

                      if video_url:
                          progress.progress(1.0, text="✅ 影片生成完成！正在下載…")
                          # 下載影片
                          vid_resp = _requests.get(video_url, timeout=120)
                          vid_resp.raise_for_status()
                          raw_video = vid_resp.content

                          # 疊加背景音樂（BGM）
                          if add_bgm:
                              with st.spinner("🎵 疊加背景音樂中，請稍候…"):
                                  raw_video = _mix_bgm_into_video(raw_video, bgm_volume)

                          st.session_state.video_bytes = raw_video
                          st.success(f"✅ 穿搭短影音生成成功！（Kling 3.0 · {mode} · {duration}s{' · 含 BGM' if add_bgm else ''}）")
                          _c5 = _cost_kling(duration, mode)
                          st.session_state.cost_step5 = _c5
                          st.info(f"💰 本步驟花費：${_c5:.4f}（Kling v3 · {mode} · {duration} 秒，預估值）")
                      elif task_status != "failed":
                          st.error("❌ 影片生成超時（已等待 10 分鐘），請稍後再試。")

              except Exception as e:
                  import traceback
                  st.error(f"❌ 影片生成失敗：{e}")
                  st.code(traceback.format_exc(), language="text")
                  st.markdown(
                      '<div class="info-box">💡 提示：請確認 Kling AI API Key 正確且有足夠配額。'
                      '前往 <a href="https://kling.ai/dev/resource-pack-manage" target="_blank">kling.ai/dev/resource-pack-manage</a> 查看。</div>',
                      unsafe_allow_html=True,
                  )

# 顯示已生成的影片
if show_step5 and st.session_state.video_bytes:
    st.markdown("### 🎥 穿搭短影音預覽")
    st.video(st.session_state.video_bytes, format="video/mp4")

    col_vdl1, col_vdl2 = st.columns(2)
    with col_vdl1:
        st.download_button(
            label="💾 下載影片 MP4",
            data=st.session_state.video_bytes,
            file_name="sock_styling_reel.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    with col_vdl2:
        st.markdown(f"**影片資訊**：`{len(st.session_state.video_bytes)/1024/1024:.1f} MB`")

if _is_full:
    st.divider()

# ─────────────────────────────────────────
# STEP 6：電商首圖製作（實穿照 + 商品去背圖 + 行銷文字）
# ─────────────────────────────────────────
# Session state 初始化
if "hero_banner_bytes" not in st.session_state:
    st.session_state.hero_banner_bytes = None
if "cost_step6" not in st.session_state:
    st.session_state.cost_step6 = 0.0

if show_step6:
  st.markdown('<div class="step-header">Step 6 · 🏷️ 電商首圖製作（實穿照 + 商品圖 + 行銷文字）</div>', unsafe_allow_html=True)

  if not api_key:
    st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
  else:
    st.markdown(
        '<div class="info-box">💡 選擇一張模特兒實穿照作為主視覺，搭配各顏色款式的商品去背圖，'
        'AI 會將行銷文字與所有素材合成為專業電商首圖。</div>',
        unsafe_allow_html=True,
    )

    # ── 區塊 A：選擇模特兒實穿照 ──
    st.markdown("#### 📷 主視覺：模特兒實穿照")
    _s6_model_imgs = [i for i in (st.session_state.model_images or []) if i.get("bytes")]
    _s6_selected_model_bytes = None

    # 雙模式選擇器
    if _s6_model_imgs and _is_full:
        _s6_model_source = st.radio(
            "實穿照來源",
            ["使用流程中的實穿照", "直接上傳照片"],
            key="s6_model_source",
            horizontal=True,
        )
    elif not _is_full:
        _s6_model_source = "直接上傳照片"
        st.info("📷 獨立模式：請上傳模特兒實穿照或商品照片。")
    else:
        _s6_model_source = "直接上傳照片"

    if _s6_model_source == "使用流程中的實穿照" and _s6_model_imgs:
        _s6_cols_per_row = min(len(_s6_model_imgs), 4)
        _s6_cols = st.columns(_s6_cols_per_row)
        for idx, img_data in enumerate(_s6_model_imgs):
            with _s6_cols[idx % _s6_cols_per_row]:
                st.image(img_data["bytes"], caption=img_data["label"], width=160)
        _s6_model_labels = [img["label"] for img in _s6_model_imgs]
        _s6_selected_label = st.radio(
            "選擇一張實穿照作為首圖主視覺",
            _s6_model_labels,
            horizontal=True,
            key="s6_model_select",
        )
        _s6_selected_model = next(i for i in _s6_model_imgs if i["label"] == _s6_selected_label)
        _s6_selected_model_bytes = _s6_selected_model["bytes"]
    else:
        _s6_model_upload = st.file_uploader(
            "上傳模特兒實穿照（選填，單張）",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="s6_model_upload",
            help="上傳一張模特兒實穿照作為首圖主視覺；也可跳過，只使用商品去背圖",
        )
        if _s6_model_upload:
            _s6_model_upload.seek(0)
            _s6_selected_model_bytes = _s6_model_upload.read()
            st.image(_s6_selected_model_bytes, caption="已上傳的主視覺照片", width=200)

    st.markdown("---")

    # ── 區塊 B：上傳商品去背圖（多張，各顏色款式）──
    st.markdown("#### 🎨 商品去背圖（各顏色/款式白底圖）")
    st.caption("上傳所有顏色款式的商品去背圖，AI 會將它們排列在首圖中展示。")
    _s6_product_uploads = st.file_uploader(
        "上傳商品去背圖（可多選）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="s6_product_uploads",
        help="建議上傳白底去背的商品圖，每個顏色/款式各一張",
    )

    # 預覽已上傳的去背圖
    if _s6_product_uploads:
        st.markdown(f"已上傳 **{len(_s6_product_uploads)}** 張商品去背圖：")
        _s6_preview_cols = st.columns(min(len(_s6_product_uploads), 6))
        for idx, f in enumerate(_s6_product_uploads):
            with _s6_preview_cols[idx % len(_s6_preview_cols)]:
                st.image(f, width=120)

    st.markdown("---")

    # ── 區塊 C：行銷文字與設計選項 ──
    col_text, col_opts = st.columns([2, 1])
    with col_text:
        hero_text = st.text_area(
            "✏️ 圖片顯示文字（選填）",
            placeholder="例如：日本製純棉中筒襪\n透氣舒適 百搭必備\n限時特價 NT$199\n\n💡 留空時 AI 會自動根據商品圖片發想標題與文案",
            height=100,
            key="hero_banner_text",
            help="這裡輸入的文字會直接顯示在首圖上；留空的話，AI 會自動發想文案",
        )
        hero_note = st.text_area(
            "📝 補充說明（不顯示在圖片上）",
            placeholder="例如：這款是針織材質不是刺繡、主打夏季透氣、目標客群是 20-35 歲女性…",
            height=80,
            key="hero_banner_note",
            help="這裡的內容只供 AI 參考，幫助它更了解商品特色，不會直接出現在圖片上",
        )
    with col_opts:
        hero_style = st.selectbox(
            "🎨 設計風格",
            [
                "簡約質感（乾淨俐落、高級感）",
                "活潑可愛（繽紛色彩、年輕活力）",
                "韓系文青（柔和色調、清新自然）",
                "時尚潮流（大膽對比、視覺衝擊）",
                "溫馨居家（暖色系、舒適療癒）",
            ],
            key="hero_banner_style",
        )
        hero_size = st.selectbox(
            "📐 圖片尺寸",
            ["1:1 正方形（電商主圖）", "4:3 橫式（網頁 Banner）", "3:4 直式（手機瀏覽）"],
            key="hero_banner_size",
        )
        hero_lang = st.selectbox(
            "🌐 文字語言",
            ["繁體中文", "English", "中英混合"],
            key="hero_banner_lang",
        )

    # 按鈕永遠顯示，點擊後驗證條件
    _has_any_image = bool(_s6_selected_model_bytes or _s6_product_uploads)
    if st.button("🏷️ 生成電商首圖", type="primary", use_container_width=False, key="btn_hero_banner"):
        if not _has_any_image:
            st.warning("⚠️ 請先選擇一張實穿照，或上傳至少一張商品去背圖。")
        else:
            _s6_status = st.status("🏷️ 電商首圖生成中…", expanded=True)
            try:
                _s6_status.write("📋 準備素材中…")
                _s6_img_count = (1 if _s6_selected_model_bytes else 0) + len(_s6_product_uploads or [])
                _s6_text_mode = "使用者提供的行銷文字" if hero_text.strip() else "AI 自動發想文案"
                _s6_status.write(f"共 {_s6_img_count} 張圖片 · {_s6_text_mode}")

                gemini_client = genai.Client(api_key=api_key)

                # 設計風格對應的英文描述
                style_map = {
                    "簡約質感（乾淨俐落、高級感）": "minimalist, clean, premium luxury feel, elegant typography, muted refined color palette",
                    "活潑可愛（繽紛色彩、年輕活力）": "playful, colorful, youthful energy, fun typography with rounded edges, vibrant candy colors",
                    "韓系文青（柔和色調、清新自然）": "Korean aesthetic, soft pastel tones, fresh and natural, delicate serif or sans-serif font, dreamy atmosphere",
                    "時尚潮流（大膽對比、視覺衝擊）": "bold fashion-forward, high contrast colors, impactful typography, edgy modern design, street style vibe",
                    "溫馨居家（暖色系、舒適療癒）": "warm cozy homey feel, warm earth tones, soft rounded typography, comfortable and healing atmosphere",
                }
                style_desc = style_map.get(hero_style, "clean modern e-commerce style")

                # 語言指令
                lang_map = {
                    "繁體中文": "All text on the image MUST be in Traditional Chinese (繁體中文). Use Traditional Chinese characters only.",
                    "English": "All text on the image must be in English.",
                    "中英混合": "Main headline in Traditional Chinese (繁體中文), with supplementary English text for style/branding.",
                }
                lang_instruction = lang_map.get(hero_lang, lang_map["繁體中文"])

                # 尺寸設定
                size_instruction = "1:1 square format (1024x1024)"
                if "4:3" in hero_size:
                    size_instruction = "4:3 landscape format (1024x768)"
                elif "3:4" in hero_size:
                    size_instruction = "3:4 portrait format (768x1024)"

                # 動態組合 prompt 素材描述
                image_desc_parts = []
                if _s6_selected_model_bytes and _s6_product_uploads:
                    image_desc_parts.append(
                        "IMAGE 1 is a model wearing the product (use as the main visual / hero shot)."
                    )
                    for i in range(len(_s6_product_uploads)):
                        image_desc_parts.append(
                            f"IMAGE {i+2} is a product cutout photo on white background (color variant {i+1})."
                        )
                    composition_instruction = (
                        "Place the model photo as the main visual (hero shot) — it should be the largest and most prominent element. "
                        "Arrange the product cutout photos in a neat row or grid nearby (smaller size) to showcase all available color options. "
                        "Each color variant can be labeled with a small color dot or text tag. "
                        "Add a 'COLOR CHOICE' or '顏色選擇' section label above the product cutouts."
                    )
                elif _s6_selected_model_bytes:
                    image_desc_parts.append(
                        "IMAGE 1 is a model wearing the product (use as the main visual / hero shot)."
                    )
                    composition_instruction = (
                        "Use the model photo as the full main visual. "
                        "Overlay the marketing text in a balanced layout that complements the model pose."
                    )
                else:
                    for i in range(len(_s6_product_uploads)):
                        image_desc_parts.append(
                            f"IMAGE {i+1} is a product cutout photo on white background (color variant {i+1})."
                        )
                    composition_instruction = (
                        "Arrange all product cutout photos in an attractive layout to showcase the full color range. "
                        "Each color variant can be labeled with a small color dot or text tag."
                    )

                images_description = "\n".join(image_desc_parts)

                # 行銷文字區塊：有輸入就用，沒輸入讓 AI 自動發想
                if hero_text.strip():
                    text_block = f"""[MARKETING TEXT TO DISPLAY ON IMAGE]
{hero_text}"""
                else:
                    text_block = """[MARKETING TEXT — AUTO GENERATE]
The user did NOT provide specific marketing text.
You MUST analyze the product images and automatically create compelling marketing copy, including:
- A catchy headline that highlights the product's key feature or style
- 1-2 short sub-headlines about material, comfort, or unique selling points
- If multiple color variants are provided, add color labels for each variant
Write the text in a style suitable for e-commerce product listing hero images."""

                # 補充說明區塊：僅供 AI 參考，不直接顯示在圖片上
                note_block = ""
                if hero_note.strip():
                    note_block = f"""
[ADDITIONAL CONTEXT — DO NOT display this text on the image]
The user provided the following notes about the product for your reference only.
Use this information to better understand the product and improve the marketing text/design,
but do NOT put this text directly on the image:
{hero_note}"""

                hero_banner_prompt = f"""You are a professional e-commerce graphic designer and marketing consultant.
Your task is to create an attractive e-commerce hero/listing image by compositing the provided images with marketing text.

[UPLOADED IMAGES]
{images_description}

[COMPOSITION]
{composition_instruction}

{text_block}
{note_block}

[LANGUAGE REQUIREMENT]
{lang_instruction}

[DESIGN GUIDELINES]
1. ANALYZE all uploaded images: identify color palettes, styles, and how they relate to each other.
2. OPTIMIZE the marketing text: refine wording to be compelling and concise for e-commerce. If no text was provided, create original marketing copy based on the product images.
3. LAYOUT: Create a professional e-commerce listing image composition. The model photo (if provided) is the hero visual. Product cutouts (if provided) should be arranged neatly to show all available options.
4. TYPOGRAPHY: Use {style_desc}. Choose font styles that harmonize with the overall mood.
5. COLOR: Select text colors that contrast well for readability while maintaining aesthetic harmony with the product colors.
6. HIERARCHY: Main headline large and prominent. Sub-text (price, features) smaller. Color variant labels smallest.
7. FORMAT: {size_instruction}
8. For each color variant, use a small colored circle/dot indicator or a concise text label (e.g., ▼黑底, ▼粉底) to identify each option.

[QUALITY STANDARDS]
- Professional e-commerce platform listing quality
- Clean, organized layout with clear visual hierarchy
- Text must be crisp, legible, and well-placed
- Product images must not be distorted or cropped awkwardly
- Readability on both desktop and mobile screens

[AVOID]
- Text covering important product details
- Cluttered or busy layouts
- Hard-to-read color combinations
- Distorting product images
- Watermarks or placeholder text
- Overlapping product cutout images"""

                # 組合所有圖片 + prompt
                _s6_status.write("🖼️ 上傳圖片到 Gemini API…")
                content_parts = []
                if _s6_selected_model_bytes:
                    content_parts.append(
                        types.Part.from_bytes(data=_s6_selected_model_bytes, mime_type="image/png")
                    )
                    _s6_status.write("  ✅ 實穿照已加入")
                if _s6_product_uploads:
                    for f in _s6_product_uploads:
                        f.seek(0)
                        _f_bytes = f.read()
                        _f_mime = getattr(f, "type", "image/jpeg") or "image/jpeg"
                        content_parts.append(
                            types.Part.from_bytes(data=_f_bytes, mime_type=_f_mime)
                        )
                    _s6_status.write(f"  ✅ {len(_s6_product_uploads)} 張去背圖已加入")
                content_parts.append(hero_banner_prompt)

                _s6_status.write("🤖 Gemini 正在生成首圖（通常需要 30～90 秒）…")
                _s6_start_time = _time.time()

                response = retry_api_call(
                    gemini_client.models.generate_content,
                    model="gemini-3.1-flash-image-preview",
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=types.ImageConfig(image_size="1K"),
                    ),
                )

                _s6_elapsed = _time.time() - _s6_start_time
                _s6_status.write(f"📡 API 回應完成（耗時 {_s6_elapsed:.1f} 秒），正在處理結果…")

                banner_bytes = None
                _s6_text_response = ""
                candidates = getattr(response, "candidates", None)
                if candidates and len(candidates) > 0:
                    content = getattr(candidates[0], "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts:
                        for part in parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                banner_bytes = part.inline_data.data
                            elif hasattr(part, "text") and part.text:
                                _s6_text_response += part.text
                    else:
                        _s6_status.write("⚠️ API 回應中沒有 parts")
                else:
                    # 顯示完整回應以便 debug
                    _s6_finish_reason = ""
                    if candidates and len(candidates) > 0:
                        _s6_finish_reason = getattr(candidates[0], "finish_reason", "unknown")
                    _s6_status.write(f"⚠️ API 回應異常 — candidates 數量: {len(candidates) if candidates else 0}, finish_reason: {_s6_finish_reason}")

                if banner_bytes:
                    st.session_state.hero_banner_bytes = banner_bytes
                    _c6 = _cost_gemini_images(1)
                    st.session_state.cost_step6 = _c6
                    _s6_status.update(label="✅ 電商首圖生成成功！", state="complete", expanded=False)
                    st.success("✅ 電商首圖生成成功！")
                    st.info(f"💰 本步驟花費：${_c6:.4f}（Gemini 圖片生成 × 1，耗時 {_s6_elapsed:.1f} 秒）")
                    if _s6_text_response:
                        with st.expander("💬 Gemini 設計說明", expanded=False):
                            st.markdown(_s6_text_response)
                else:
                    _s6_status.update(label="❌ 首圖生成失敗", state="error", expanded=True)
                    st.error("❌ 未收到生成的圖片，請重試。")
                    if _s6_text_response:
                        st.warning(f"Gemini 回應文字（可能包含錯誤訊息）：\n\n{_s6_text_response}")

            except Exception as e:
                import traceback
                _s6_status.update(label="❌ 首圖生成失敗", state="error", expanded=True)
                st.error(f"❌ 首圖生成失敗：{e}")
                st.code(traceback.format_exc(), language="text")

# 顯示已生成的首圖
if show_step6 and st.session_state.hero_banner_bytes:
    st.markdown("### 🏷️ 電商首圖預覽")
    banner_img = Image.open(io.BytesIO(st.session_state.hero_banner_bytes))
    st.image(banner_img, caption="AI 生成的電商首圖", use_container_width=True)

    col_bd1, col_bd2 = st.columns(2)
    with col_bd1:
        st.download_button(
            label="💾 下載首圖 PNG",
            data=st.session_state.hero_banner_bytes,
            file_name="ecommerce_hero_image.png",
            mime="image/png",
            use_container_width=True,
            key="dl_hero_banner",
        )
    with col_bd2:
        _banner_size_mb = len(st.session_state.hero_banner_bytes) / 1024 / 1024
        st.markdown(f"**圖片大小**：`{_banner_size_mb:.1f} MB` · `{banner_img.width}×{banner_img.height}`")

    if st.button("🔄 重新生成", key="btn_hero_banner_retry"):
        st.session_state.hero_banner_bytes = None
        st.rerun()

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("🧦 AI 電商素材生成器 · Powered by Google Gemini + Anthropic Claude + Veo 3.1 · Built with Streamlit")
