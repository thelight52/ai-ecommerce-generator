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
import pathlib

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

    if api_key and anthropic_key:
        st.markdown('<div class="success-box">✅ 兩組 API Key 已設定</div>', unsafe_allow_html=True)
    else:
        missing = []
        if not api_key: missing.append("Gemini")
        if not anthropic_key: missing.append("Anthropic")
        st.markdown(
            f'<div class="info-box">⚠️ 尚未設定：{", ".join(missing)} API Key</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📋 流程說明")
    st.markdown("""
1. 🖼️ **上傳**平拍照
2. 🔍 **分析**自動產出提示詞
3. 🎨 **生成**模特兒實穿照
4. ✍️ **生成**社群貼文文案
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
for key in ["prompts", "model_image_bytes", "model_images", "captions", "upload_mime", "selected_scene"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "model_images" not in st.session_state or st.session_state.model_images is None:
    st.session_state.model_images = []

# ─────────────────────────────────────────
# 主標題
# ─────────────────────────────────────────
st.title("🧦 AI 電商素材生成器")
st.caption("以 Google AI Studio (Gemini) 為核心，從平拍照一鍵產出模特兒實穿照與社群文案")
st.divider()

# ─────────────────────────────────────────
# STEP 1：上傳平拍照
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 1 · 🖼️ 上傳商品平拍照</div>', unsafe_allow_html=True)

# 範例圖片檔案路徑
SAMPLE_IMAGE_PATH = pathlib.Path(__file__).parent / "test_product.jpg"

uploaded_file = st.file_uploader(
    "選擇商品平拍圖片（支援 JPG / PNG / WEBP）",
    type=["jpg", "jpeg", "png", "webp"],
    help="建議使用清晰、白底或淺色背景的商品平拍照"
)

# MockUploadedFile：模擬 Streamlit UploadedFile 介面
class _MockFile:
    def __init__(self, data, name, mime_type):
        self._data = data
        self.name = name
        self.type = mime_type
        self.size = len(data)
    def read(self): return self._data
    def getvalue(self): return self._data
    def seek(self, pos): pass

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
        st.image(uploaded_file.getvalue(), caption="📸 已上傳的平拍照", use_container_width=True)
    with col_info:
        st.markdown('<div class="success-box">✅ 圖片上傳成功，可繼續下一步</div>', unsafe_allow_html=True)
        st.markdown(f"- **檔名**：`{uploaded_file.name}`")
        st.markdown(f"- **大小**：`{uploaded_file.size / 1024:.1f} KB`")
        st.markdown(f"- **格式**：`{uploaded_file.type}`")

st.divider()

# ─────────────────────────────────────────
# STEP 2：自動產出提示詞
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 2 · 🔍 自動分析並產出提示詞</div>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("請先完成 Step 1 上傳圖片")
elif not anthropic_key:
    st.warning("請先在左側 Sidebar 輸入 Anthropic API Key")
else:
    # ── 場景選擇（原本在 Step 3，移至此處） ──
    scene_options = {
        "簡約室內（白色大理石地板）": "sitting on white marble floor, clean minimal indoor background, soft natural window light, warm white tones",
        "咖啡廳外拍（暖陽散景）": "outdoor cafe setting, warm golden afternoon sunlight, blurred bokeh background, film photography aesthetic",
        "清爽白背景（電商主圖）": "pure white studio background, soft diffused studio lighting, bright and airy atmosphere, e-commerce hero shot",
    }
    selected_scene = st.selectbox("🏠 選擇場景", list(scene_options.keys()))
    st.session_state.selected_scene = selected_scene

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
    scene_desc = scene_options[selected_scene]
    sock_type_zh = sock_type.split(" — ")[0]

    if st.button("🔍 分析圖片並自動產出提示詞", type="primary", use_container_width=False):
        with st.spinner("Claude 正在分析商品圖片，自動生成提示詞…"):
            try:
                claude_client = anthropic.Anthropic(api_key=anthropic_key)
                img_bytes = uploaded_file.getvalue()
                img_base64 = base64.standard_b64encode(img_bytes).decode("utf-8")
                mime_type = st.session_state.upload_mime or "image/jpeg"

                analysis_prompt = f"""You are a professional e-commerce fashion photographer and AI image prompt engineer specializing in Korean style.

Analyze this product flat lay image carefully and generate AI image generation prompts for a Korean female model wearing this product.

PRODUCT INFO (MUST appear in the generated prompts):
- Sock type: {sock_info_en}{sock_length_desc}
- Scene / Background: {scene_desc}

IMPORTANT RULES:
- The generated positive_en prompt MUST explicitly contain these exact details:
  1. The sock type: "{sock_info_en}"
  2. The sock length: "{sock_length if sock_length else 'not specified'}" (include exact measurement if provided)
  3. The scene description keywords from: "{scene_desc}"
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
  "positive_en": "Korean female model, slim legs, wearing {sock_info_en}{sock_length_desc}, [pose details], lower body shot from waist down, [outfit pairing], {scene_desc}, [lighting], [photography quality]",
  "positive_zh": "韓系女性模特兒，穿著{sock_type_zh}{sock_length_zh}，[姿勢細節]，腰部以下畫面，[服裝搭配]，[場景]，[光線]，[攝影質感]",
  "negative_en": "full body, face visible, upper body dominant, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, missing product, duplicate body parts, bad anatomy, extra fingers, nsfw, wrong sock length, barefoot"
}}"""

                response = claude_client.messages.create(
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

                text = response.content[0].text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                st.session_state.prompts = json.loads(text)
                st.success("✅ 提示詞已自動生成！可在下方編輯後再生成圖片。")

            except json.JSONDecodeError:
                st.session_state.prompts = {
                    "positive_en": response.text,
                    "positive_zh": "",
                    "negative_en": "full body, face visible, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, duplicate body parts, bad anatomy, extra fingers",
                }
                st.warning("⚠️ JSON 解析失敗，原始回覆已填入正向提示詞欄位，請手動調整。")
            except Exception as e:
                st.error(f"❌ 分析失敗：{e}")

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
# STEP 3：生成模特兒實穿照（5 張照片組）
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 3 · 🎨 生成模特兒實穿照組（5 張）</div>', unsafe_allow_html=True)

# 5 張照片的拍攝角度定義
SHOT_CONFIGS = [
    {
        "label": "📷 全身照 ①（正面站姿）",
        "shot_desc": "Full body shot, Korean female model standing naturally, front-facing with slight hip tilt, head to toe visible, model's face partially visible from nose down, elegant standing pose showcasing the complete outfit and socks coordination",
    },
    {
        "label": "📷 全身照 ②（側面走路）",
        "shot_desc": "Full body shot, Korean female model walking or mid-stride, captured from side angle, head to toe visible, model's face partially visible from nose down, dynamic motion showing how the socks look during movement, natural and candid editorial feel",
    },
    {
        "label": "🦵 下半身特寫 ①（坐姿）",
        "shot_desc": "Lower body shot from waist down only, model seated on chair or stool with legs crossed, focus on the socks and leg styling, close-up showing fabric texture and pattern detail, no face visible",
    },
    {
        "label": "🦶 腳部特寫 ②（站姿腳部）",
        "shot_desc": "Close-up shot of feet and lower legs only, model standing on tiptoe or casual stance, camera angle from knee down, detailed view of sock pattern and fit on the feet and ankles, sharp macro-level fabric texture",
    },
    {
        "label": "🦵 下半身特寫 ③（生活情境）",
        "shot_desc": "Lower body lifestyle shot from waist down, model in relaxed casual pose like sitting on floor or leaning against wall, socks clearly visible as the hero of the composition, cozy and relatable everyday moment",
    },
]

if not st.session_state.prompts:
    st.info("請先完成 Step 2 產出提示詞")
elif not api_key:
    st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
else:
    # 顯示 Step 2 選擇的場景（唯讀提示）
    if st.session_state.selected_scene:
        st.info(f"🏠 使用場景：**{st.session_state.selected_scene}**（可在 Step 2 更換）")

    st.markdown("將一次生成 **5 張連貫的照片組**：2 張全身照 + 3 張腳部 / 下半身特寫")

    if st.button("🎨 生成 5 張模特兒實穿照組", type="primary", use_container_width=False):
        client = genai.Client(api_key=api_key)

        # 從 Step 2 取場景描述
        scene_options_map = {
            "簡約室內（白色大理石地板）": "sitting on white marble floor, clean minimal indoor background, soft natural window light, warm white tones",
            "咖啡廳外拍（暖陽散景）": "outdoor cafe setting, warm golden afternoon sunlight, blurred bokeh background, film photography aesthetic",
            "清爽白背景（電商主圖）": "pure white studio background, soft diffused studio lighting, bright and airy atmosphere, e-commerce hero shot",
        }
        scene_desc = scene_options_map.get(
            st.session_state.selected_scene or "清爽白背景（電商主圖）",
            "pure white studio background, soft diffused studio lighting"
        )

        base_prompt = st.session_state.prompts["positive_en"]
        neg_prompt = st.session_state.prompts["negative_en"]

        # 準備上傳的原始商品圖片
        ref_part = None
        if uploaded_file:
            img_bytes = uploaded_file.getvalue()
            mime_type = st.session_state.upload_mime or "image/jpeg"
            ref_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

        generated_images = []
        progress_bar = st.progress(0, text="準備生成照片組…")

        for idx, shot in enumerate(SHOT_CONFIGS):
            progress_bar.progress(
                (idx) / len(SHOT_CONFIGS),
                text=f"正在生成 {shot['label']}（{idx+1}/5）…約需 30～60 秒"
            )

            generation_prompt = (
                f"Using the sock/stocking design shown in the reference image, "
                f"generate a photorealistic e-commerce model photo. "
                f"A Korean female model wearing these exact socks with the same pattern, color, and design. "
                f"Shot type: {shot['shot_desc']}. "
                f"Scene: {scene_desc}. "
                f"Style: {base_prompt}, "
                f"photorealistic, commercial e-commerce photography, 8K resolution, "
                f"sharp fabric texture, feminine and elegant, editorial fashion quality. "
                f"The socks must faithfully reproduce the pattern from the reference image. "
                f"Maintain visual consistency: same model body type, same outfit styling, same lighting across all shots. "
                f"Avoid: {neg_prompt}"
            )

            content_parts = []
            if ref_part:
                content_parts.append(ref_part)
            content_parts.append(generation_prompt)

            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-image-preview",
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                image_bytes = None
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_bytes = part.inline_data.data
                        break

                if image_bytes:
                    generated_images.append({"label": shot["label"], "bytes": image_bytes})
                else:
                    generated_images.append({"label": shot["label"], "bytes": None, "error": "未收到圖片"})

            except Exception as e:
                generated_images.append({"label": shot["label"], "bytes": None, "error": str(e)})

        progress_bar.progress(1.0, text="✅ 照片組生成完成！")
        st.session_state.model_images = generated_images
        # 向下相容：取第一張成功的圖片作為 model_image_bytes（給 Step 4 用）
        for img in generated_images:
            if img.get("bytes"):
                st.session_state.model_image_bytes = img["bytes"]
                break
        st.success(f"✅ 成功生成 {sum(1 for i in generated_images if i.get('bytes'))} / 5 張照片！")

# 顯示生成的照片組
if st.session_state.model_images:
    st.markdown("### 📸 實穿照片組")

    # 第一行：2 張全身照
    st.markdown("**👗 全身照**")
    full_cols = st.columns(2)
    for idx, img_data in enumerate(st.session_state.model_images[:2]):
        with full_cols[idx]:
            if img_data.get("bytes"):
                gen_img = Image.open(io.BytesIO(img_data["bytes"]))
                st.image(gen_img, caption=img_data["label"], use_container_width=True)
                st.download_button(
                    label=f"💾 下載",
                    data=img_data["bytes"],
                    file_name=f"model_fullbody_{idx+1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_full_{idx}",
                )
            else:
                st.error(f"❌ {img_data['label']}：{img_data.get('error', '生成失敗')}")

    # 第二行：3 張特寫
    st.markdown("**🦶 腳部 / 下半身特寫**")
    detail_cols = st.columns(3)
    for idx, img_data in enumerate(st.session_state.model_images[2:5]):
        with detail_cols[idx]:
            if img_data.get("bytes"):
                gen_img = Image.open(io.BytesIO(img_data["bytes"]))
                st.image(gen_img, caption=img_data["label"], use_container_width=True)
                st.download_button(
                    label=f"💾 下載",
                    data=img_data["bytes"],
                    file_name=f"model_detail_{idx+1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_detail_{idx}",
                )
            else:
                st.error(f"❌ {img_data['label']}：{img_data.get('error', '生成失敗')}")

    # 一鍵全部下載提示
    successful = [i for i in st.session_state.model_images if i.get("bytes")]
    if len(successful) > 1:
        st.markdown("---")
        st.markdown(f"📊 **圖片資訊**：共 {len(successful)} 張成功")
        for img_data in successful:
            img_info = Image.open(io.BytesIO(img_data["bytes"]))
            st.caption(f"  {img_data['label']}：{img_info.width}×{img_info.height}，{len(img_data['bytes'])/1024:.0f} KB")

st.divider()

# ─────────────────────────────────────────
# STEP 4：生成社群貼文文案
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 4 · ✍️ 生成 Instagram 社群貼文文案</div>', unsafe_allow_html=True)

if not anthropic_key:
    st.warning("請先在左側 Sidebar 輸入 Anthropic API Key")
else:
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

    # 判斷是否有實穿照可參考（多張或單張）
    successful_images = [i for i in (st.session_state.model_images or []) if i.get("bytes")]
    has_model_image = len(successful_images) > 0 or st.session_state.model_image_bytes is not None
    if not has_model_image:
        st.info("💡 建議先完成 Step 3 生成實穿照，文案將根據實穿照的場景情境撰寫更精準的內容。")

    if st.button("✍️ 生成社群貼文文案", type="primary", use_container_width=False):
        with st.spinner("Claude 正在根據實穿照撰寫情境文案…" if has_model_image else "Claude 正在撰寫文案…"):
            try:
                claude_client = anthropic.Anthropic(api_key=anthropic_key)

                lang_instruction = {
                    "繁體中文": "請全程使用繁體中文（台灣用語）撰寫，包含標題、內文與 hashtag。",
                    "繁體中文 + English": "標題與內文使用繁體中文，hashtag 中英混合。",
                    "English only": "Please write entirely in English including all hashtags.",
                }[caption_lang]

                extra = f"\n- 商品特色補充：{product_desc}" if product_desc else ""

                # 取得 Step 2 的場景與襪子資訊
                scene_name = st.session_state.selected_scene or "未指定"
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

                # 組合訊息內容：附上所有成功的實穿照（最多 5 張）
                message_content = []
                if successful_images:
                    for si in successful_images[:5]:
                        img_b64 = base64.standard_b64encode(si["bytes"]).decode("utf-8")
                        message_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        })
                elif st.session_state.model_image_bytes:
                    img_b64 = base64.standard_b64encode(st.session_state.model_image_bytes).decode("utf-8")
                    message_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    })
                message_content.append({"type": "text", "text": caption_prompt})

                response = claude_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": message_content}],
                )
                st.session_state.captions = response.content[0].text
                st.success("✅ 文案生成完成！（已參考實穿照場景）" if has_model_image else "✅ 文案生成完成！")

            except Exception as e:
                st.error(f"❌ 文案生成失敗：{e}")

if st.session_state.captions:
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
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("🧦 AI 電商素材生成器 · Powered by Google Gemini API · Built with Streamlit")
