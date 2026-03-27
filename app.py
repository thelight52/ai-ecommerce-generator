"""
AI 電商素材生成器
基於 Google AI Studio (Gemini) 的全自動電商素材生成流程

流程：上傳襪子平拍照 → 自動產出提示詞 → 生成模特兒實穿照 → 生成社群文案
"""

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
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

    # 優先順序：環境變數 > session_state 快取 > 空白
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if "api_key_cache" not in st.session_state:
        st.session_state.api_key_cache = env_key

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.api_key_cache,
        placeholder="AIzaSy...",
        help="前往 https://aistudio.google.com/apikey 取得免費 API Key"
    )

    if api_key:
        st.session_state.api_key_cache = api_key
        st.markdown('<div class="success-box">✅ API Key 已設定</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box">👉 請前往 <a href="https://aistudio.google.com/apikey" target="_blank">Google AI Studio</a> 取得 API Key</div>',
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
- **分析 & 文案**：`gemini-1.5-flash`
- **圖片生成**：`gemini-2.0-flash-preview-image-generation`
""")
    st.caption("Powered by Google AI Studio")

# ─────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────
for key in ["prompts", "model_image_bytes", "captions", "upload_mime"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
elif not api_key:
    st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
else:
    if st.button("🔍 分析圖片並自動產出提示詞", type="primary", use_container_width=False):
        with st.spinner("Gemini 正在分析商品圖片，自動生成提示詞…"):
            try:
                client = genai.Client(api_key=api_key)
                img_bytes = uploaded_file.getvalue()

                analysis_prompt = """You are a professional e-commerce fashion photographer and AI image prompt engineer specializing in Korean style.

Analyze this product flat lay image carefully and generate AI image generation prompts for a Korean female model wearing this product.

IMPORTANT RULES:
- Do NOT describe the specific pattern, color, or design details of the product itself
- Focus on the MODEL SCENE: pose, angle, background, lighting, styling
- Shot must be LOWER BODY only (waist down, include waist)
- Korean female model aesthetic, slim legs
- E-commerce commercial quality
- Slight side angle to showcase the product
- The negative prompt should prevent common AI image generation errors

Return ONLY a valid JSON object (no markdown, no extra text) with this exact structure:
{
  "positive_en": "Korean female model, slim legs, [pose details], lower body shot from waist down, [outfit pairing], [background], [lighting], [photography quality]",
  "positive_zh": "韓系女性模特兒，[姿勢細節]，腰部以下畫面，[服裝搭配]，[背景]，[光線]，[攝影質感]",
  "negative_en": "full body, face visible, upper body dominant, extra limbs, distorted feet, deformed toes, blurry, low quality, pixelated, watermark, text overlay, logo, jpeg artifacts, overexposed, dark shadows, plastic skin, unrealistic proportions, missing product, duplicate body parts, bad anatomy, extra fingers, nsfw"
}"""

                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type=st.session_state.upload_mime,
                        ),
                        analysis_prompt,
                    ],
                )

                text = response.text.strip()
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
# STEP 3：生成模特兒實穿照
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 3 · 🎨 生成模特兒實穿照</div>', unsafe_allow_html=True)

if not st.session_state.prompts:
    st.info("請先完成 Step 2 產出提示詞")
elif not api_key:
    st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
else:
    scene_options = {
        "簡約室內（白色大理石地板）": "sitting on white marble floor, clean minimal indoor background, soft natural window light, warm white tones",
        "咖啡廳外拍（暖陽散景）": "outdoor cafe setting, warm golden afternoon sunlight, blurred bokeh background, film photography aesthetic",
        "清爽白背景（電商主圖）": "pure white studio background, soft diffused studio lighting, bright and airy atmosphere, e-commerce hero shot",
    }
    selected_scene = st.selectbox("🏠 選擇場景", list(scene_options.keys()))
    scene_desc = scene_options[selected_scene]

    if st.button("🎨 生成模特兒實穿照", type="primary", use_container_width=False):
        with st.spinner("Gemini 正在生成圖片，約需 30～60 秒…"):
            try:
                client = genai.Client(api_key=api_key)

                base_prompt = st.session_state.prompts["positive_en"]
                generation_prompt = (
                    f"{base_prompt}, {scene_desc}, "
                    f"photorealistic, commercial e-commerce photography, 8K resolution, "
                    f"sharp fabric texture, feminine and elegant, editorial fashion quality. "
                    f"Avoid: {st.session_state.prompts['negative_en']}"
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=generation_prompt,
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
                    st.session_state.model_image_bytes = image_bytes
                    st.success("✅ 模特兒實穿照生成成功！")
                else:
                    st.error("❌ 未收到圖片，請確認 API Key 有圖片生成配額，或稍後再試。")

            except Exception as e:
                st.error(f"❌ 生成失敗：{e}")
                st.markdown(
                    '<div class="info-box">💡 提示：圖片生成需要付費 API Key（Gemini API 免費層可能不支援 image generation）。'
                    '請至 <a href="https://aistudio.google.com/apikey" target="_blank">AI Studio</a> 確認您的配額。</div>',
                    unsafe_allow_html=True,
                )

if st.session_state.model_image_bytes:
    col_gen, col_dl = st.columns([2, 1])
    with col_gen:
        gen_img = Image.open(io.BytesIO(st.session_state.model_image_bytes))
        st.image(gen_img, caption="✨ AI 生成的模特兒實穿照", use_container_width=True)
    with col_dl:
        st.markdown("#### ⬇️ 下載圖片")
        st.download_button(
            label="💾 下載 PNG",
            data=st.session_state.model_image_bytes,
            file_name="model_wearing_photo.png",
            mime="image/png",
            use_container_width=True,
        )
        st.markdown("---")
        st.markdown("**圖片資訊**")
        gen_img_info = Image.open(io.BytesIO(st.session_state.model_image_bytes))
        st.markdown(f"- 尺寸：`{gen_img_info.width} × {gen_img_info.height}`")
        st.markdown(f"- 大小：`{len(st.session_state.model_image_bytes) / 1024:.0f} KB`")

st.divider()

# ─────────────────────────────────────────
# STEP 4：生成社群貼文文案
# ─────────────────────────────────────────
st.markdown('<div class="step-header">Step 4 · ✍️ 生成 Instagram 社群貼文文案</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("請先在左側 Sidebar 輸入 Gemini API Key")
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

    if st.button("✍️ 生成社群貼文文案", type="primary", use_container_width=False):
        with st.spinner("Gemini 正在撰寫文案…"):
            try:
                client = genai.Client(api_key=api_key)

                lang_instruction = {
                    "繁體中文": "請全程使用繁體中文（台灣用語）撰寫，包含標題、內文與 hashtag。",
                    "繁體中文 + English": "標題與內文使用繁體中文，hashtag 中英混合。",
                    "English only": "Please write entirely in English including all hashtags.",
                }[caption_lang]

                extra = f"\n商品特色補充：{product_desc}" if product_desc else ""

                caption_prompt = f"""你是一位專業的電商社群媒體文案師，擅長韓系時尚品牌的 Instagram 行銷。

請為以下商品生成一篇高互動率的 Instagram 貼文文案：
- 商品類型：韓系可愛中筒襪（電商商品）
- 風格定位：{caption_style}{extra}
- 語言規範：{lang_instruction}

請依照以下格式輸出，不要加其他說明：

【標題】
（1行，吸睛有力，可含表情符號）

【貼文內容】
（4～6行，自然口語化，帶出商品亮點，含適量表情符號）

【Call to Action】
（1行，引導互動或購買）

【Hashtags】
（20～25個，分行整理，涵蓋：商品、穿搭、韓系、季節、品味生活 等主題）
"""
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=caption_prompt,
                )
                st.session_state.captions = response.text
                st.success("✅ 文案生成完成！")

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
