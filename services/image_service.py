"""
圖片生成服務 — 呼叫 Gemini API 從商品圖生成模特兒實穿照
環境變數: GEMINI_API_KEY
"""

import base64
import io
import os

from google import genai
from google.genai import types
from PIL import Image

# 風格 → 英文描述映射
_STYLE_MAP = {
    "韓系甜美": "Korean sweet feminine style, soft pastel colors, gentle natural light, dreamy atmosphere",
    "日系清新": "Japanese fresh clean style, minimalist, soft diffused daylight, airy and calm",
    "都市街拍": "urban street fashion editorial, high contrast, city background, trendy and chic",
    "咖啡廳慵懶": "cozy cafe ambiance, warm bokeh background, relaxed lifestyle, golden hour tones",
    "電商主圖": "clean white e-commerce studio background, soft diffused lighting, professional product shot",
}


def generate_product_image(
    product_image_base64: str,
    style: str = "電商主圖",
) -> str:
    """
    使用 Gemini 圖生圖，根據商品照生成實穿展示圖。
    回傳: base64 data URL (data:image/png;base64,...)
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # 支援帶 data URL 前綴或純 base64
    raw = product_image_base64
    mime_type = "image/jpeg"
    if raw.startswith("data:"):
        header, raw = raw.split(",", 1)
        if "png" in header:
            mime_type = "image/png"
        elif "webp" in header:
            mime_type = "image/webp"

    image_bytes = base64.b64decode(raw)
    style_desc = _STYLE_MAP.get(style, _STYLE_MAP["電商主圖"])

    generation_prompt = (
        f"Generate a lifestyle product photo based on the provided product image. "
        f"Style: {style_desc}. "
        f"CRITICAL: Keep the EXACT SAME product design, color, and pattern as shown in the reference. "
        f"Professional e-commerce quality, clear product visibility, natural lighting."
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            generation_prompt,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(image_size="1K"),
        ),
    )

    image_data = None
    candidates = getattr(response, "candidates", None)
    if candidates and len(candidates) > 0:
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data
                break

    if not image_data:
        raise ValueError("Gemini 未回傳圖片，請稍後重試")

    # 統一縮放到 1024×1024
    img = Image.open(io.BytesIO(image_data))
    if img.width != 1024 or img.height != 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_data = buf.getvalue()

    b64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
