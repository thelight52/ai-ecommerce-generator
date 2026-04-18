"""
情境照生成服務 — 用 Gemini 將商品合成到指定場景中
環境變數: GEMINI_API_KEY
"""

import base64
import io
import os

from google import genai
from google.genai import types
from PIL import Image

# 場景 → 英文描述
_SCENE_MAP = {
    "咖啡廳": "cozy cafe table top, warm wooden surface, coffee cup beside, soft bokeh background, golden hour natural light",
    "臥室": "cozy bedroom setting, soft white bedsheets, morning sunlight filtering through sheer curtains, pillow beside product",
    "戶外草地": "lush green outdoor grass field, bright natural daylight, soft bokeh of trees and sky in background",
    "書桌": "clean minimalist desk setup, soft natural side light, notebook and stationery as props, neutral tones",
    "浴室": "spa-style bathroom counter, white marble surface, small plants and candles as accents, soft warm light",
    "廚房": "bright modern kitchen counter, white and wood tones, fresh ingredients as props, natural window light",
    "客廳": "stylish living room, light sofa and throw blanket, indoor plants, warm ambient light",
    "木質桌面": "clean natural wood table surface, minimalist styling, soft diffused overhead light",
    "大理石桌面": "luxurious white marble surface, clean and elegant, soft studio lighting from above",
    "戶外街道": "urban street sidewalk, blurred city background, natural outdoor light, editorial fashion vibe",
}

# 風格色調 → 英文修飾
_STYLE_MAP = {
    "溫暖色調": "warm golden tones, cozy and inviting atmosphere, slight warm filter",
    "清新冷調": "cool crisp tones, fresh and airy, slight cool-toned filter",
    "高對比": "high contrast, bold shadows and highlights, dramatic editorial look",
    "柔和粉嫩": "soft pastel palette, dreamy and gentle, light and airy aesthetic",
    "自然寫實": "natural realistic lighting, true-to-life colors, no filter effect",
}


def generate_scene_image(
    product_image_base64: str,
    scene: str,
    style: str | None = None,
) -> str:
    """
    將商品平拍照合成到指定場景中。
    - scene: 場景名稱（中文），對應 _SCENE_MAP；未知場景直接當英文描述使用
    - style: 色調風格（選填）
    回傳: base64 data URL (data:image/png;base64,...)
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    raw = product_image_base64
    mime_type = "image/jpeg"
    if raw.startswith("data:"):
        header, raw = raw.split(",", 1)
        if "png" in header:
            mime_type = "image/png"
        elif "webp" in header:
            mime_type = "image/webp"

    image_bytes = base64.b64decode(raw)

    scene_desc = _SCENE_MAP.get(scene, scene)  # 找不到映射則直接用傳入字串
    style_desc = _STYLE_MAP.get(style or "", "")
    style_suffix = f" Color style: {style_desc}." if style_desc else ""

    prompt = (
        f"Place this product naturally into the following scene: {scene_desc}.{style_suffix} "
        f"The product should look like it belongs in the scene — realistic placement, "
        f"correct perspective, natural shadows and lighting that match the environment. "
        f"CRITICAL: Keep the EXACT SAME product design, color, pattern, and details as shown in the reference image. "
        f"Do NOT alter or simplify the product. "
        f"Professional e-commerce lifestyle photography quality."
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
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

    img = Image.open(io.BytesIO(image_data))
    if img.width != 1024 or img.height != 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_data = buf.getvalue()

    b64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
