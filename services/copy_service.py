"""
文案生成服務 — 呼叫 Claude API 產出 IG 文案、商品描述、亮點
環境變數: ANTHROPIC_API_KEY
"""

import json
import os

import anthropic


def generate_ig_copy(
    product_name: str,
    category: str,
    features: list[str],
    style: str = "韓系清新",
    tone: str = "輕鬆活潑",
) -> dict:
    """
    根據商品資訊生成 IG 文案。
    回傳: { igCaption, productDescription, highlights }
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    features_text = "、".join(features) if features else "無特別說明"

    prompt = f"""你是一位專業的電商社群媒體文案師，擅長韓系時尚品牌的 Instagram 行銷。

【商品資訊】
- 商品名稱：{product_name}
- 商品類型：{category}
- 商品特色：{features_text}
- 風格定調：{style}
- 語氣調性：{tone}

請生成以下內容，輸出純 JSON（不要加 markdown 代碼塊）：
{{
  "igCaption": "完整的 IG 貼文文案（含標題、內文 4-6 行、Call to Action、20-25 個 Hashtag，繁體中文）",
  "productDescription": "商品描述（2-3 句，適合電商平台，繁體中文）",
  "highlights": ["亮點1", "亮點2", "亮點3"]
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)
