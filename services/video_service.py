"""
影片生成服務 — 呼叫 Kling AI API 從商品圖生成短影音
環境變數: KLING_ACCESS_KEY, KLING_SECRET_KEY, ANTHROPIC_API_KEY
"""

import os
import time

import jwt as pyjwt
import requests
import anthropic

_KLING_ENDPOINTS = [
    "https://api-global.klingai.com",
    "https://api-singapore.klingai.com",
    "https://api-beijing.klingai.com",
]


def _kling_jwt(ak: str, sk: str) -> str:
    """生成 Kling JWT Token（有效 30 分鐘）"""
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return pyjwt.encode(payload, sk, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"})


def _generate_script_with_claude(product_name: str, product_description: str) -> str:
    """使用 Claude 生成影片拍攝腳本（英文，50 字內）"""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = (
        f"Write a short English video script (under 50 words) for a social media product video.\n"
        f"Product: {product_name}\n"
        f"Description: {product_description}\n\n"
        f"Describe model actions showcasing the product naturally. "
        f"End with a visually appealing pose. Output only the script, no extra text."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_product_video(
    product_name: str,
    product_description: str,
    image_base64: str | None = None,
    script: str | None = None,
    duration: int = 5,
    mode: str = "std",
    poll_max: int = 30,
) -> dict:
    """
    啟動 Kling 影片生成任務並輪詢等待結果。
    - poll_max: 最多輪詢次數（每次 10 秒），預設 30 次（5 分鐘）
    回傳: { videoUrl: str | None, script: str }
    """
    ak = os.environ.get("KLING_ACCESS_KEY", "")
    sk = os.environ.get("KLING_SECRET_KEY", "")

    # 無 key 時只回傳腳本
    if not ak or not sk:
        generated_script = script or _generate_script_with_claude(product_name, product_description)
        return {"videoUrl": None, "script": generated_script}

    # 生成腳本（若未提供）
    final_script = script or _generate_script_with_claude(product_name, product_description)

    token = _kling_jwt(ak, sk)
    payload: dict = {
        "model_name": "kling-v3",
        "mode": mode,
        "duration": str(duration),
        "aspect_ratio": "9:16",
        "prompt": final_script,
        "sound": "on",
    }
    if image_base64:
        # 去除 data URL 前綴
        raw = image_base64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        payload["image"] = raw

    # 嘗試各區域端點建立任務
    create_data = None
    kling_base = None
    for ep in _KLING_ENDPOINTS:
        try:
            resp = requests.post(
                f"{ep}/v1/videos/image2video",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200 and resp.text.strip():
                create_data = resp.json()
                if create_data.get("code") == 0:
                    kling_base = ep
                    break
        except Exception:
            continue

    if not create_data or create_data.get("code") != 0:
        return {"videoUrl": None, "script": final_script}

    task_id = create_data["data"]["task_id"]

    # 輪詢等待完成
    for _ in range(poll_max):
        time.sleep(10)
        token = _kling_jwt(ak, sk)
        try:
            q = requests.get(
                f"{kling_base}/v1/videos/image2video/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if not q.text.strip():
                continue
            qd = q.json()
            if qd.get("code") != 0:
                break
            status = qd["data"]["task_status"]
            if status == "succeed":
                videos = qd["data"].get("task_result", {}).get("videos", [])
                if videos:
                    return {"videoUrl": videos[0].get("url"), "script": final_script}
                break
            elif status == "failed":
                break
        except Exception:
            continue

    return {"videoUrl": None, "script": final_script}
