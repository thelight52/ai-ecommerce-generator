"""
行銷設計部 Agent API — FastAPI 入口
啟動方式: uvicorn api_server:app --host 0.0.0.0 --port 8080
驗證方式: X-Agent-Key header（對應環境變數 AGENT_API_KEY）
"""

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from services.copy_service import generate_ig_copy
from services.image_service import generate_product_image
from services.video_service import generate_product_video

app = FastAPI(
    title="行銷設計部 Agent API",
    description="AI 素材文案生成器的 Agent 端點，供 Orchestrator 呼叫",
    version="1.0.0",
)


# ── 驗證 ──────────────────────────────────────────────────────────────────────

def _verify_key(x_agent_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("AGENT_API_KEY", "")
    if expected and x_agent_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Agent-Key")


# ── Request / Response 模型 ────────────────────────────────────────────────────

class ProductInfo(BaseModel):
    name: str
    category: str
    features: list[str]
    style: Optional[str] = "韓系清新"
    tone: Optional[str] = "輕鬆活潑"


class CopyRequest(BaseModel):
    taskId: str
    product: ProductInfo


class ImageRequest(BaseModel):
    taskId: str
    productImageBase64: str
    style: Optional[str] = "電商主圖"


class VideoProductInfo(BaseModel):
    name: str
    description: str


class VideoRequest(BaseModel):
    taskId: str
    product: VideoProductInfo
    script: Optional[str] = None


# ── 端點 ──────────────────────────────────────────────────────────────────────

@app.get("/api/agent/health")
def health():
    return {
        "status": "ok",
        "agent": "行銷設計部",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capabilities": ["generate-copy", "generate-image", "generate-video"],
    }


@app.post("/api/agent/generate-copy")
def route_generate_copy(req: CopyRequest, _=Depends(_verify_key)):
    """根據商品資訊生成 IG 文案、商品描述與亮點。"""
    try:
        result = generate_ig_copy(
            product_name=req.product.name,
            category=req.product.category,
            features=req.product.features,
            style=req.product.style or "韓系清新",
            tone=req.product.tone or "輕鬆活潑",
        )
        return {"taskId": req.taskId, "status": "done", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/generate-image")
def route_generate_image(req: ImageRequest, _=Depends(_verify_key)):
    """從商品圖（base64）生成模特兒實穿照，回傳 base64 data URL。"""
    try:
        image_url = generate_product_image(
            product_image_base64=req.productImageBase64,
            style=req.style or "電商主圖",
        )
        return {"taskId": req.taskId, "status": "done", "result": {"generatedImageUrl": image_url}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/generate-video")
def route_generate_video(req: VideoRequest, _=Depends(_verify_key)):
    """啟動 Kling 影片生成，同步等待結果（最長 5 分鐘）。"""
    try:
        result = generate_product_video(
            product_name=req.product.name,
            product_description=req.product.description,
            script=req.script,
        )
        return {"taskId": req.taskId, "status": "done", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
