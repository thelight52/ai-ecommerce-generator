import subprocess
import sys
import os

# Zeabur 偵測到 Python 專案時會執行 python main.py
# APP_MODE=api  → 啟動 FastAPI Agent Server (uvicorn)
# APP_MODE 未設 → 啟動 Streamlit（預設）
if __name__ == "__main__":
    port = os.environ.get("PORT", "8080")
    mode = os.environ.get("APP_MODE", "streamlit")

    if mode == "api":
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn", "api_server:app",
                "--host", "0.0.0.0",
                f"--port={port}",
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", "app.py",
                f"--server.port={port}",
                "--server.address=0.0.0.0",
                "--server.headless=true",
            ],
            check=True,
        )
