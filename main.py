import subprocess
import sys
import os

# Zeabur 偵測到 Python 專案時會執行 python main.py
# 此檔案確保正確啟動 Streamlit
if __name__ == "__main__":
    port = os.environ.get("PORT", "8501")
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        check=True,
    )
