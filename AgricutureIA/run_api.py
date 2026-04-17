#!/usr/bin/env python
"""Script para iniciar a API PlantVision"""
import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    subprocess.run(cmd, cwd=r"c:\TCC - Code\IA - TCC\AgricutureIA\src")
