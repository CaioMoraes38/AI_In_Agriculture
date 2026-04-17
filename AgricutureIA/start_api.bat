@echo off
cd /d "c:\TCC - Code\IA - TCC\AgricutureIA\src"
"c:\TCC - Code\IA - TCC\AgricutureIA\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
