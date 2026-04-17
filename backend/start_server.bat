@echo off
echo =======================================================
echo          Factory AI Face Recognition Backend 
echo        (NVIDIA GPU Hardware Acceleration Mode)
echo =======================================================
echo.
echo Activating Conda Python 3.10 Environment...
call conda activate factory_ai
echo Environment activated. Launching API Server...
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
