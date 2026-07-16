@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

echo ========================================
echo   증명사진 미백 · 누끼
echo ========================================
echo.

set "PY_CMD="

REM 64비트 Python 우선 탐색 (MediaPipe·OpenCV는 32비트 미지원)
for %%T in (-3.11-64 -3.12-64 -3.13-64 -3-64) do (
    py %%T -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py %%T"
        goto :py_found
    )
)

py -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto :py_found
)

python -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :py_found
)

echo [오류] 64비트 Python 3.10 이상이 필요합니다.
echo.
echo 현재 PC에는 32비트 Python만 설치되어 있거나 Python이 없습니다.
echo 이 앱은 MediaPipe를 사용하므로 64비트 Python만 지원합니다.
echo.
echo 1. https://www.python.org/downloads/ 에서 Python 3.11 (64-bit) 설치
echo 2. 설치 시 "Add Python to PATH" 체크
echo 3. run.bat 다시 실행
echo.
echo 이미 32비트 Python이 있다면 64비트 버전을 추가로 설치해도 됩니다.
pause
exit /b 1

:py_found
echo [+] Python: %PY_CMD% (64-bit)
echo.

if not exist "venv\Scripts\python.exe" (
    echo [1/3] 가상환경 생성 중...
    %PY_CMD% -m venv venv
    if not exist "venv\Scripts\python.exe" (
        echo.
        echo 가상환경 생성에 실패했습니다.
        pause
        exit /b 1
    )
) else (
    venv\Scripts\python.exe -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [1/3] 32비트 가상환경 감지 — 64비트로 재생성 중...
        rmdir /s /q venv
        %PY_CMD% -m venv venv
        if not exist "venv\Scripts\python.exe" (
            echo.
            echo 가상환경 생성에 실패했습니다.
            pause
            exit /b 1
        )
    ) else (
        echo [1/3] 가상환경 확인 완료
    )
)

if not exist "models\face_landmarker.task" (
    echo [+] 얼굴 인식 모델 다운로드 중...
    if not exist "models" mkdir models
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile 'models\face_landmarker.task'"
)

if not exist "models\selfie_segmenter.tflite" (
    echo [+] 누끼 모델 다운로드 중...
    if not exist "models" mkdir models
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite' -OutFile 'models\selfie_segmenter.tflite'"
)

echo [2/3] 패키지 설치 중...
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\python.exe -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)

echo [3/3] 서버 실행 중...
echo.
echo   http://localhost:5000
echo   API 문서: http://localhost:5000/docs
echo.
venv\Scripts\python.exe server.py

if errorlevel 1 (
    echo.
    echo 서버 실행 중 오류가 발생했습니다.
    pause
)
