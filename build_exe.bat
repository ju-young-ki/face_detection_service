@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ========================================
echo   실행 파일 빌드 (PyInstaller)
echo ========================================
echo.

set "PY="

REM 64비트 Python 우선 탐색 (MediaPipe·OpenCV는 32비트 미지원)
for %%T in (-3.11-64 -3.12-64 -3.13-64 -3-64) do (
    py %%T -c "import struct; raise SystemExit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=py %%T"
        goto :py_found
    )
)

py -c "import struct; raise SystemExit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY=py"
    goto :py_found
)

python -c "import struct; raise SystemExit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :py_found
)

echo [오류] 64비트 Python 3.10 이상이 필요합니다.
echo   설치 경로 예: E:\Python
echo   설치 시 "Add Python to PATH" 체크 후 터미널을 다시 열어주세요.
pause
exit /b 1

:py_found
echo [+] Python: %PY% (64-bit)
echo.

REM 기존 venv가 깨졌거나(기반 Python 삭제 등) 없으면 재생성
set "NEED_VENV=0"
if not exist "build_venv\Scripts\python.exe" set "NEED_VENV=1"
if exist "build_venv\Scripts\python.exe" (
    build_venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if errorlevel 1 set "NEED_VENV=1"
)

if "%NEED_VENV%"=="1" (
    echo [+] 빌드용 가상환경 생성/재생성...
    if exist "build_venv" rmdir /s /q "build_venv"
    %PY% -m venv build_venv
    if errorlevel 1 (
        echo [오류] 가상환경 생성에 실패했습니다.
        pause
        exit /b 1
    )
)

call build_venv\Scripts\activate.bat
build_venv\Scripts\python.exe -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [오류] 가상환경 Python이 동작하지 않습니다.
    echo   build_venv 폴더를 삭제한 뒤 다시 실행해주세요.
    pause
    exit /b 1
)

echo [+] 패키지 설치...
build_venv\Scripts\python.exe -m pip install --upgrade pip
build_venv\Scripts\python.exe -m pip install -r requirements.txt
build_venv\Scripts\python.exe -m pip install pyinstaller

REM 실행 중이면 dist 폴더 삭제가 막히므로 먼저 종료
taskkill /F /IM "증명사진미백.exe" >nul 2>&1
taskkill /F /IM "FaceDetectionServer.exe" >nul 2>&1

echo.
echo [+] 데스크톱 앱 빌드 중...
build_venv\Scripts\python.exe -m PyInstaller --noconfirm --clean build_exe.spec
if errorlevel 1 goto :fail

echo.
echo [+] FastAPI 서버 빌드 중...
build_venv\Scripts\python.exe -m PyInstaller --noconfirm --clean build_server.spec
if errorlevel 1 goto :fail

REM 실행 파일 옆에서 defaults.json 수정 가능하도록 복사
copy /Y defaults.json "dist\증명사진미백\defaults.json" >nul
copy /Y defaults.json "dist\FaceDetectionServer\defaults.json" >nul

echo.
echo ========================================
echo   빌드 완료
echo ========================================
echo.
echo   데스크톱 앱:
echo     dist\증명사진미백\증명사진미백.exe
echo.
echo   FastAPI 서버:
echo     dist\FaceDetectionServer\FaceDetectionServer.exe
echo     → http://127.0.0.1:5000
echo.
pause
exit /b 0

:fail
echo.
echo [오류] 빌드에 실패했습니다.
pause
exit /b 1
