@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ========================================
echo   실행 파일 빌드 (PyInstaller)
echo ========================================
echo.

set "PY=py -3.11"

%PY% -c "import struct; raise SystemExit(0 if struct.calcsize('P')==8 else 1)" >nul 2>&1
if errorlevel 1 (
    echo [오류] 64비트 Python 3.11이 필요합니다.
    echo   py install 3.11
    pause
    exit /b 1
)

if not exist "build_venv\Scripts\python.exe" (
    echo [+] 빌드용 가상환경 생성...
    %PY% -m venv build_venv
)

call build_venv\Scripts\activate.bat

echo [+] 패키지 설치...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

echo.
echo [+] 데스크톱 앱 빌드 중...
pyinstaller --noconfirm --clean build_exe.spec
if errorlevel 1 goto :fail

echo.
echo [+] FastAPI 서버 빌드 중...
pyinstaller --noconfirm --clean build_server.spec
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
