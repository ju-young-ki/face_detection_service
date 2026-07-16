# 증명사진 미백 · 잡티제거

틱톡 카메라 뷰티 필터 스타일로 증명사진의 피부를 미백하고 잡티를 제거하는 Windows 데스크톱 앱 + FastAPI 로컬 서버입니다.

## 기능

- **미백** — 피부 영역만 자연스럽게 밝게 보정
- **스무딩** — 모공·잔주름을 부드럽게 (틱톡 필터 느낌)
- **원본/보정 비교** — 좌우 미리보기

## 추가된 기능

- **미백** — 여러명도 인식 하여 처리함
- **배경제거** — 여러명도 인식 하여 배경 제거함
- **선명도** — 스무딩 후 떨어진 선명도 조정
- **컬러조정** — 감마, 콘트라스트, R, G, B, 색온도, 색조, 채도 조절
- **FastAPI 서버** — 규격 크롭·얼굴 감지 REST API

## 요구 사항

- Windows 10 이상
- [Python 3.10+ **64-bit**](https://www.python.org/downloads/) (설치 시 "Add Python to PATH" 체크)
- **32비트 Python은 지원하지 않습니다** (MediaPipe·OpenCV 제약)

## 실행 방법 (데스크톱 앱)

1. **`run.bat` 더블클릭** (가장 쉬운 방법)
2. 첫 실행 시 Python 가상환경, 패키지, 얼굴 인식 모델이 자동 설치됩니다
3. **이미지 열기**로 증명사진 선택
4. 슬라이더로 효과 조절 후 **저장하기**

### 수동 실행

```powershell
cd d:\FlutterWork\face_detection_service
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## FastAPI 서버

규격 크롭·얼굴 감지 미리보기를 HTTP API로 제공합니다.

### 서버 실행

```powershell
cd d:\FlutterWork\face_detection_service
.\venv\Scripts\activate
python server.py
```

또는:

```powershell
uvicorn server:app --host 0.0.0.0 --port 5000
```

| 항목 | 주소 |
|------|------|
| API | http://127.0.0.1:5000 |
| Swagger UI | http://127.0.0.1:5000/docs |
| 헬스체크 | http://127.0.0.1:5000/health |

이미지 업로드는 모두 **`multipart/form-data`** 입니다.

파라미터를 **생략**하면 `defaults.json`에 저장된 기본값이 적용됩니다.  
현재 기본값은 `GET /api/defaults` 또는 `defaults.json` 파일에서 확인·수정할 수 있습니다.

### API 목록

| 메서드 | 경로 | 설명 | 응답 |
|--------|------|------|------|
| GET | `/` | 서비스 안내 | JSON |
| GET | `/health` | 헬스체크 | JSON |
| GET | `/api/defaults` | 기본 설정값 (`defaults.json`) | JSON |
| POST | `/crop` | `crop_category` 규격 얼굴 크롭 | JPEG |
| POST | `/api/face-detect/preview` | 얼굴 박스 미리보기 | JPEG |

### 기본 설정 (`defaults.json`)

요청에서 파라미터를 보내지 않으면 아래 파일의 값이 사용됩니다.

```json
{
  "whitening": 0.80,
  "smooth": 0.80,
  "sharpness": 0.5,
  "gamma": 1.0,
  "contrast": 0.0,
  "red": 0.0,
  "green": 0.0,
  "blue": 0.0,
  "temperature": 0.0,
  "hue": 0.0,
  "saturation": 0.0,
  "cutout": true,
  "crop_category": "420",
  "face_size": 10,
  "top_margin_mm": 1.0
}
```

파일을 수정한 뒤 서버를 재시작하면 새 기본값이 적용됩니다.

### `/crop` 파라미터

| 필드 | 범위 | 기본값 출처 | 의미 |
|------|------|-------------|------|
| `image` | 이미지 | (필수) | 업로드 파일 |
| `crop_category` | 360/420/540/600/720 | `defaults.json` | 크롭 규격 |
| `face_size` | 1~30 | `defaults.json` | 얼굴 크기(기준 10) |
| `top_margin_mm` | 0~15 | `defaults.json` | 머리 위 여백(mm) |
| `cutout` | true/false | `defaults.json` | 배경 제거(누끼) |
| `whitening` 등 | — | `defaults.json` | 미백·스무딩·색보정 |

처리 순서: **미백·누끼 → 얼굴 기준 크롭 → JPEG**.  
`cutout=true`(기본)이면 배경을 제거하고 흰색으로 합성합니다.

### curl 예제

아래 `photo.jpg`를 실제 파일 경로로 바꾸세요.  
CMD에서는 `^`, PowerShell에서는 `` ` `` 로 줄을 이어 쓸 수 있습니다.

#### 헬스체크 · 기본값 조회

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/api/defaults
```

#### 규격 크롭 (`/crop`)

| Category | Dimensions | Ratio |
|----------|------------|-------|
| `420` | 420×540 | 3.5×4.5 |
| `360` | 360×480 | 3×4 |
| `600` | 600×600 | 5×5 |
| `720` | 720×840 | 6×7 |
| `540` | 540×660 | 4.5×5.5 |

`face_size` (정수, 기본 10, 범위 1~30):
- `10` — 기준 얼굴 크기
- `11` 이상 — 얼굴이 커짐 (클로즈업)
- `9` 이하 — 얼굴이 작아짐 (줌아웃)

```bash
curl -X POST http://127.0.0.1:5000/crop ^
  -F "image=@photo.jpg" ^
  -F "crop_category=360" ^
  -F "face_size=12" ^
  -o crop_360.jpg
```

기본값(`crop_category=420`, `face_size=10`)만 사용할 때:

```bash
curl -X POST http://127.0.0.1:5000/crop ^
  -F "image=@photo.jpg" ^
  -o crop.jpg
```

#### 얼굴 박스 미리보기

```bash
curl -X POST http://127.0.0.1:5000/api/face-detect/preview ^
  -F "file=@photo.jpg" ^
  -o faces_preview.jpg
```

### 서버 사용 팁

- 브라우저에서 http://127.0.0.1:5000/docs 로 파일을 올려 바로 테스트할 수 있습니다.
- 기동 직후에는 MediaPipe 초기화가 끝날 수 있으니 `/health`로 확인하세요.
- Windows에서 경로에 공백이 있으면 `"image=@C:\path\with space\photo.jpg"`처럼 따옴표로 감싸세요.

## 사용 팁

| 프리셋 | 용도 |
|--------|------|
| **자연스럽게** | 과한 보정 없이 살짝만 |
| **틱톡 스타일** | 밝고 매끈한 뷰티 필터 |
| **증명사진** | 공식 제출용에 적합한 수준 |

- 증명사진 제출 전에는 **미백·스무딩을 너무 높이지 마세요** — 기관마다 보정 허용 범위가 다릅니다.
- 정면 얼굴이 잘 보이는 사진에서 가장 좋은 결과가 나옵니다.

## 기술

- Python, OpenCV, MediaPipe (얼굴·피부 영역 감지)
- CustomTkinter (Windows GUI)
- FastAPI, Uvicorn (로컬 REST API)

## 머리끝 찾는 방식

1. 세그멘테이션으로 배경 제거 후 흰색 배경 이미지 생성 (_make_white_cutout)
2. 얼굴 좌우 범위(얼굴 윤곽 랜드마크) 안에서
3. 이미지 맨 위부터 아래로 스캔해 흰색이 아닌 첫 행을 머리끝으로 사용
4. 턱끝·좌우는 기존처럼 랜드마크 사용
