"""증명사진 미백·누끼 이미지 처리 엔진."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from paths import resource_dir

_MODEL_PATH = resource_dir() / "models" / "face_landmarker.task"
_SEGMENTER_MODEL_PATH = resource_dir() / "models" / "selfie_segmenter.tflite"


def _indices_from_connections(connections) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for connection in connections:
        for idx in (connection.start, connection.end):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


class PhotoProcessor:
    """틱톡 카메라 스타일의 피부 미백·누끼 처리기."""

    _LANDMARK_CHIN = 152
    _WHITE_THRESHOLD = 250
    _FACE_X_MARGIN_RATIO = 0.05

    # 여권 사진 규격: 35×45mm, 300dpi (category 420 과 동일 비율)
    _PASSPORT_ASPECT = 35 / 45
    _PASSPORT_WIDTH_PX = round(35 / 25.4 * 300)
    _PASSPORT_HEIGHT_PX = round(45 / 25.4 * 300)
    _PASSPORT_HEIGHT_MM = 45
    _HEAD_HEIGHT_RATIO = 34 / 45
    _DEFAULT_TOP_MARGIN_MM = 2.0
    _DEFAULT_FACE_SIZE = 10
    _MIN_FACE_SIZE = 1
    _MAX_FACE_SIZE = 30

    # crop_category → 출력 픽셀 / 비율 / 사진 세로 길이(mm, top_margin 계산용)
    # ※ height_mm 은 실제 mm (예: 3.5×4.5 → 45). 비율 숫자(4.5)를 쓰면 여백이 ~10배 과다해짐.
    CROP_CATEGORIES: dict[str, dict[str, float | int | str]] = {
        "420": {"width": 420, "height": 540, "ratio": "3.5x4.5", "height_mm": 45.0},
        "360": {"width": 360, "height": 480, "ratio": "3x4", "height_mm": 40.0},
        "600": {"width": 600, "height": 600, "ratio": "5x5", "height_mm": 50.0},
        "720": {"width": 720, "height": 840, "ratio": "6x7", "height_mm": 70.0},
        "540": {"width": 540, "height": 660, "ratio": "4.5x5.5", "height_mm": 55.0},
    }

    _FACE_OVAL = _indices_from_connections(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
    _LEFT_EYE = _indices_from_connections(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    _RIGHT_EYE = _indices_from_connections(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    _LIPS = _indices_from_connections(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)

    def __init__(self) -> None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"얼굴 인식 모델이 없습니다: {_MODEL_PATH}\n"
                "README의 모델 다운로드 안내를 확인해주세요."
            )
        if not _SEGMENTER_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"누끼 모델이 없습니다: {_SEGMENTER_MODEL_PATH}\n"
                "README의 모델 다운로드 안내를 확인해주세요."
            )

        # model_asset_path는 Windows에서 비ASCII 경로(한글 폴더명 등)를
        # 열지 못하는 경우가 있어, Python으로 읽은 바이트를 넘긴다.
        face_model = _MODEL_PATH.read_bytes()
        segmenter_model = _SEGMENTER_MODEL_PATH.read_bytes()

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=face_model),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        segmenter_options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_buffer=segmenter_model),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)

    def close(self) -> None:
        self._landmarker.close()
        self._segmenter.close()

    def draw_face_boxes(
        self,
        image_bgr: np.ndarray,
        *,
        detect_from: np.ndarray | None = None,
    ) -> np.ndarray:
        """얼굴을 감지해 머리끝~턱끝 범위에 적색 사각형을 그린다."""
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        annotated = image_bgr.copy()
        if annotated.ndim == 3 and annotated.shape[2] == 4:
            annotated = annotated[:, :, :3].copy()

        detect_image = detect_from if detect_from is not None else image_bgr
        boxes = self.get_face_boxes(detect_image)

        h, w = annotated.shape[:2]
        detect_h, detect_w = detect_image.shape[:2]
        scale_x = w / detect_w
        scale_y = h / detect_h
        for x1, y1, x2, y2 in boxes:
            if scale_x != 1.0 or scale_y != 1.0:
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return annotated

    def get_face_boxes(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """감지된 얼굴의 머리끝~턱끝 사각형 목록을 반환한다."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = image_bgr[:, :, :3]

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        white_cutout = self._make_white_cutout(image_bgr)

        boxes: list[tuple[int, int, int, int]] = []
        for landmarks in result.face_landmarks:
            boxes.append(self._face_bbox_from_landmarks(landmarks, w, h, white_cutout))
        return boxes

    def crop_passport_photo(
        self,
        image_bgr: np.ndarray,
        *,
        top_margin_mm: float | None = None,
        face_size: int | None = None,
    ) -> np.ndarray:
        """여권 사진 규격(35×45mm ≈ category 420)으로 얼굴을 기준 크롭한다."""
        return self.crop_by_category(
            image_bgr,
            "420",
            top_margin_mm=top_margin_mm,
            face_size=face_size,
            output_size=(self._PASSPORT_WIDTH_PX, self._PASSPORT_HEIGHT_PX),
        )

    def crop_by_category(
        self,
        image_bgr: np.ndarray,
        crop_category: str | int,
        *,
        top_margin_mm: float | None = None,
        face_size: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """crop_category 규격으로 얼굴을 기준 크롭한 뒤 지정 해상도로 리사이즈한다.

        Category / Dimensions / Ratio
          420 / 420×540 / 3.5×4.5
          360 / 360×480 / 3×4
          600 / 600×600 / 5×5
          720 / 720×840 / 6×7
          540 / 540×660 / 4.5×5.5

        face_size:
          기준 10. 크면 얼굴이 커지고(클로즈업), 작으면 얼굴이 작아진다(줌아웃).
        """
        key = str(crop_category).strip()
        spec = self.CROP_CATEGORIES.get(key)
        if spec is None:
            supported = ", ".join(sorted(self.CROP_CATEGORIES))
            raise ValueError(
                f"지원하지 않는 crop_category: {crop_category}. "
                f"사용 가능: {supported}"
            )

        out_w = int(spec["width"])
        out_h = int(spec["height"])
        aspect = out_w / out_h
        height_mm = float(spec["height_mm"])

        if top_margin_mm is None:
            top_margin_mm = self._DEFAULT_TOP_MARGIN_MM
        top_margin_mm = float(np.clip(top_margin_mm, 0.0, 15.0))
        top_margin_ratio = top_margin_mm / height_mm

        if face_size is None:
            face_size = self._DEFAULT_FACE_SIZE
        face_size = int(
            np.clip(int(face_size), self._MIN_FACE_SIZE, self._MAX_FACE_SIZE)
        )
        # face_size=10 → 기준 비율. 값이 클수록 머리 점유 비율↑ → 얼굴이 커 보임
        head_ratio = self._HEAD_HEIGHT_RATIO * (face_size / self._DEFAULT_FACE_SIZE)
        head_ratio = float(np.clip(head_ratio, 0.35, 0.95))

        image = self.composite_on_white(image_bgr)
        boxes = self.get_face_boxes(image)
        if not boxes:
            raise ValueError("얼굴을 찾을 수 없습니다. 다른 사진을 사용해주세요.")

        x1, y1, x2, y2 = max(
            boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
        )
        face_cx = (x1 + x2) / 2
        face_h = max(y2 - y1, 1)

        crop_h = face_h / head_ratio
        crop_w = crop_h * aspect
        crop_x1 = int(face_cx - crop_w / 2)
        crop_y1 = int(y1 - top_margin_ratio * crop_h)
        crop_x2 = int(crop_x1 + crop_w)
        crop_y2 = int(crop_y1 + crop_h)

        cropped = self._crop_with_white_pad(image, crop_x1, crop_y1, crop_x2, crop_y2)
        if output_size is None:
            output_size = (out_w, out_h)
        return cv2.resize(
            cropped,
            output_size,
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def get_crop_category_info(crop_category: str | int) -> dict[str, float | int | str]:
        """crop_category 메타데이터를 반환한다. 없으면 ValueError."""
        key = str(crop_category).strip()
        spec = PhotoProcessor.CROP_CATEGORIES.get(key)
        if spec is None:
            supported = ", ".join(sorted(PhotoProcessor.CROP_CATEGORIES))
            raise ValueError(
                f"지원하지 않는 crop_category: {crop_category}. "
                f"사용 가능: {supported}"
            )
        return dict(spec)

    @staticmethod
    def composite_on_white(image: np.ndarray) -> np.ndarray:
        """투명 배경 이미지를 흰색 배경 위에 합성한다."""
        if image.ndim != 3 or image.shape[2] != 4:
            return image

        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)

    @staticmethod
    def _crop_with_white_pad(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """이미지 범위를 벗어나는 영역은 흰색으로 채워 크롭한다."""
        crop_w = max(1, int(x2 - x1))
        crop_h = max(1, int(y2 - y1))
        canvas = np.full((crop_h, crop_w, 3), 255, dtype=np.uint8)

        h, w = image.shape[:2]
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return canvas

        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        canvas[dst_y1 : dst_y1 + (src_y2 - src_y1), dst_x1 : dst_x1 + (src_x2 - src_x1)] = image[
            src_y1:src_y2, src_x1:src_x2
        ]
        return canvas

    def _face_bbox_from_landmarks(
        self,
        landmarks,
        width: int,
        height: int,
        white_cutout: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """배경 제거 후 상단 스캔으로 머리끝, 랜드마크로 턱끝·좌우를 계산한다."""
        xs = [int(landmarks[i].x * width) for i in self._FACE_OVAL]
        left_x = min(xs)
        right_x = max(xs)
        margin_x = int((right_x - left_x) * self._FACE_X_MARGIN_RATIO)

        x1 = max(0, left_x - margin_x)
        x2 = min(width - 1, right_x + margin_x)
        y1 = self._find_head_top_y(white_cutout, x1, x2)
        if y1 is None:
            ys = [int(landmarks[i].y * height) for i in self._FACE_OVAL]
            y1 = max(0, min(ys))
        y2 = min(height - 1, int(landmarks[self._LANDMARK_CHIN].y * height))
        return x1, y1, x2, y2

    def _find_head_top_y(
        self,
        white_cutout: np.ndarray,
        left_x: int,
        right_x: int,
    ) -> int | None:
        """흰 배경 이미지에서 위→아래로 스캔해 첫 비(非)백색 행을 머리끝으로 본다."""
        if left_x >= right_x:
            return 0

        column = white_cutout[:, left_x : right_x + 1]
        non_white = np.any(column < self._WHITE_THRESHOLD, axis=2)
        rows_with_person = np.any(non_white, axis=1)
        hit = np.flatnonzero(rows_with_person)
        if hit.size == 0:
            return None
        return int(hit[0])

    def _make_white_cutout(self, image_bgr: np.ndarray) -> np.ndarray:
        """인물만 남기고 배경을 흰색으로 채운 이미지를 만든다."""
        h, w = image_bgr.shape[:2]
        mask = self._get_person_mask(image_bgr)
        if mask is None:
            return np.full((h, w, 3), 255, dtype=np.uint8)

        alpha = mask[..., None]
        bgr = image_bgr.astype(np.float32)
        white = np.full_like(bgr, 255.0)
        return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)

    def _get_person_mask(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """세그멘테이션 마스크를 0~1 float 배열로 반환한다."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        seg_result = self._segmenter.segment(mp_image)

        if not seg_result.confidence_masks:
            return None

        mask = seg_result.confidence_masks[0].numpy_view().copy()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=4, sigmaY=4)
        return np.clip(mask, 0.0, 1.0)

    def process(
        self,
        image_bgr: np.ndarray,
        *,
        whitening: float = 0.45,
        smooth: float = 0.55,
        sharpness: float = 0.0,
        gamma: float = 1.0,
        contrast: float = 0.0,
        red: float = 0.0,
        green: float = 0.0,
        blue: float = 0.0,
        temperature: float = 0.0,
        hue: float = 0.0,
        saturation: float = 0.0,
        cutout: bool = False,
    ) -> np.ndarray:
        """이미지에 미백·스무딩·색상·선명도·누끼 효과를 적용한다."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("유효하지 않은 이미지입니다.")

        whitening = float(np.clip(whitening, 0.0, 1.0))
        smooth = float(np.clip(smooth, 0.0, 1.0))
        sharpness = float(np.clip(sharpness, 0.0, 1.0))
        gamma = float(np.clip(gamma, 0.5, 2.0))
        contrast = float(np.clip(contrast, -100.0, 100.0))
        red = float(np.clip(red, -100.0, 100.0))
        green = float(np.clip(green, -100.0, 100.0))
        blue = float(np.clip(blue, -100.0, 100.0))
        temperature = float(np.clip(temperature, -100.0, 100.0))
        hue = float(np.clip(hue, -180.0, 180.0))
        saturation = float(np.clip(saturation, -100.0, 100.0))

        skin_mask = self._build_skin_mask(image_bgr)
        if skin_mask is None:
            skin_mask = self._fallback_skin_mask(image_bgr)

        skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), sigmaX=8, sigmaY=8)
        skin_mask = np.clip(skin_mask, 0.0, 1.0)

        result = image_bgr.astype(np.float32)

        if smooth > 0.01:
            result = self._smooth_skin(result, skin_mask, smooth)

        if whitening > 0.01:
            result = self._whiten_skin(result, skin_mask, whitening)

        if self._color_adjustment_needed(
            gamma, contrast, red, green, blue, temperature, hue, saturation
        ):
            result = self._apply_color_adjustments(
                result,
                gamma=gamma,
                contrast=contrast,
                red=red,
                green=green,
                blue=blue,
                temperature=temperature,
                hue=hue,
                saturation=saturation,
            )

        if sharpness > 0.01:
            result = self._sharpen_image(result, sharpness)

        result = np.clip(result, 0, 255).astype(np.uint8)

        if cutout:
            result = self._apply_cutout(result)

        return result

    @staticmethod
    def _color_adjustment_needed(
        gamma: float,
        contrast: float,
        red: float,
        green: float,
        blue: float,
        temperature: float,
        hue: float,
        saturation: float,
    ) -> bool:
        return (
            abs(gamma - 1.0) > 0.01
            or abs(contrast) > 0.5
            or abs(red) > 0.5
            or abs(green) > 0.5
            or abs(blue) > 0.5
            or abs(temperature) > 0.5
            or abs(hue) > 0.5
            or abs(saturation) > 0.5
        )

    @staticmethod
    def _apply_color_adjustments(
        image: np.ndarray,
        *,
        gamma: float,
        contrast: float,
        red: float,
        green: float,
        blue: float,
        temperature: float,
        hue: float,
        saturation: float,
    ) -> np.ndarray:
        """감마·콘트라스트·RGB·색온도·색조·채도를 적용한다."""
        result = image.astype(np.float32)

        if abs(gamma - 1.0) > 0.01:
            normalized = np.clip(result / 255.0, 0.0, 1.0)
            result = np.power(normalized, 1.0 / gamma) * 255.0

        if abs(contrast) > 0.5:
            factor = 1.0 + contrast / 100.0
            result = (result - 128.0) * factor + 128.0

        if abs(red) > 0.5 or abs(green) > 0.5 or abs(blue) > 0.5:
            offsets = np.array([blue, green, red], dtype=np.float32) * 0.4
            result += offsets

        if abs(temperature) > 0.5:
            temp = temperature / 100.0
            result[:, :, 2] *= 1.0 + temp * 0.25
            result[:, :, 0] *= 1.0 - temp * 0.25

        result = np.clip(result, 0, 255)

        if abs(hue) > 0.5 or abs(saturation) > 0.5:
            hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            if abs(hue) > 0.5:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue / 2.0) % 180.0
            if abs(saturation) > 0.5:
                sat_factor = 1.0 + saturation / 100.0
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        return result

    def _build_skin_mask(self, image_bgr: np.ndarray) -> np.ndarray | None:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        mask = np.zeros((h, w), dtype=np.uint8)
        for landmarks in result.face_landmarks:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            self._fill_polygon(face_mask, landmarks, self._FACE_OVAL, w, h, 255)

            for region in (self._LEFT_EYE, self._RIGHT_EYE, self._LIPS):
                hole = np.zeros((h, w), dtype=np.uint8)
                self._fill_polygon(hole, landmarks, region, w, h, 255)
                face_mask = cv2.subtract(face_mask, hole)

            mask = cv2.bitwise_or(mask, face_mask)

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=12, sigmaY=12)
        return mask.astype(np.float32) / 255.0

    @staticmethod
    def _fill_polygon(
        canvas: np.ndarray,
        landmarks,
        indices: list[int],
        width: int,
        height: int,
        value: int,
    ) -> None:
        points = np.array(
            [[int(lm.x * width), int(lm.y * height)] for lm in (landmarks[i] for i in indices)],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], value)

    @staticmethod
    def _fallback_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
        """얼굴 미검출 시 YCrCb 기반 피부색 마스크."""
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)
        return mask.astype(np.float32) / 255.0

    def _apply_cutout(self, image_bgr: np.ndarray) -> np.ndarray:
        """인물 영역만 남기고 배경을 투명하게 제거한다."""
        mask = self._get_person_mask(image_bgr)
        if mask is None:
            bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = 255
            return bgra

        alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    @staticmethod
    def _smooth_skin(
        image: np.ndarray,
        skin_mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """양방향 필터로 틱톡 스타일 피부 스무딩."""
        d = int(5 + strength * 10)
        sigma_color = 30 + strength * 50
        sigma_space = 30 + strength * 50
        smoothed = cv2.bilateralFilter(
            image.astype(np.uint8),
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        ).astype(np.float32)

        detail = image - smoothed
        preserve = 1.0 - strength * 0.65
        smoothed = smoothed + detail * preserve

        blend = (skin_mask * (0.35 + strength * 0.65))[..., None]
        return image * (1.0 - blend) + smoothed * blend

    @staticmethod
    def _whiten_skin(
        image: np.ndarray,
        skin_mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """LAB 색공간에서 피부 밝기를 자연스럽게 올리는 미백."""
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]

        lift = 8.0 + strength * 28.0
        whitened_l = np.clip(l_channel + lift, 0, 255)

        adaptive = 1.0 - (l_channel / 255.0) * 0.35
        blend_l = skin_mask * adaptive * (0.4 + strength * 0.6)
        lab[:, :, 0] = l_channel * (1.0 - blend_l) + whitened_l * blend_l

        neutral_pull = skin_mask * strength * 0.12
        lab[:, :, 1] = lab[:, :, 1] * (1.0 - neutral_pull) + 128.0 * neutral_pull
        lab[:, :, 2] = lab[:, :, 2] * (1.0 - neutral_pull) + 128.0 * neutral_pull

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - skin_mask * strength * 0.08), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + skin_mask * strength * 6.0, 0, 255)
        toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        global_blend = skin_mask[..., None] * (0.25 + strength * 0.5)
        return image * (1.0 - global_blend) + toned * global_blend

    @staticmethod
    def _sharpen_image(image: np.ndarray, strength: float) -> np.ndarray:
        """언샤프 마스크로 전체 이미지 선명도를 조절한다."""
        sigma = 1.0 + strength * 2.5
        blurred = cv2.GaussianBlur(
            image.astype(np.uint8),
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        ).astype(np.float32)
        detail = image - blurred
        amount = 0.4 + strength * 2.2
        return image + detail * amount
