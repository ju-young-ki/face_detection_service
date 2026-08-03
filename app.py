"""증명사진 미백·누끼 - Windows 데스크톱 앱."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from defaults_config import get_defaults, save_defaults
from processor import PhotoProcessor

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class WhiteningApp(ctk.CTk):
  PREVIEW_MAX = 720

  def __init__(self) -> None:
    super().__init__()

    self.title("증명사진 미백 · 누끼")
    self.geometry("1100x760")
    self.minsize(900, 640)

    self._processor = PhotoProcessor()
    self._source_bgr: np.ndarray | None = None
    self._result_bgr: np.ndarray | None = None
    self._source_path: str | None = None
    self._preview_photo: ImageTk.PhotoImage | None = None
    self._processing = False
    self._after_id: str | None = None
    self._show_face_boxes = False
    self._pending_passport_export = False
    self._value_label_refreshers: list[callable] = []

    self._build_ui()
    self.protocol("WM_DELETE_WINDOW", self._on_close)

  def _build_ui(self) -> None:
    self.grid_columnconfigure(0, weight=1)
    self.grid_rowconfigure(1, weight=1)

    header = ctk.CTkFrame(self, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))
    header.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
      header,
      text="증명사진 미백 · 누끼",
      font=ctk.CTkFont(size=24, weight="bold"),
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkLabel(
      header,
      text="틱톡 카메라 뷰티 필터 스타일 · 피부 보정 및 배경 제거",
      font=ctk.CTkFont(size=13),
      text_color="#666666",
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    body = ctk.CTkFrame(self, fg_color="transparent")
    body.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
    body.grid_columnconfigure(0, weight=3)
    body.grid_columnconfigure(1, weight=1)
    body.grid_rowconfigure(0, weight=1)

    preview_frame = ctk.CTkFrame(body)
    preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    preview_frame.grid_rowconfigure(1, weight=1)
    preview_frame.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
      preview_frame,
      text="미리보기 (왼쪽: 원본 / 오른쪽: 보정)",
      font=ctk.CTkFont(size=14, weight="bold"),
    ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

    self._preview_label = ctk.CTkLabel(
      preview_frame,
      text="이미지를 열어주세요\n\n[이미지 열기] 버튼을 눌러주세요",
      font=ctk.CTkFont(size=15),
      text_color="#888888",
    )
    self._preview_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    panel = ctk.CTkFrame(body, width=300)
    panel.grid(row=0, column=1, sticky="nsew")
    panel.grid_columnconfigure(0, weight=1)
    panel.grid_rowconfigure(1, weight=1)

    ctk.CTkLabel(
      panel,
      text="효과 조절",
      font=ctk.CTkFont(size=16, weight="bold"),
    ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

    controls = ctk.CTkScrollableFrame(panel, fg_color="transparent")
    controls.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
    controls.grid_columnconfigure(0, weight=1)

    defaults = get_defaults()

    self._whitening_var = tk.DoubleVar(value=float(defaults["whitening"]))
    self._smooth_var = tk.DoubleVar(value=float(defaults["smooth"]))
    self._sharpness_var = tk.DoubleVar(value=float(defaults["sharpness"]))
    self._gamma_var = tk.DoubleVar(value=float(defaults["gamma"]))
    self._contrast_var = tk.DoubleVar(value=float(defaults["contrast"]))
    self._red_var = tk.DoubleVar(value=float(defaults["red"]))
    self._green_var = tk.DoubleVar(value=float(defaults["green"]))
    self._blue_var = tk.DoubleVar(value=float(defaults["blue"]))
    self._temperature_var = tk.DoubleVar(value=float(defaults["temperature"]))
    self._hue_var = tk.DoubleVar(value=float(defaults["hue"]))
    self._saturation_var = tk.DoubleVar(value=float(defaults["saturation"]))
    self._forehead_shine_var = tk.DoubleVar(value=float(defaults.get("forehead_shine", 0.0)))
    self._density_var = tk.DoubleVar(value=float(defaults.get("density", 0.0)))
    self._cutout_var = tk.BooleanVar(value=bool(defaults["cutout"]))
    self._head_margin_var = tk.DoubleVar(value=float(defaults["top_margin_mm"]))

    row = 0
    self._add_slider(controls, row, "미백", self._whitening_var, "피부를 밝고 깨끗하게")
    row += 3
    self._add_slider(controls, row, "스무딩", self._smooth_var, "모공·잔주름 부드럽게")
    row += 3
    self._add_slider(
      controls,
      row,
      "이마 광 제거",
      self._forehead_shine_var,
      "이마 하이라이트·번들거림 완화",
    )
    row += 3
    self._add_slider(controls, row, "진하게", self._density_var, "톤을 누르고 색을 진하게")
    row += 3
    self._add_slider(controls, row, "선명도", self._sharpness_var, "윤곽과 디테일을 또렷하게")
    row += 3

    ctk.CTkLabel(
      controls,
      text="색상 조절",
      font=ctk.CTkFont(size=14, weight="bold"),
    ).grid(row=row, column=0, sticky="w", padx=8, pady=(12, 4))
    row += 1

    self._add_gamma_slider(controls, row, "감마", self._gamma_var, "중간 톤 밝기 조절")
    row += 3
    self._add_balance_slider(controls, row, "콘트라스트", self._contrast_var, "밝고 어두운 대비")
    row += 3
    self._add_balance_slider(controls, row, "빨강 (R)", self._red_var, "Red 채널")
    row += 3
    self._add_balance_slider(controls, row, "초록 (G)", self._green_var, "Green 채널")
    row += 3
    self._add_balance_slider(controls, row, "파랑 (B)", self._blue_var, "Blue 채널")
    row += 3
    self._add_balance_slider(
      controls, row, "색온도", self._temperature_var, "차갑게 ← → 따뜻하게"
    )
    row += 3
    self._add_balance_slider(
      controls, row, "색조", self._hue_var, "전체 색상 톤 회전", minimum=-180, maximum=180
    )
    row += 3
    self._add_balance_slider(controls, row, "채도", self._saturation_var, "색 선명도")
    row += 3

    self._add_mm_slider(
      controls,
      row,
      "머리끝 여백",
      self._head_margin_var,
      "여권·증명사진 상단 여백",
      minimum=0.0,
      maximum=10.0,
    )
    row += 3

    cutout_frame = ctk.CTkFrame(controls, fg_color="transparent")
    cutout_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(12, 0))
    cutout_frame.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
      cutout_frame,
      text="누끼",
      font=ctk.CTkFont(size=14, weight="bold"),
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkSwitch(
      cutout_frame,
      text="배경 제거",
      variable=self._cutout_var,
      command=self._schedule_process,
    ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    ctk.CTkLabel(
      cutout_frame,
      text="인물만 남기고 배경을 흰색으로",
      font=ctk.CTkFont(size=11),
      text_color="#888888",
    ).grid(row=2, column=0, sticky="w", pady=(4, 0))
    row += 1

    preset_frame = ctk.CTkFrame(controls, fg_color="transparent")
    preset_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
    preset_frame.grid_columnconfigure((0, 1), weight=1)

    ctk.CTkButton(
      preset_frame,
      text="자연스럽게",
      command=lambda: self._apply_preset(0.30, 0.40),
    ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

    ctk.CTkButton(
      preset_frame,
      text="틱톡 스타일",
      command=lambda: self._apply_preset(0.55, 0.65),
    ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    ctk.CTkButton(
      preset_frame,
      text="증명사진",
      command=self._export_passport_photo,
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    ctk.CTkButton(
      preset_frame,
      text="얼굴 표시",
      fg_color="#C0392B",
      hover_color="#922B21",
      command=self._toggle_face_boxes,
    ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    row += 1

    self._status_label = ctk.CTkLabel(
      controls,
      text="준비됨",
      font=ctk.CTkFont(size=12),
      text_color="#666666",
    )
    self._status_label.grid(row=row, column=0, sticky="w", padx=8, pady=(12, 8))

    footer = ctk.CTkFrame(self, fg_color="transparent")
    footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 16))
    footer.grid_columnconfigure(5, weight=1)

    ctk.CTkButton(footer, text="이미지 열기", width=120, command=self._open_image).grid(
      row=0, column=0, padx=(0, 8)
    )
    ctk.CTkButton(footer, text="다시 처리", width=120, command=self._schedule_process).grid(
      row=0, column=1, padx=(0, 8)
    )
    self._save_btn = ctk.CTkButton(
      footer,
      text="저장하기",
      width=120,
      state="disabled",
      command=self._save_image,
    )
    self._save_btn.grid(row=0, column=2, padx=(0, 8))

    ctk.CTkButton(
      footer,
      text="초기화",
      width=100,
      fg_color="#888888",
      hover_color="#666666",
      command=self._reset_sliders,
    ).grid(row=0, column=3, padx=(0, 8))

    ctk.CTkButton(
      footer,
      text="설정 저장",
      width=100,
      fg_color="#2E7D32",
      hover_color="#1B5E20",
      command=self._save_settings,
    ).grid(row=0, column=4)

  def _register_value_label_refresher(self, refresher: callable) -> None:
    self._value_label_refreshers.append(refresher)

  def _refresh_value_labels(self) -> None:
    for refresher in self._value_label_refreshers:
      refresher()

  def _add_slider(
    self,
    parent: ctk.CTkFrame,
    row: int,
    label: str,
    variable: tk.DoubleVar,
    hint: str,
  ) -> None:
    ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=14, weight="bold")).grid(
      row=row, column=0, sticky="w", padx=16, pady=(8, 0)
    )
    value_label = ctk.CTkLabel(
      parent,
      text=f"{int(variable.get() * 100)}%",
      font=ctk.CTkFont(size=12),
    )
    value_label.grid(row=row, column=0, sticky="e", padx=16, pady=(8, 0))

    def to_value(pos: float) -> float:
      return float(pos) / 100.0

    def to_position(ratio: float) -> float:
      return float(ratio) * 100.0

    def refresh_label(value: float | None = None, *, sync_slider: bool = True) -> None:
      current = variable.get() if value is None else float(value)
      value_label.configure(text=f"{int(round(current * 100))}%")
      if sync_slider:
        slider.set(to_position(current))

    def on_change(pos: float) -> None:
      actual = to_value(pos)
      variable.set(actual)
      value_label.configure(text=f"{int(round(actual * 100))}%")
      self._schedule_process()

    self._register_value_label_refresher(refresh_label)

    slider = ctk.CTkSlider(
      parent,
      from_=0,
      to=100,
      number_of_steps=100,
      command=on_change,
    )
    slider.set(to_position(variable.get()))
    slider.grid(row=row + 1, column=0, sticky="ew", padx=16, pady=(4, 0))

    ctk.CTkLabel(parent, text=hint, font=ctk.CTkFont(size=11), text_color="#888888").grid(
      row=row + 2, column=0, sticky="w", padx=16, pady=(0, 4)
    )

  def _add_gamma_slider(
    self,
    parent: ctk.CTkFrame,
    row: int,
    label: str,
    variable: tk.DoubleVar,
    hint: str,
  ) -> None:
    minimum = 0.5
    maximum = 2.0
    steps = 150
    span = maximum - minimum

    ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=14, weight="bold")).grid(
      row=row, column=0, sticky="w", padx=8, pady=(8, 0)
    )
    value_label = ctk.CTkLabel(parent, text=f"{variable.get():.2f}", font=ctk.CTkFont(size=12))
    value_label.grid(row=row, column=0, sticky="e", padx=8, pady=(8, 0))

    def to_value(pos: float) -> float:
      return minimum + float(pos) / steps * span

    def to_position(gamma: float) -> float:
      return (float(gamma) - minimum) / span * steps

    def refresh_label(value: float | None = None, *, sync_slider: bool = True) -> None:
      current = variable.get() if value is None else float(value)
      value_label.configure(text=f"{current:.2f}")
      if sync_slider:
        slider.set(to_position(current))

    def on_change(pos: float) -> None:
      actual = to_value(pos)
      variable.set(actual)
      value_label.configure(text=f"{actual:.2f}")
      self._schedule_process()

    self._register_value_label_refresher(refresh_label)

    slider = ctk.CTkSlider(
      parent,
      from_=0,
      to=steps,
      number_of_steps=steps,
      command=on_change,
    )
    slider.set(to_position(variable.get()))
    slider.grid(row=row + 1, column=0, sticky="ew", padx=8, pady=(4, 0))

    ctk.CTkLabel(parent, text=hint, font=ctk.CTkFont(size=11), text_color="#888888").grid(
      row=row + 2, column=0, sticky="w", padx=8, pady=(0, 4)
    )

  def _add_balance_slider(
    self,
    parent: ctk.CTkFrame,
    row: int,
    label: str,
    variable: tk.DoubleVar,
    hint: str,
    *,
    minimum: float = -100.0,
    maximum: float = 100.0,
  ) -> None:
    span = maximum - minimum

    ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=14, weight="bold")).grid(
      row=row, column=0, sticky="w", padx=8, pady=(8, 0)
    )
    value_label = ctk.CTkLabel(
      parent,
      text=f"{variable.get():+.0f}",
      font=ctk.CTkFont(size=12),
    )
    value_label.grid(row=row, column=0, sticky="e", padx=8, pady=(8, 0))

    neutral_pos = span / 2.0

    def to_value(pos: float) -> float:
      step = round(float(pos))
      if step == round(neutral_pos):
        return 0.0
      actual = minimum + step
      return float(round(max(minimum, min(maximum, actual))))

    def to_position(display: float) -> float:
      if abs(float(display)) < 0.5:
        return neutral_pos
      return float(round(float(display) - minimum))

    def refresh_label(value: float | None = None, *, sync_slider: bool = True) -> None:
      current = variable.get() if value is None else float(value)
      if abs(current) < 0.5:
        current = 0.0
      value_label.configure(text=f"{current:+.0f}")
      if sync_slider:
        slider.set(to_position(current))

    def on_change(pos: float) -> None:
      actual = to_value(pos)
      variable.set(actual)
      value_label.configure(text=f"{actual:+.0f}")
      if actual == 0.0:
        slider.set(neutral_pos)
      self._schedule_process()

    self._register_value_label_refresher(refresh_label)

    slider = ctk.CTkSlider(
      parent,
      from_=0,
      to=span,
      number_of_steps=int(span),
      command=on_change,
    )
    slider.set(to_position(variable.get()))
    slider.grid(row=row + 1, column=0, sticky="ew", padx=8, pady=(4, 0))

    ctk.CTkLabel(parent, text=hint, font=ctk.CTkFont(size=11), text_color="#888888").grid(
      row=row + 2, column=0, sticky="w", padx=8, pady=(0, 4)
    )

  def _add_mm_slider(
    self,
    parent: ctk.CTkFrame,
    row: int,
    label: str,
    variable: tk.DoubleVar,
    hint: str,
    *,
    minimum: float,
    maximum: float,
  ) -> None:
    steps = int((maximum - minimum) * 10)

    ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=14, weight="bold")).grid(
      row=row, column=0, sticky="w", padx=16, pady=(8, 0)
    )
    value_label = ctk.CTkLabel(
      parent,
      text=f"{variable.get():.1f}mm",
      font=ctk.CTkFont(size=12),
    )
    value_label.grid(row=row, column=0, sticky="e", padx=16, pady=(8, 0))

    def to_value(pos: float) -> float:
      return minimum + float(pos) / steps * (maximum - minimum)

    def to_position(mm: float) -> float:
      if maximum == minimum:
        return 0.0
      return (float(mm) - minimum) / (maximum - minimum) * steps

    def refresh_label(value: float | None = None, *, sync_slider: bool = True) -> None:
      current = variable.get() if value is None else float(value)
      value_label.configure(text=f"{current:.1f}mm")
      if sync_slider:
        slider.set(to_position(current))

    def on_change(pos: float) -> None:
      actual = to_value(pos)
      variable.set(actual)
      value_label.configure(text=f"{actual:.1f}mm")

    self._register_value_label_refresher(refresh_label)

    slider = ctk.CTkSlider(
      parent,
      from_=0,
      to=steps,
      number_of_steps=steps,
      command=on_change,
    )
    slider.set(to_position(variable.get()))
    slider.grid(row=row + 1, column=0, sticky="ew", padx=16, pady=(4, 0))

    ctk.CTkLabel(parent, text=hint, font=ctk.CTkFont(size=11), text_color="#888888").grid(
      row=row + 2, column=0, sticky="w", padx=16, pady=(0, 4)
    )

  def _apply_preset(self, whitening: float, smooth: float) -> None:
    self._whitening_var.set(whitening)
    self._smooth_var.set(smooth)
    self._refresh_value_labels()
    self._schedule_process()

  def _toggle_face_boxes(self) -> None:
    if self._result_bgr is None:
      messagebox.showwarning("안내", "먼저 이미지를 열고 처리해주세요.")
      return

    self._show_face_boxes = not self._show_face_boxes
    self._update_preview()
    state = "켜짐" if self._show_face_boxes else "꺼짐"
    self._status_label.configure(text=f"얼굴 표시 {state}")

  def _export_passport_photo(self) -> None:
    if self._source_bgr is None:
      messagebox.showwarning("안내", "먼저 이미지를 열어주세요.")
      return

    if self._result_bgr is not None and not self._processing:
      self._save_passport_photo()
      return

    self._pending_passport_export = True
    self._status_label.configure(text="여권 사진 규격 처리 중...")
    self._schedule_process()

  def _reset_sliders(self) -> None:
    defaults = get_defaults()
    self._whitening_var.set(float(defaults["whitening"]))
    self._smooth_var.set(float(defaults["smooth"]))
    self._sharpness_var.set(float(defaults["sharpness"]))
    self._gamma_var.set(float(defaults["gamma"]))
    self._contrast_var.set(float(defaults["contrast"]))
    self._red_var.set(float(defaults["red"]))
    self._green_var.set(float(defaults["green"]))
    self._blue_var.set(float(defaults["blue"]))
    self._temperature_var.set(float(defaults["temperature"]))
    self._hue_var.set(float(defaults["hue"]))
    self._saturation_var.set(float(defaults["saturation"]))
    self._forehead_shine_var.set(float(defaults.get("forehead_shine", 0.0)))
    self._density_var.set(float(defaults.get("density", 0.0)))
    self._cutout_var.set(bool(defaults["cutout"]))
    self._head_margin_var.set(float(defaults["top_margin_mm"]))
    self._refresh_value_labels()
    self._schedule_process()

  def _current_settings(self) -> dict:
    return {
      "whitening": self._whitening_var.get(),
      "smooth": self._smooth_var.get(),
      "sharpness": self._sharpness_var.get(),
      "gamma": self._gamma_var.get(),
      "contrast": self._contrast_var.get(),
      "red": self._red_var.get(),
      "green": self._green_var.get(),
      "blue": self._blue_var.get(),
      "temperature": self._temperature_var.get(),
      "hue": self._hue_var.get(),
      "saturation": self._saturation_var.get(),
      "forehead_shine": self._forehead_shine_var.get(),
      "density": self._density_var.get(),
      "cutout": self._cutout_var.get(),
      "top_margin_mm": self._head_margin_var.get(),
    }

  def _save_settings(self) -> None:
    try:
      path = save_defaults(self._current_settings())
    except OSError as exc:
      messagebox.showerror("오류", f"설정 저장에 실패했습니다.\n{exc}")
      return

    self._status_label.configure(text=f"설정 저장됨: {path.name}")
    messagebox.showinfo(
      "설정 저장 완료",
      f"현재 슬라이더 값이 기본 설정으로 저장되었습니다.\n\n{path}",
    )

  def _open_image(self) -> None:
    path = filedialog.askopenfilename(
      title="증명사진 선택",
      filetypes=[
        ("이미지", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("모든 파일", "*.*"),
      ],
    )
    if path:
      self._load_image(path)

  def _load_image(self, path: str) -> None:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
      messagebox.showerror("오류", "이미지를 열 수 없습니다.")
      return

    self._source_bgr = image
    self._source_path = path
    self._show_face_boxes = False
    self._pending_passport_export = False
    self._status_label.configure(text=f"불러옴: {os.path.basename(path)}")
    self._schedule_process()

  def _schedule_process(self) -> None:
    if self._source_bgr is None:
      return

    if self._after_id is not None:
      self.after_cancel(self._after_id)

    self._after_id = self.after(180, self._run_process)

  def _run_process(self) -> None:
    if self._source_bgr is None or self._processing:
      return

    self._processing = True
    self._status_label.configure(text="처리 중...")
    source = self._source_bgr.copy()
    whitening = self._whitening_var.get()
    smooth = self._smooth_var.get()
    sharpness = self._sharpness_var.get()
    gamma = self._gamma_var.get()
    contrast = self._contrast_var.get()
    red = self._red_var.get()
    green = self._green_var.get()
    blue = self._blue_var.get()
    temperature = self._temperature_var.get()
    hue = self._hue_var.get()
    saturation = self._saturation_var.get()
    forehead_shine = self._forehead_shine_var.get()
    density = self._density_var.get()
    cutout = self._cutout_var.get()

    def work() -> None:
      try:
        result = self._processor.process(
          source,
          whitening=whitening,
          smooth=smooth,
          sharpness=sharpness,
          gamma=gamma,
          contrast=contrast,
          red=red,
          green=green,
          blue=blue,
          temperature=temperature,
          hue=hue,
          saturation=saturation,
          forehead_shine=forehead_shine,
          density=density,
          cutout=cutout,
        )
        self.after(0, lambda: self._on_process_done(result, None))
      except Exception as exc:
        self.after(0, lambda: self._on_process_done(None, str(exc)))

    threading.Thread(target=work, daemon=True).start()

  def _on_process_done(self, result: np.ndarray | None, error: str | None) -> None:
    self._processing = False

    if error:
      self._status_label.configure(text=f"오류: {error}")
      messagebox.showerror("처리 오류", error)
      return

    self._result_bgr = result
    self._update_preview()
    self._save_btn.configure(state="normal")

    if self._pending_passport_export:
      self._pending_passport_export = False
      self._save_passport_photo()
      return

    self._status_label.configure(text="처리 완료")

  def _get_right_preview_image(self) -> np.ndarray:
    """보정 미리보기/저장용 이미지. 얼굴 표시가 켜져 있으면 사각형을 포함한다."""
    if self._result_bgr is None:
      raise ValueError("처리된 이미지가 없습니다.")

    image = self._composite_for_preview(self._result_bgr)
    if self._show_face_boxes:
      image = self._processor.draw_face_boxes(image, detect_from=self._source_bgr)
    return image

  def _update_preview(self) -> None:
    if self._source_bgr is None or self._result_bgr is None:
      return

    left = self._source_bgr
    right = self._get_right_preview_image()
    combined = self._make_side_by_side(left, right)
    preview = self._resize_for_preview(combined)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    self._preview_photo = ImageTk.PhotoImage(pil_image)
    self._preview_label.configure(image=self._preview_photo, text="")

  @staticmethod
  def _composite_for_preview(image: np.ndarray) -> np.ndarray:
    """투명 배경 이미지는 흰색 배경 위에 합성해 미리보기용 BGR로 변환."""
    return PhotoProcessor.composite_on_white(image)

  @staticmethod
  def _make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    divider_x = left.shape[1]
    cv2.line(canvas, (divider_x, 0), (divider_x, h - 1), (220, 220, 220), 2)
    cv2.putText(
      canvas,
      "원본",
      (16, 32),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )
    cv2.putText(
      canvas,
      "보정",
      (divider_x + 16, 32),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )
    return canvas

  def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(self.PREVIEW_MAX / w, self.PREVIEW_MAX / h, 1.0)
    if scale >= 1.0:
      return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

  def _save_passport_photo(self) -> None:
    if self._result_bgr is None:
      return

    try:
      passport = self._processor.crop_passport_photo(
        self._result_bgr,
        top_margin_mm=self._head_margin_var.get(),
      )
      if self._show_face_boxes:
        passport = self._processor.draw_face_boxes(passport)
    except ValueError as exc:
      self._status_label.configure(text="여권 사진 저장 실패")
      messagebox.showerror("여권 사진", str(exc))
      return

    default_name = "passport_photo.jpg"
    if self._source_path:
      base, _ = os.path.splitext(os.path.basename(self._source_path))
      default_name = f"{base}_passport.jpg"

    path = filedialog.asksaveasfilename(
      title="여권 사진 저장 (35×45mm)",
      defaultextension=".jpg",
      initialfile=default_name,
      filetypes=[
        ("JPEG", "*.jpg"),
        ("모든 파일", "*.*"),
      ],
    )
    if not path:
      self._status_label.configure(text="여권 사진 저장 취소됨")
      return

    base, ext = os.path.splitext(path)
    jpg_path = path if ext.lower() in (".jpg", ".jpeg") else f"{base}.jpg"

    try:
      ok, encoded = cv2.imencode(".jpg", passport, [cv2.IMWRITE_JPEG_QUALITY, 95])
      if not ok:
        raise RuntimeError("JPG 저장에 실패했습니다.")
      encoded.tofile(jpg_path)
    except Exception as exc:
      messagebox.showerror("오류", f"여권 사진 저장에 실패했습니다.\n{exc}")
      return

    self._status_label.configure(text=f"여권 사진 저장됨: {os.path.basename(jpg_path)}")
    messagebox.showinfo(
      "여권 사진 저장 완료",
      f"35×45mm 여권 사진 규격으로 저장되었습니다.\n\n"
      f"JPG: {jpg_path}\n"
      f"크기: {passport.shape[1]}×{passport.shape[0]}px (300dpi)",
    )

  def _save_image(self) -> None:
    if self._result_bgr is None:
      return

    default_name = "whitened_photo.jpg"
    if self._source_path:
      base, _ = os.path.splitext(os.path.basename(self._source_path))
      default_name = f"{base}_whitened.jpg"

    path = filedialog.asksaveasfilename(
      title="보정 이미지 저장",
      defaultextension=".jpg",
      initialfile=default_name,
      filetypes=[
        ("JPEG", "*.jpg"),
        ("모든 파일", "*.*"),
      ],
    )
    if not path:
      return

    base, ext = os.path.splitext(path)
    jpg_path = path if ext.lower() in (".jpg", ".jpeg") else f"{base}.jpg"

    try:
      jpg_image = self._get_right_preview_image()
      ok, encoded = cv2.imencode(".jpg", jpg_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
      if not ok:
        raise RuntimeError("JPG 저장에 실패했습니다.")
      encoded.tofile(jpg_path)
    except Exception as exc:
      messagebox.showerror("오류", f"이미지 저장에 실패했습니다.\n{exc}")
      return

    self._status_label.configure(text=f"저장됨: {os.path.basename(jpg_path)}")
    messagebox.showinfo("저장 완료", f"이미지가 저장되었습니다.\n\nJPG: {jpg_path}")

  def _on_close(self) -> None:
    self._processor.close()
    self.destroy()


def main() -> None:
    app = WhiteningApp()
    app.mainloop()


if __name__ == "__main__":
    main()
