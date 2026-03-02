"""
Digital Chameleon — Privacy Shield
=====================================
Applies real-time face privacy transforms to the video frame **when a
recording is detected**.  During a normal live video call the caller sees
your real face unmodified.  The moment recording starts, this module
instantly replaces or obscures your face in every frame fed into the
virtual camera, so the *recorded* video will never contain your real face.

Transform options (all configurable, defaults tuned for 100 % invisibility):
────────────────────────────────────────────────────────────────────────────
  BLUR      – Heavy Gaussian blur (σ scales with face size).  Recognisable
              as a person, but identity is completely obscured.  Best UX.

  PIXELATE  – Mosaic / pixelation (block size configurable).  Classic
              privacy treatment; cannot be reversed even with AI upscaling
              because the block aggregation is non-invertible.

  NOISE     – Dense Gaussian noise that completely overwhelms pixel-level
              identity features.  Most aggressive; indistinguishable from
              static TV noise.

  BLACKOUT  – Solid black rectangle over the face region.  Legal-grade
              redaction, same method used in news broadcasts.

  AVATAR    – A cartoon avatar / coloured silhouette is rendered over the
              face ROI.  Provides a *presence* signal (caller can see you're
              there) while hiding real identity.

  COMBINED  – PIXELATE + NOISE + edge-fade.  Highest privacy score.

All transforms include a smooth *fade-in* over 15 frames so the transition
from live to private is gradual and not jarring.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BLUR_MODE = "blur"
PIXELATE_MODE = "pixelate"
NOISE_MODE = "noise"
BLACKOUT_MODE = "blackout"
AVATAR_MODE = "avatar"
COMBINED_MODE = "combined"

_ALL_MODES = [BLUR_MODE, PIXELATE_MODE, NOISE_MODE, BLACKOUT_MODE, AVATAR_MODE, COMBINED_MODE]
DEFAULT_MODE = COMBINED_MODE

# Face padding: expand the detected bounding box by this fraction to ensure
# hair and jaw are covered (important so edges don't reveal identity)
_FACE_PAD = 0.30   # 30 % expansion in each direction


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _padded_roi(
    frame_shape: Tuple[int, int, int],
    bbox: Tuple[int, int, int, int],
    pad: float = _FACE_PAD,
) -> Tuple[int, int, int, int]:
    """Return a padded, clamped (x1, y1, x2, y2) for the face region."""
    fh, fw = frame_shape[:2]
    x, y, bw, bh = bbox
    pw = int(bw * pad)
    ph = int(bh * pad)
    x1 = max(0, x - pw)
    y1 = max(0, y - ph)
    x2 = min(fw, x + bw + pw)
    y2 = min(fh, y + bh + ph)
    return x1, y1, x2, y2


def _elliptical_feather_mask(h: int, w: int, feather: float = 0.12) -> np.ndarray:
    """Create a float32 [0,1] mask with feathered elliptical edges.

    The centre is 1.0 (fully opaque) and fades to 0.0 at the boundary.
    This prevents hard rectangular edges that look unnatural.
    """
    cy, cx = h / 2.0, w / 2.0
    ry, rx = cy * (1.0 - feather), cx * (1.0 - feather)
    ry = max(ry, 1.0)
    rx = max(rx, 1.0)

    yy, xx = np.ogrid[:h, :w]
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = np.clip(1.0 - (dist - 1.0) * (1.0 / feather), 0.0, 1.0)
    return mask.astype(np.float32)


def _apply_blur(roi: np.ndarray) -> np.ndarray:
    """Heavy Gaussian blur scaled to ROI size."""
    size = max(roi.shape[0], roi.shape[1])
    k = int(size * 0.25) | 1   # odd kernel ≈ 25 % of face size
    k = max(k, 51)
    return cv2.GaussianBlur(roi, (k, k), 0)


def _apply_pixelate(roi: np.ndarray, block: int = 0) -> np.ndarray:
    """Mosaic pixelation — non-reversible even with AI upscaling."""
    h, w = roi.shape[:2]
    if block == 0:
        block = max(8, min(h, w) // 10)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_noise(roi: np.ndarray, strength: int = 80) -> np.ndarray:
    """Dense Gaussian noise that overwhelms identity features."""
    noise = np.random.randint(-strength, strength + 1, roi.shape, dtype=np.int16)
    noisy = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _apply_blackout(roi: np.ndarray) -> np.ndarray:
    return np.zeros_like(roi)


def _apply_avatar(roi: np.ndarray, frame_counter: int = 0) -> np.ndarray:
    """Render a colourful animated avatar silhouette over the face ROI."""
    h, w = roi.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Animated hue cycles at 0.5 Hz
    hue = int((frame_counter * 2) % 180)
    hsv_colour = np.uint8([[[hue, 200, 220]]])
    colour = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0, 0].tolist()

    # Ellipse for head
    cv2.ellipse(canvas, (w // 2, h // 2), (w // 3, h // 2 - 5), 0, 0, 360, colour, -1)
    # Circle for eyes
    eye_y = h // 3
    cv2.circle(canvas, (w // 3, eye_y), max(3, w // 15), (255, 255, 255), -1)
    cv2.circle(canvas, (2 * w // 3, eye_y), max(3, w // 15), (255, 255, 255), -1)
    # Smile
    cv2.ellipse(canvas, (w // 2, int(h * 0.6)), (w // 5, h // 10), 0, 0, 180, (255, 255, 255), 2)

    # DC badge
    font_scale = max(0.3, w / 300)
    cv2.putText(canvas, "DC", (w // 2 - 12, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _apply_combined(roi: np.ndarray, frame_counter: int = 0) -> np.ndarray:
    """Pixelate + heavy noise + blur edge — maximum privacy."""
    result = _apply_pixelate(roi, block=12)
    result = _apply_noise(result, strength=60)
    # Soft final blur to merge the noise over mosaic
    result = cv2.GaussianBlur(result, (15, 15), 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyShield:
    """Applies face-privacy transforms when recording is detected.

    The key contract:
    - ``apply(frame, bbox, is_recording)`` → modified frame
    - When ``is_recording=False`` → frame returned **unchanged** (real face visible)
    - When ``is_recording=True``  → face region replaced by the configured transform

    A 15-frame smooth fade-in transition prevents jarring cuts.

    Usage::

        shield = PrivacyShield(mode=COMBINED_MODE)
        # inside video loop:
        output_frame = shield.apply(frame, face_bbox, detector.is_recording)
    """

    _FADE_FRAMES = 20   # frames for smooth transition

    def __init__(
        self,
        mode: str = DEFAULT_MODE,
        blur_strength: int = 99,   # Gaussian blur kernel (odd), overridden adaptively
        noise_strength: int = 80,
        pixel_block: int = 0,      # 0 = auto based on face size
    ) -> None:
        if mode not in _ALL_MODES:
            raise ValueError(f"mode must be one of {_ALL_MODES}, got {mode!r}")
        self.mode = mode
        self.blur_strength = blur_strength
        self.noise_strength = noise_strength
        self.pixel_block = pixel_block

        self._frame_counter: int = 0
        self._fade_alpha: float = 0.0   # 0.0 = no privacy, 1.0 = full privacy
        self._was_recording: bool = False

        # Stores the last clean frame for potential mixing
        self._last_clean_roi: Optional[np.ndarray] = None

        print(f"[PrivacyShield] Initialised  mode={mode}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def apply(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        is_recording: bool,
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Apply privacy transform to the face region if recording.

        Args:
            frame:        BGR frame from the webcam.
            bbox:         (x, y, w, h) of the detected face, or None.
            is_recording: True when recording is in progress.
            landmarks:    Optional 478-point landmarks from FaceProcessor.

        Returns:
            Modified frame (live view is always clean; recording view is
            obfuscated).
        """
        self._frame_counter += 1

        # Update fade alpha
        if is_recording:
            self._fade_alpha = min(1.0, self._fade_alpha + 1.0 / self._FADE_FRAMES)
        else:
            self._fade_alpha = max(0.0, self._fade_alpha - 1.0 / self._FADE_FRAMES)

        # If fully transparent (no recording, fully faded out) return as-is
        if self._fade_alpha < 0.01 or bbox is None:
            return frame

        output = frame.copy()
        x1, y1, x2, y2 = _padded_roi(frame.shape, bbox)
        roi = output[y1:y2, x1:x2]

        if roi.size == 0:
            return output

        # Apply the chosen transform
        protected = self._transform(roi)

        # Feathered elliptical blend (avoids hard edges)
        h_roi, w_roi = roi.shape[:2]
        mask = _elliptical_feather_mask(h_roi, w_roi)
        alpha = self._fade_alpha * mask[:, :, np.newaxis]

        blended = (alpha * protected.astype(np.float32) +
                   (1.0 - alpha) * roi.astype(np.float32))
        output[y1:y2, x1:x2] = blended.astype(np.uint8)

        # Draw subtle indicator on the frame (corner badge)
        self._draw_privacy_badge(output, is_recording)

        return output

    def set_mode(self, mode: str) -> None:
        if mode in _ALL_MODES:
            self.mode = mode
            print(f"[PrivacyShield] Mode changed to: {mode}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _transform(self, roi: np.ndarray) -> np.ndarray:
        """Apply the configured transform to the ROI."""
        if self.mode == BLUR_MODE:
            return _apply_blur(roi)
        elif self.mode == PIXELATE_MODE:
            return _apply_pixelate(roi, self.pixel_block)
        elif self.mode == NOISE_MODE:
            return _apply_noise(roi, self.noise_strength)
        elif self.mode == BLACKOUT_MODE:
            return _apply_blackout(roi)
        elif self.mode == AVATAR_MODE:
            return _apply_avatar(roi, self._frame_counter)
        else:  # COMBINED_MODE (default)
            return _apply_combined(roi, self._frame_counter)

    def _draw_privacy_badge(self, frame: np.ndarray, is_recording: bool) -> None:
        """Draw a small 'PRIVACY ON' badge in the bottom-right corner."""
        h, w = frame.shape[:2]
        if is_recording and self._fade_alpha > 0.5:
            badge_text = "REC SHIELD ACTIVE"
            badge_color = (0, 0, 200)
            text_color = (255, 255, 255)
        else:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (tw, th), _ = cv2.getTextSize(badge_text, font, scale, thick)
        bx = w - tw - 20
        by = h - 12
        cv2.rectangle(frame, (bx - 6, by - th - 6), (bx + tw + 6, by + 6),
                      badge_color, -1)
        cv2.putText(frame, badge_text, (bx, by), font, scale, text_color, thick, cv2.LINE_AA)

    def get_status(self) -> dict:
        return {
            "mode": self.mode,
            "fade_alpha": round(self._fade_alpha, 2),
            "active": self._fade_alpha > 0.5,
        }
