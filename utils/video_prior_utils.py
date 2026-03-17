import os
from typing import Dict, List

import numpy as np
from PIL import Image

from utils.image_utils import crop_img

try:
    import cv2
except ImportError:
    cv2 = None


class TemporalPriorExtractor:
    """
    Build a 3-channel color temporal prior for video-frame restoration.

    The network-facing prior is a color RGB guidance image. Internally, the
    extractor also computes artifact/stability maps for deterministic filtering
    and debugging.

    temporal_semantics:
      - "transient": assumes rain/snow are transient in time (original behavior)
      - "stable": assumes rain/snow occupy stable temporal bands and should be
        suppressed via temporal band filtering
    """

    def __init__(
        self,
        frame_paths: List[str],
        window_size: int = 9,
        align_mode: str = "none",
        warp_mode: str = "affine",
        temporal_semantics: str = "transient",
        stable_band_ratio: float = 0.15,
        ecc_iterations: int = 30,
        ecc_eps: float = 1e-4,
        crop_base: int = 16,
        cache_frames: bool = True,
        cache_priors: bool = True,
    ):
        if not frame_paths:
            raise ValueError("frame_paths is empty")

        self.frame_paths = frame_paths
        self.window_size = self._normalize_window_size(window_size)
        self.radius = self.window_size // 2
        self.align_mode = align_mode.lower()
        self.warp_mode = warp_mode.lower()
        self.temporal_semantics = temporal_semantics.lower()
        self.stable_band_ratio = float(stable_band_ratio)
        self.ecc_iterations = int(ecc_iterations)
        self.ecc_eps = float(ecc_eps)
        self.crop_base = int(crop_base)
        self.cache_frames = bool(cache_frames)
        self.cache_priors = bool(cache_priors)

        self._frame_cache: Dict[int, np.ndarray] = {}
        self._prior_cache: Dict[int, np.ndarray] = {}
        self._prior_bg_cache: Dict[int, tuple] = {}
        self._prior_detail_cache: Dict[int, tuple] = {}

        if self.align_mode not in {"none", "ecc"}:
            raise ValueError("Unsupported align_mode: {}".format(self.align_mode))
        if self.warp_mode not in {"translation", "euclidean", "affine", "homography"}:
            raise ValueError("Unsupported warp_mode: {}".format(self.warp_mode))
        if self.temporal_semantics not in {"transient", "stable"}:
            raise ValueError("Unsupported temporal_semantics: {}".format(self.temporal_semantics))
        if not (0.0 < self.stable_band_ratio < 0.5):
            raise ValueError("stable_band_ratio must be in (0, 0.5), got {}".format(self.stable_band_ratio))
        if self.align_mode == "ecc" and cv2 is None:
            raise ImportError("align_mode='ecc' requires OpenCV (cv2) to be installed")

    @staticmethod
    def _normalize_window_size(window_size: int) -> int:
        size = max(int(window_size), 3)
        if size % 2 == 0:
            size += 1
        return size

    def _read_frame(self, index: int) -> np.ndarray:
        image = np.array(Image.open(self.frame_paths[index]).convert("RGB"))
        return crop_img(image, base=self.crop_base)

    def get_frame(self, index: int) -> np.ndarray:
        if self.cache_frames and index in self._frame_cache:
            return self._frame_cache[index]

        frame = self._read_frame(index)
        if self.cache_frames:
            self._frame_cache[index] = frame
        return frame

    @staticmethod
    def _reflect_index(index: int, total: int) -> int:
        if total <= 1:
            return 0
        while index < 0 or index >= total:
            if index < 0:
                index = -index
            if index >= total:
                index = 2 * total - index - 2
        return index

    def _window_indices(self, center_index: int) -> List[int]:
        total = len(self.frame_paths)
        indices = []
        for offset in range(-self.radius, self.radius + 1):
            idx = self._reflect_index(center_index + offset, total)
            indices.append(idx)
        return indices

    def _motion_mode(self):
        if self.warp_mode == "translation":
            return cv2.MOTION_TRANSLATION
        if self.warp_mode == "euclidean":
            return cv2.MOTION_EUCLIDEAN
        if self.warp_mode == "affine":
            return cv2.MOTION_AFFINE
        return cv2.MOTION_HOMOGRAPHY

    def _align_frames_to_center(self, frames: List[np.ndarray], center_local_index: int) -> List[np.ndarray]:
        if self.align_mode != "ecc":
            return frames

        aligned_frames = []
        ref = frames[center_local_index]
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = ref_gray.shape
        motion_mode = self._motion_mode()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_eps,
        )

        for idx, frame in enumerate(frames):
            if idx == center_local_index:
                aligned_frames.append(frame)
                continue

            cur_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            if motion_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray,
                    cur_gray,
                    warp_matrix,
                    motion_mode,
                    criteria,
                )
                if motion_mode == cv2.MOTION_HOMOGRAPHY:
                    aligned = cv2.warpPerspective(
                        frame,
                        warp_matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                else:
                    aligned = cv2.warpAffine(
                        frame,
                        warp_matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                aligned_frames.append(aligned)
            except cv2.error:
                aligned_frames.append(frame)

        return aligned_frames

    @staticmethod
    def _normalize_map(map_2d: np.ndarray, percentile: float = 95.0) -> np.ndarray:
        map_2d = np.maximum(map_2d.astype(np.float32), 0.0)
        scale = np.percentile(map_2d, percentile) + 1e-6
        return np.clip(map_2d / scale, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _compute_motion_reliability(cls, aligned_stack: np.ndarray, center_local_index: int):
        center = aligned_stack[center_local_index].astype(np.float32)
        temporal_delta = np.abs(aligned_stack.astype(np.float32) - center[None]).mean(axis=(0, 3))
        motion = cls._normalize_map(temporal_delta)
        reliability = np.clip(1.0 - motion, 0.15, 1.0).astype(np.float32)
        return reliability, motion

    def _compute_boundary_reliability(self, center_index: int) -> float:
        total = len(self.frame_paths)
        if self.radius <= 0 or total <= 1:
            return 1.0
        available = min(center_index, total - 1 - center_index)
        ratio = min(max(float(available) / float(self.radius), 0.0), 1.0)
        return 0.35 + 0.65 * ratio

    @classmethod
    def _build_prior_transient(cls, aligned_stack: np.ndarray, center_local_index: int):
        aligned_stack = aligned_stack.astype(np.float32)
        center = aligned_stack[center_local_index]
        bg_prior = np.median(aligned_stack, axis=0)

        residual = np.abs(center - bg_prior).mean(axis=2)
        variance = np.std(aligned_stack, axis=0).mean(axis=2)
        center_rgb = np.clip(center / 255.0, 0.0, 1.0).astype(np.float32)
        prior_rgb = np.clip(bg_prior / 255.0, 0.0, 1.0).astype(np.float32)
        artifact = cls._normalize_map(residual)
        stability = np.clip(1.0 - cls._normalize_map(variance), 0.0, 1.0)

        motion_reliability, _ = cls._compute_motion_reliability(aligned_stack, center_local_index)
        prior_rgb = prior_rgb * motion_reliability[:, :, None] + center_rgb * (1.0 - motion_reliability[:, :, None])
        artifact = np.clip(artifact * motion_reliability, 0.0, 1.0)
        stability = np.clip(stability * (0.6 + 0.4 * motion_reliability), 0.0, 1.0)
        return prior_rgb, artifact, stability

    def _temporal_fft_split(self, aligned_stack: np.ndarray):
        stack = aligned_stack.astype(np.float32)
        t = stack.shape[0]

        spectrum = np.fft.fft(stack, axis=0)
        freqs = np.abs(np.fft.fftfreq(t))
        low_mask = (freqs <= self.stable_band_ratio).reshape(t, 1, 1, 1)

        low_fft = spectrum * low_mask
        high_fft = spectrum * (~low_mask)

        low = np.fft.ifft(low_fft, axis=0).real.astype(np.float32)
        high = np.fft.ifft(high_fft, axis=0).real.astype(np.float32)
        return low, high

    def _build_prior_stable(self, aligned_stack: np.ndarray, center_local_index: int):
        """
        Stable-band prior:
          - Estimate stable temporal bands via low-pass component.
          - Suppress stable bands to form a color guidance image.
          - Use low-band dominance and temporal stability as auxiliary maps.
        """
        aligned_stack = aligned_stack.astype(np.float32)
        center = aligned_stack[center_local_index]

        low_stack, high_stack = self._temporal_fft_split(aligned_stack)
        center_low = low_stack[center_local_index]
        center_high = high_stack[center_local_index]

        # Remove stable temporal bands and keep dynamic content emphasis.
        bg_prior = np.clip(center + center_high, 0.0, 255.0)

        low_energy = np.abs(center_low).mean(axis=2)
        high_energy = np.abs(center_high).mean(axis=2)
        low_ratio = low_energy / (low_energy + high_energy + 1e-6)

        temporal_std = np.std(aligned_stack, axis=0).mean(axis=2)
        temporal_std = temporal_std / (temporal_std.max() + 1e-6)
        stability = np.clip(1.0 - temporal_std, 0.0, 1.0)

        center_rgb = np.clip(center / 255.0, 0.0, 1.0).astype(np.float32)
        prior_rgb = np.clip(bg_prior / 255.0, 0.0, 1.0).astype(np.float32)
        artifact = np.clip(np.power(low_ratio, 0.8), 0.0, 1.0).astype(np.float32)
        confidence = stability.astype(np.float32)

        motion_reliability, _ = self._compute_motion_reliability(aligned_stack, center_local_index)
        prior_rgb = prior_rgb * motion_reliability[:, :, None] + center_rgb * (1.0 - motion_reliability[:, :, None])
        artifact = np.clip(artifact * motion_reliability, 0.0, 1.0)
        confidence = np.clip(confidence * (0.5 + 0.5 * motion_reliability), 0.0, 1.0)
        return prior_rgb, artifact, confidence

    def _build_prior_from_stack(self, aligned_stack: np.ndarray, center_local_index: int):
        if self.temporal_semantics == "stable":
            return self._build_prior_stable(aligned_stack, center_local_index)
        return self._build_prior_transient(aligned_stack, center_local_index)

    def _build_background_from_stack(self, aligned_stack: np.ndarray) -> np.ndarray:
        bg = np.median(aligned_stack.astype(np.float32), axis=0)
        bg = np.clip(bg / 255.0, 0.0, 1.0).astype(np.float32)
        return bg

    def _build_aligned_stack(self, center_index: int):
        indices = self._window_indices(center_index)
        frames = [self.get_frame(i) for i in indices]
        center_local_index = len(indices) // 2
        ref_h, ref_w = frames[center_local_index].shape[:2]
        normalized_frames = []
        for frame in frames:
            if frame.shape[0] != ref_h or frame.shape[1] != ref_w:
                frame = np.array(
                    Image.fromarray(frame).resize((ref_w, ref_h), resample=Image.BILINEAR)
                )
            normalized_frames.append(frame)

        aligned_frames = self._align_frames_to_center(normalized_frames, center_local_index)
        aligned_stack = np.stack(aligned_frames, axis=0)
        return aligned_stack, center_local_index

    def _get_prior_detail_bundle(self, center_index: int):
        if self.cache_priors and center_index in self._prior_detail_cache:
            return self._prior_detail_cache[center_index]

        aligned_stack, center_local_index = self._build_aligned_stack(center_index)
        prior, artifact, confidence = self._build_prior_from_stack(aligned_stack, center_local_index)
        bg_rgb = self._build_background_from_stack(aligned_stack)

        boundary_reliability = self._compute_boundary_reliability(center_index)
        if boundary_reliability < 0.999:
            center_rgb = np.clip(aligned_stack[center_local_index] / 255.0, 0.0, 1.0).astype(np.float32)
            prior = prior * boundary_reliability + center_rgb * (1.0 - boundary_reliability)
            artifact = np.clip(artifact * boundary_reliability, 0.0, 1.0)
            confidence = np.clip(confidence * (0.5 + 0.5 * boundary_reliability), 0.0, 1.0)

        bundle = (prior.astype(np.float32), bg_rgb.astype(np.float32), artifact.astype(np.float32), confidence.astype(np.float32))

        if self.cache_priors:
            self._prior_detail_cache[center_index] = bundle
            self._prior_cache[center_index] = bundle[0]
        return bundle

    def get_prior_with_background(self, center_index: int):
        if self.cache_priors and center_index in self._prior_bg_cache:
            return self._prior_bg_cache[center_index]

        prior, bg_rgb, _, _ = self._get_prior_detail_bundle(center_index)
        if self.cache_priors:
            self._prior_bg_cache[center_index] = (prior, bg_rgb)
        return prior, bg_rgb

    def get_prior(self, center_index: int) -> np.ndarray:
        if self.cache_priors and center_index in self._prior_cache:
            return self._prior_cache[center_index]
        prior, _ = self.get_prior_with_background(center_index)
        return prior

    def get_prior_aux_maps(self, center_index: int):
        _, bg_rgb, artifact, confidence = self._get_prior_detail_bundle(center_index)
        return artifact, confidence, bg_rgb

    def _compute_edge_strength(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = image_rgb.mean(axis=2).astype(np.float32)
        if cv2 is not None:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        else:
            gx = np.zeros_like(gray, dtype=np.float32)
            gy = np.zeros_like(gray, dtype=np.float32)
            gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
            gy[1:, :] = gray[1:, :] - gray[:-1, :]

        edge = np.sqrt(gx * gx + gy * gy)
        scale = np.percentile(edge, 95) + 1e-6
        edge = np.clip(edge / scale, 0.0, 1.0)
        return edge.astype(np.float32)

    def get_filtered_frame(
        self,
        center_index: int,
        alpha: float = 0.8,
        threshold: float = 0.35,
        mask_cap: float = 0.65,
        detail_protect: float = 0.75,
        artifact_relax: float = 0.8,
        residual_boost: float = 0.55,
    ) -> np.ndarray:
        """
        Prior-guided deterministic pre-filtering for inference-only usage.
        It blends current frame with temporal background in regions marked as artifacts.
        """
        prior, bg_rgb, artifact, confidence = self._get_prior_detail_bundle(center_index)
        center = self.get_frame(center_index).astype(np.float32) / 255.0

        if self.temporal_semantics == "stable":
            artifact = np.clip(np.power(artifact * confidence, 0.7), 0.0, 1.0)
        else:
            artifact = np.clip(0.65 * artifact + 0.35 * (1.0 - confidence), 0.0, 1.0)

        alpha = float(np.clip(alpha, 0.0, 1.0))
        threshold = float(np.clip(threshold, 0.0, 0.95))
        mask_cap = float(np.clip(mask_cap, 0.0, 1.0))
        detail_protect = float(np.clip(detail_protect, 0.0, 1.0))
        artifact_relax = float(np.clip(artifact_relax, 0.0, 2.0))
        residual_boost = float(np.clip(residual_boost, 0.0, 2.0))
        denom = max(1.0 - threshold, 1e-6)
        mask = np.clip((artifact - threshold) / denom, 0.0, 1.0)

        # Estimate actual difference-to-background and use it to strengthen artifact areas.
        residual = np.abs(center - bg_rgb).mean(axis=2)
        residual_scale = np.percentile(residual, 95) + 1e-6
        residual = np.clip(residual / residual_scale, 0.0, 1.0)

        # Preserve details in low-artifact areas, but relax protection where artifact is strong.
        if detail_protect > 0:
            edge = self._compute_edge_strength(center)
            edge_protect = detail_protect * edge * np.power(np.clip(1.0 - artifact, 0.0, 1.0), artifact_relax)
            mask = mask * np.clip(1.0 - edge_protect, 0.0, 1.0)

        if residual_boost > 0:
            mask = mask * (1.0 + residual_boost * residual)

        dynamic_cap = np.clip(mask_cap + residual_boost * (1.0 - mask_cap) * residual, 0.0, 1.0)
        mask = np.minimum(mask * alpha, dynamic_cap)[:, :, None]

        filtered = center * (1.0 - mask) + prior * mask
        filtered = np.clip(filtered * 255.0, 0.0, 255.0).astype(np.uint8)
        return filtered
