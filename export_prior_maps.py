import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from utils.video_prior_utils import TemporalPriorExtractor


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _has_image_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in VALID_EXTS:
            return True
    return False


def _resolve_input_dir(input_path: str) -> Path:
    base = Path(input_path)
    candidate = base / "input"
    if candidate.is_dir() and _has_image_files(candidate):
        return candidate
    if _has_image_files(base):
        return base
    raise FileNotFoundError(f"No image sequence found under: {input_path}")


def _default_output_dir(raw_input_path: str, resolved_input_dir: Path) -> Path:
    raw_path = Path(raw_input_path)
    if (raw_path / "input").is_dir():
        return raw_path / "prior_maps"
    return resolved_input_dir / "prior_maps"


def _collect_frames(input_dir: Path):
    frame_paths = [
        str(path)
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    ]
    if not frame_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")
    return frame_paths


def _save_gray(array_01: np.ndarray, output_path: Path):
    array_u8 = np.clip(array_01 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array_u8, mode="L").save(output_path)


def _save_rgb(array_01: np.ndarray, output_path: Path):
    array_u8 = np.clip(array_01 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array_u8, mode="RGB").save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export 3-channel temporal prior maps for a video-frame sequence."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Frame-sequence directory, or a parent directory containing an input/ subfolder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <seq>/prior_maps",
    )
    parser.add_argument("--prior_window_size", type=int, default=9)
    parser.add_argument(
        "--prior_align_mode",
        type=str,
        default="ecc",
        choices=["none", "ecc"],
    )
    parser.add_argument(
        "--prior_warp_mode",
        type=str,
        default="affine",
        choices=["translation", "euclidean", "affine", "homography"],
    )
    parser.add_argument(
        "--prior_temporal_semantics",
        type=str,
        default="stable",
        choices=["transient", "stable"],
    )
    parser.add_argument("--prior_stable_band_ratio", type=float, default=0.10)
    parser.add_argument(
        "--save_aux_maps",
        action="store_true",
        help="Also save artifact/confidence maps used by deterministic prior filtering.",
    )
    parser.add_argument(
        "--save_frame_copy",
        action="store_true",
        help="Also save the cropped input frame used by the prior extractor.",
    )
    args = parser.parse_args()

    input_dir = _resolve_input_dir(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.input_path, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = _collect_frames(input_dir)
    extractor = TemporalPriorExtractor(
        frame_paths=frame_paths,
        window_size=args.prior_window_size,
        align_mode=args.prior_align_mode,
        warp_mode=args.prior_warp_mode,
        temporal_semantics=args.prior_temporal_semantics,
        stable_band_ratio=args.prior_stable_band_ratio,
        crop_base=16,
    )

    for idx, frame_path in enumerate(frame_paths):
        stem = Path(frame_path).stem
        prior, bg_rgb = extractor.get_prior_with_background(idx)
        artifact, confidence, _ = extractor.get_prior_aux_maps(idx)

        _save_rgb(prior, output_dir / f"{stem}_prior_rgb.png")
        _save_gray(prior[:, :, 0], output_dir / f"{stem}_r.png")
        _save_gray(prior[:, :, 1], output_dir / f"{stem}_g.png")
        _save_gray(prior[:, :, 2], output_dir / f"{stem}_b.png")
        _save_rgb(bg_rgb, output_dir / f"{stem}_bg_rgb.png")
        if args.save_aux_maps:
            _save_gray(artifact, output_dir / f"{stem}_artifact.png")
            _save_gray(confidence, output_dir / f"{stem}_confidence.png")

        if args.save_frame_copy:
            frame_rgb = extractor.get_frame(idx).astype(np.float32) / 255.0
            _save_rgb(frame_rgb, output_dir / f"{stem}_frame.png")

    print(f"Exported {len(frame_paths)} prior maps to: {output_dir}")
    print("Channel definition:")
    print("  prior_rgb: color background guidance prior")
    print("  r/g/b: per-channel view of the color prior")
    if args.save_aux_maps:
        print("  artifact/confidence: auxiliary maps for filtering/debug")


if __name__ == "__main__":
    main()
