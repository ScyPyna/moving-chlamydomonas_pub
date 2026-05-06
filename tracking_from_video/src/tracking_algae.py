from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime

from tracking_core import (
    load_config_json,
    load_video_to_ram,
    subtract_background,
    run_locate_to_h5,
    run_link_in_h5,
    load_linked_df,
    save_outputs,
)

log = logging.getLogger("tracking_algae") ##


def process_one(video_path: Path, out_dir: Path, cfg_path: Path, overwrite: bool) -> None:
    cfg = load_config_json(cfg_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / video_path.stem
    h5_path = out_base.with_suffix(".h5")

    if h5_path.exists() and not overwrite:
        log.info("Skip locate (exists): %s", h5_path.name)
        # Per sicurezza carichiamo nFrames comunque per memory default
        frames, fps, duration = load_video_to_ram(video_path, cfg.channel)
        nFrames = frames.shape[0]
        del frames
    else:
        if h5_path.exists() and overwrite:
            log.info("Removing stale H5 (overwrite): %s", h5_path.name)
            h5_path.unlink()
        log.info("Load video -> RAM: %s", video_path.name)
        frames, fps, duration = load_video_to_ram(video_path, cfg.channel)
        nFrames = frames.shape[0]

        if cfg.background_subtraction:
            log.info("Background subtraction: %s", cfg.background_method)
            frames = subtract_background(frames, cfg.background_method)

        log.info("Locate -> %s", h5_path.name)
        run_locate_to_h5(frames, h5_path, cfg)
        del frames

    log.info("Linking in %s", h5_path.name)
    run_link_in_h5(h5_path, cfg, nFrames=nFrames)

    log.info("Export pkl/txt: %s", out_base.name)
    df = load_linked_df(h5_path)
    save_outputs(df, out_base, filter_stubs=cfg.filter_stubs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch tracking for microalgae videos (trackpy).")
    parser.add_argument("--input", type=Path, required=True, help="Input folder with videos.")
    parser.add_argument("--glob", type=str, default="tAlgae*.avi", help="Glob pattern for videos (ignored if --files is given).")
    parser.add_argument("--files", type=str, nargs="+", default=None, help="Specific video filenames to process (e.g. tAlgae68.avi tAlgae69.avi).")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_tracking.json.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder. Default: <input>/results_<timestamp>")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_dir = args.input
    out_dir = args.out
    if out_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        out_dir = in_dir / f"results_{stamp}"

    if args.files:
        videos = sorted([in_dir / f for f in args.files])
        missing = [v for v in videos if not v.exists()]
        if missing:
            raise SystemExit(f"Video non trovati: {[str(m) for m in missing]}")
    else:
        videos = sorted(in_dir.glob(args.glob))

    if not videos:
        raise SystemExit(f"No videos found in {in_dir} with pattern {args.glob}")

    log.info("Found %d videos", len(videos))
    log.info("Output dir: %s", out_dir)

    for vp in videos:
        try:
            process_one(vp, out_dir, args.config, overwrite=args.overwrite)
        except Exception as e:
            log.exception("Failed on %s: %s", vp.name, e)


if __name__ == "__main__":
    main()