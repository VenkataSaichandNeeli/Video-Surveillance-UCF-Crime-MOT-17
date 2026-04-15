#!/usr/bin/env python3
"""
Surveillance Pipeline — CLI entry point.

Examples
--------
# Process a video file
python run.py --video input.mp4 --zones config/zones.json --output results/

# Process a MOT17 sequence (folder containing seqinfo.ini)
python run.py --video data/MOT17-05-FRCNN/ --zones config/zones.json --output results/ --mode sequence

# Process a static image folder (tracking disabled, annotation only)
python run.py --video data/images/ --zones config/zones.json --output results/ --mode images

# Run the interactive zone editor before processing
python run.py --video input.mp4 --zones config/zones.json --output results/ --edit-zones

# Run with MOT17 evaluation against ground truth
python run.py --video data/MOT17-05-FRCNN/ --zones config/zones.json --output results/ --evaluate

# Force CPU (useful when GPU VRAM is limited)
python run.py --video input.mp4 --zones config/zones.json --output results/ --device cpu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging must be configured before any src imports so all loggers
# inherit the root handler.
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3", "transformers.modeling_utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Video Surveillance: Detection, Tracking & Event Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required ── #
    parser.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to a video file, MOT17 sequence directory, or image folder.",
    )
    parser.add_argument(
        "--zones",
        required=True,
        metavar="ZONES_JSON",
        help="Path to zones.json defining polygon zones of interest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="OUTPUT_DIR",
        help="Directory where all outputs will be written.",
    )

    # ── Optional ── #
    parser.add_argument(
        "--config",
        default="config/default_config.json",
        metavar="CONFIG_JSON",
        help="Path to config.json (default: config/default_config.json).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "video", "sequence", "images"],
        default="auto",
        help=(
            "Input mode. 'auto' infers from filesystem layout. "
            "'video' = single video file. "
            "'sequence' = MOT17-style image sequence with seqinfo.ini. "
            "'images' = flat folder of images (tracking disabled)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. 'auto' selects CUDA if available (default: auto).",
    )
    parser.add_argument(
        "--edit-zones",
        action="store_true",
        help="Launch the interactive zone editor before running the pipeline.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Compute MOTA/MOTP metrics against ground truth. "
            "Requires a gt/gt.txt file inside the input sequence directory."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        metavar="N",
        help="Process every Nth frame (overrides config value).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Show a live annotated preview window while processing. "
            "Press Q or Esc inside the window to stop early."
        ),
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    logger = logging.getLogger("run")

    # ── Validate paths ── #
    input_path = Path(args.video).resolve()
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    zones_path = Path(args.zones).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Import after logging is configured ── #
    from src.config import PipelineCfg
    from src.pipeline import SurveillancePipeline
    from src.zone_editor import ZoneEditor

    # ── Build config ── #
    cfg = PipelineCfg.load(
        config_path=args.config,
        zones_path=str(zones_path),
        device_override=args.device,
    )

    # CLI overrides
    if args.sample_rate is not None:
        if args.sample_rate < 1:
            logger.error("--sample-rate must be >= 1")
            return 1
        cfg.frame_sample_rate = args.sample_rate
        logger.info("Frame sample rate overridden to %d", cfg.frame_sample_rate)

    # Ensure face DB directory exists
    Path(cfg.face_db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Optional zone editor ── #
    if args.edit_zones:
        logger.info("Launching interactive zone editor …")
        editor = ZoneEditor(
            input_path=input_path,
            zones_path=zones_path,
            cfg=cfg,
        )
        editor.run()
        # Re-resolve zones path in case the editor wrote a new file
        cfg.zones_path = str(zones_path)

    # ── Verify zones file exists ── #
    if not zones_path.exists():
        logger.warning(
            "zones.json not found at %s — pipeline will run without zone logic. "
            "Use --edit-zones to create zones interactively.",
            zones_path,
        )

    # ── Run pipeline ── #
    logger.info("=" * 60)
    logger.info("Input  : %s", input_path)
    logger.info("Zones  : %s", zones_path)
    logger.info("Output : %s", output_dir)
    logger.info("Mode   : %s", args.mode)
    logger.info("Device : %s", cfg.device)
    logger.info("=" * 60)

    pipeline = SurveillancePipeline(
        input_path=input_path,
        output_dir=output_dir,
        config=cfg,
        mode=args.mode,
        evaluate=args.evaluate,
        show_preview=args.preview,
    )

    exit_code = 0
    try:
        t0 = time.perf_counter()
        pipeline.run()
        elapsed = time.perf_counter() - t0
        logger.info("Pipeline finished in %.1f seconds.", elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user — flushing outputs …")
        exit_code = 130  # standard SIGINT exit code

    except Exception as exc:
        logger.exception("Pipeline crashed: %s", exc)
        exit_code = 1

    finally:
        pipeline.cleanup()

    _print_summary(output_dir, logger)
    return exit_code


def _print_summary(output_dir: Path, logger: logging.Logger) -> None:
    """Log the paths of all generated output files."""
    logger.info("─" * 60)
    logger.info("Output files:")
    for pattern in ("output.mp4", "events.csv", "mot_evaluation.json"):
        p = output_dir / pattern
        if p.exists():
            size_mb = p.stat().st_size / 1_048_576
            logger.info("  %-35s  %.1f MB", p.name, size_mb)
    frames_dir = output_dir / "sample_frames"
    if frames_dir.exists():
        count = sum(1 for _ in frames_dir.iterdir())
        logger.info("  sample_frames/  (%d event snapshots)", count)
    logger.info("─" * 60)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
