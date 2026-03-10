"""
Divery Vision Pipeline - Module A-2: Best-Shot 추출 실행
=======================================================
사용법:
    conda activate diveary-vision
    cd CFD_fishial
    python run_best_shot.py --video test_videos/dive_clip.mp4
"""

import argparse
import sys
from pathlib import Path

from best_shot_extractor import BestShotExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divery Module A-2 - ByteTrack Best-Shot Extractor",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Single video file path. If not provided, scans video_dir from config.",
    )
    return parser.parse_args()


def collect_videos(video_dir: str, extensions: list[str]) -> list[Path]:
    vdir = Path(video_dir)
    if not vdir.exists():
        print(f"[ERROR] Video directory not found: {vdir.resolve()}")
        sys.exit(1)
    files = []
    for ext in extensions:
        files.extend(vdir.glob(f"*{ext}"))
        files.extend(vdir.glob(f"*{ext.upper()}"))
    files = sorted(set(files))
    if not files:
        print(f"[ERROR] No video files in '{vdir}'. Extensions: {extensions}")
        sys.exit(1)
    return files


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("[Divery] Module A-2: ByteTrack Best-Shot Extractor")
    print("=" * 60)

    extractor = BestShotExtractor(config_path=args.config)

    try:
        if args.video:
            video_files = [Path(args.video)]
        else:
            cfg_input = extractor.config["input"]
            video_files = collect_videos(
                cfg_input["video_dir"], cfg_input["extensions"]
            )

        print(f"\n[INFO] Videos to process: {len(video_files)}")
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {vf.name}")

        all_results = {}
        for vf in video_files:
            result = extractor.run(str(vf))
            all_results[vf.name] = result

        # Final summary
        total_ids = sum(len(r) for r in all_results.values())
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Total unique tracked IDs across all videos: {total_ids}")
        print(f"  Best shots saved to: {extractor.best_shots_dir.resolve()}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
