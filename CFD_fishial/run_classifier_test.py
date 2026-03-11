"""
Divery Vision Pipeline - Module B: BioCLIP-2 Classifier 실행
=============================================================
사용법:
    conda activate diveary-vision
    cd CFD_fishial

    # best_shots 폴더의 이미지 분류
    python run_classifier_test.py

    # 특정 폴더 지정
    python run_classifier_test.py --image-dir ./results/best_shots
"""

import argparse
from pathlib import Path

from classifier_tester import ClassifierTester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divery Module B - BioCLIP-2 Classifier Test",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Config file path (default: config.yaml)",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Image directory to classify. Default: results/best_shots from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("[Divery] Module B: BioCLIP-2 Classifier Test")
    print("=" * 60)

    tester = ClassifierTester(config_path=args.config)

    try:
        image_dir = args.image_dir
        if image_dir is None:
            image_dir = tester.config["tracker"].get(
                "best_shots_dir", "./results/best_shots"
            )

        print(f"\n[INFO] Image directory: {Path(image_dir).resolve()}")

        results = tester.run(image_dir)

        if results:
            print(f"\n[DONE] {len(results)} images classified successfully.")
        else:
            print("\n[DONE] No images were classified.")

    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
