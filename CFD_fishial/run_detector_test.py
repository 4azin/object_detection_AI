"""
Divery Vision Pipeline - Module A: Detector 독립 테스트 실행
==========================================================
사용법:
    conda activate diveary-vision
    cd CFD_fishial
    python run_detector_test.py
    python run_detector_test.py --config my_config.yaml
    python run_detector_test.py --video single_video.mp4
"""

import argparse
import sys
from pathlib import Path

from detector_tester import DetectorTester


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Divery Module A - CFD Detector 독립 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_detector_test.py                        # config.yaml 기본 사용
  python run_detector_test.py --config custom.yaml   # 커스텀 설정 파일
  python run_detector_test.py --video dive_clip.mp4  # 단일 영상 테스트
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="설정 파일 경로 (기본값: config.yaml)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="단일 영상 파일 경로. 지정하면 config의 video_dir 대신 이 파일만 테스트.",
    )
    return parser.parse_args()


def collect_video_files(video_dir: str, extensions: list[str]) -> list[Path]:
    """지정 디렉토리에서 지원 확장자의 영상 파일 목록을 수집한다."""
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        print(f"[오류] 영상 디렉토리가 존재하지 않습니다: {video_dir_path.resolve()}")
        print(f"  → '{video_dir_path}' 폴더를 생성하고 테스트 영상을 넣어주세요.")
        sys.exit(1)

    video_files = []
    for ext in extensions:
        video_files.extend(video_dir_path.glob(f"*{ext}"))
        # 대소문자 혼용 확장자도 수집 (e.g., .MP4, .MOV)
        video_files.extend(video_dir_path.glob(f"*{ext.upper()}"))

    # 중복 제거 및 정렬
    video_files = sorted(set(video_files))

    if not video_files:
        print(f"[오류] '{video_dir_path}' 에 영상 파일이 없습니다.")
        print(f"  → 지원 확장자: {extensions}")
        sys.exit(1)

    return video_files


def main() -> None:
    """메인 실행 함수."""
    args = parse_args()

    print("=" * 60)
    print("[Divery] Vision Pipeline - Module A: Detector 독립 테스트")
    print("=" * 60)

    # DetectorTester 초기화
    tester = DetectorTester(config_path=args.config)

    try:
        if args.video:
            # 단일 영상 테스트
            video_files = [Path(args.video)]
        else:
            # config 기반 디렉토리 스캔
            cfg_input = tester.config["input"]
            video_files = collect_video_files(
                cfg_input["video_dir"], cfg_input["extensions"]
            )

        print(f"\n[INFO] 테스트 대상 영상: {len(video_files)}개")
        for i, vf in enumerate(video_files, 1):
            print(f"  {i}. {vf.name}")

        # 영상별 테스트 실행
        all_logs = {}
        for video_file in video_files:
            log = tester.run(str(video_file))
            all_logs[video_file.name] = log

        # 전체 요약
        print("\n" + "=" * 60)
        print("[SUMMARY] 전체 테스트 요약")
        print("=" * 60)
        total_frames_all = 0
        total_objects_all = 0

        for name, log in all_logs.items():
            n_frames = len(log)
            n_objects = sum(e["Object_Count"] for e in log)
            avg_fps = sum(e["FPS"] for e in log) / n_frames if n_frames > 0 else 0
            total_frames_all += n_frames
            total_objects_all += n_objects
            print(f"  {name}: {n_frames} frames, {n_objects} detections, {avg_fps:.1f} avg FPS")

        print(f"\n  총 처리 프레임: {total_frames_all}")
        print(f"  총 탐지 객체  : {total_objects_all}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n[오류] 테스트 중 예외 발생: {e}")
        raise
    finally:
        # GPU 메모리 해제
        tester.cleanup()


if __name__ == "__main__":
    main()
