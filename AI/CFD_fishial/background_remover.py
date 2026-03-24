"""
Divery Vision Pipeline - Background Remover Module
===================================================
물고기 이미지에서 배경을 제거하고 PNG로 저장합니다.
rembg 라이브러리를 사용하여 자동 배경 제거를 수행합니다.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

# rembg가 설치되어 있지 않거나 onnxruntime이 없으면 graceful fallback
REMBG_AVAILABLE = False
try:
    # onnxruntime 체크 먼저
    import onnxruntime
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError as e:
    print(f"[BackgroundRemover] WARNING: rembg/onnxruntime not available ({e}). Background removal disabled.")
    print("  Install with: pip install rembg[gpu] onnxruntime-gpu")
except Exception as e:
    print(f"[BackgroundRemover] WARNING: rembg init failed ({e}). Background removal disabled.")


class BackgroundRemover:
    """
    물고기 이미지 배경 제거 클래스.

    rembg 라이브러리를 사용하여 배경을 투명하게 만들고 PNG로 저장합니다.
    GPU가 사용 가능하면 GPU 가속을 활용합니다.
    """

    def __init__(
        self,
        model_name: str = "u2net",
        output_suffix: str = "_nobg",
        keep_original: bool = True,
    ) -> None:
        """
        Args:
            model_name: rembg 모델 이름 (u2net, u2netp, u2net_human_seg 등)
            output_suffix: 출력 파일명에 추가할 접미사
            keep_original: True면 원본 유지, False면 원본 삭제
        """
        self.model_name = model_name
        self.output_suffix = output_suffix
        self.keep_original = keep_original
        self._session = None

        if REMBG_AVAILABLE:
            print(f"[BackgroundRemover] Initialized with model: {model_name}")
        else:
            print("[BackgroundRemover] Disabled (rembg not available)")

    def _ensure_session(self):
        """rembg 세션을 lazy initialize."""
        if not REMBG_AVAILABLE:
            return None
        if self._session is None:
            self._session = new_session(self.model_name)
        return self._session

    def remove_background(self, image_path: str) -> Optional[str]:
        """
        단일 이미지의 배경을 제거합니다.

        Args:
            image_path: 입력 이미지 경로 (JPG, PNG 등)

        Returns:
            배경 제거된 PNG 이미지 경로, 실패 시 None
        """
        if not REMBG_AVAILABLE:
            return None

        input_path = Path(image_path)
        if not input_path.exists():
            print(f"[BackgroundRemover] File not found: {image_path}")
            return None

        # 출력 경로 생성 (PNG로 변환)
        output_filename = f"{input_path.stem}{self.output_suffix}.png"
        output_path = input_path.parent / output_filename

        try:
            # 이미지 읽기
            with open(input_path, "rb") as f:
                input_data = f.read()

            # 배경 제거
            session = self._ensure_session()
            output_data = remove(input_data, session=session)

            # PNG로 저장
            with open(output_path, "wb") as f:
                f.write(output_data)

            # 원본 삭제 옵션
            if not self.keep_original:
                input_path.unlink()

            return str(output_path)

        except Exception as e:
            print(f"[BackgroundRemover] Error processing {image_path}: {e}")
            return None

    def remove_background_cv2(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        OpenCV 이미지(numpy array)에서 배경을 제거합니다.

        Args:
            image: BGR 형식의 OpenCV 이미지

        Returns:
            BGRA 형식의 배경 제거된 이미지, 실패 시 None
        """
        if not REMBG_AVAILABLE:
            return None

        try:
            # BGR -> RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 배경 제거 (PIL Image 반환)
            session = self._ensure_session()
            result = remove(rgb_image, session=session)

            # numpy array로 변환
            result_np = np.array(result)

            # RGBA -> BGRA
            if result_np.shape[2] == 4:
                bgra = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGRA)
                return bgra
            else:
                return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"[BackgroundRemover] Error: {e}")
            return None

    def process_directory(
        self,
        directory: str,
        extensions: List[str] = None,
    ) -> Dict[str, str]:
        """
        디렉토리 내 모든 이미지의 배경을 제거합니다.

        Args:
            directory: 이미지가 있는 디렉토리 경로
            extensions: 처리할 확장자 목록 (기본: jpg, jpeg, png)

        Returns:
            {원본경로: 결과경로} 딕셔너리
        """
        if not REMBG_AVAILABLE:
            return {}

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]

        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"[BackgroundRemover] Directory not found: {directory}")
            return {}

        results = {}
        image_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
            and self.output_suffix not in f.stem  # 이미 처리된 파일 제외
        ]

        print(f"[BackgroundRemover] Processing {len(image_files)} images in {directory}")

        for img_file in image_files:
            output_path = self.remove_background(str(img_file))
            if output_path:
                results[str(img_file)] = output_path

        print(f"[BackgroundRemover] Completed: {len(results)} images processed")
        return results

    def process_best_shots(
        self,
        best_shots: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Best Shot 딕셔너리의 이미지들에서 배경을 제거합니다.

        Args:
            best_shots: {track_id: {"path": str, "conf": float, ...}} 딕셔너리

        Returns:
            배경 제거 경로가 추가된 딕셔너리
        """
        if not REMBG_AVAILABLE:
            return best_shots

        print(f"[BackgroundRemover] Processing {len(best_shots)} best shots")

        for tid, data in best_shots.items():
            if "path" not in data:
                continue

            nobg_path = self.remove_background(data["path"])
            if nobg_path:
                data["nobg_path"] = nobg_path

        processed = sum(1 for d in best_shots.values() if "nobg_path" in d)
        print(f"[BackgroundRemover] Processed: {processed}/{len(best_shots)} images")
        return best_shots

    def cleanup(self) -> None:
        """세션 정리."""
        self._session = None


def remove_background_from_crop(
    crop: np.ndarray,
    model_name: str = "u2net",
) -> Optional[np.ndarray]:
    """
    단일 크롭 이미지에서 배경을 제거하는 헬퍼 함수.

    Args:
        crop: BGR 형식의 OpenCV 이미지
        model_name: rembg 모델 이름

    Returns:
        BGRA 형식의 배경 제거된 이미지
    """
    remover = BackgroundRemover(model_name=model_name)
    return remover.remove_background_cv2(crop)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python background_remover.py <image_path_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    remover = BackgroundRemover()

    if target.is_file():
        result = remover.remove_background(str(target))
        if result:
            print(f"Output: {result}")
        else:
            print("Failed to process image")
    elif target.is_dir():
        results = remover.process_directory(str(target))
        print(f"Processed {len(results)} images")
    else:
        print(f"Path not found: {target}")
