"""
Fish Sketch Generator Package

어종 스케치 이미지를 AI로 생성하는 패키지
"""

from .generator import create_generator, SketchGenerator
from .prompts import build_sketch_prompt, build_negative_prompt

__all__ = [
    "create_generator",
    "SketchGenerator",
    "build_sketch_prompt",
    "build_negative_prompt",
]
