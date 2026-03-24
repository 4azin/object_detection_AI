"""
Fish Sketch Generator
Generates kawaii-style fish sketches for the fish_book database entries.
Uses Stability AI or OpenAI DALL-E for image generation.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sketch-generator")


class SketchGenerator:
    """Generate kawaii-style fish sketches using AI image generation APIs."""

    def __init__(
        self,
        api_key: str,
        provider: str = "stability",
        model: str = "stable-diffusion-xl-1024-v1-0",
    ):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model

        if self.provider == "stability":
            self.api_url = "https://api.stability.ai/v1/generation"
        elif self.provider == "openai":
            self.api_url = "https://api.openai.com/v1/images/generations"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate_sketch(
        self,
        name: str,
        scientific_name: Optional[str] = None,
        style: str = "kawaii",
    ) -> bytes:
        """Generate a sketch image for a fish species."""
        prompt = self._build_prompt(name, scientific_name, style)
        log.info(f"Generating sketch for {name} with prompt: {prompt[:100]}...")

        if self.provider == "stability":
            return self._generate_stability(prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(
        self,
        name: str,
        scientific_name: Optional[str],
        style: str,
    ) -> str:
        """Build the image generation prompt."""
        species_part = name
        if scientific_name:
            species_part = f"{name} ({scientific_name})"

        style_prompts = {
            "kawaii": (
                f"cute kawaii cartoon sketch of {species_part}, "
                "simple clean line drawing, chibi style, adorable expression, "
                "white background, no text, suitable for diving logbook sticker"
            ),
            "realistic": (
                f"detailed scientific illustration of {species_part}, "
                "realistic anatomy, naturalistic colors, "
                "white background, no text, field guide style"
            ),
            "watercolor": (
                f"watercolor painting of {species_part}, "
                "soft colors, artistic style, "
                "white background, no text, japanese art influence"
            ),
        }

        return style_prompts.get(style, style_prompts["kawaii"])

    def _generate_stability(self, prompt: str) -> bytes:
        """Generate image using Stability AI."""
        url = f"{self.api_url}/{self.model}/text-to-image"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }

        body = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0},
                {"text": "blurry, bad quality, text, watermark", "weight": -1.0},
            ],
            "cfg_scale": 7,
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        }

        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        return response.content

    def _generate_openai(self, prompt: str) -> bytes:
        """Generate image using OpenAI DALL-E."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": "512x512",
            "response_format": "b64_json",
        }

        response = requests.post(self.api_url, headers=headers, json=body, timeout=60)
        response.raise_for_status()

        result = response.json()
        b64_data = result["data"][0]["b64_json"]
        return base64.b64decode(b64_data)


def create_generator_from_env() -> SketchGenerator:
    """Create a SketchGenerator from environment variables."""
    provider = os.getenv("SKETCH_PROVIDER", "stability")

    if provider == "stability":
        api_key = os.getenv("STABILITY_API_KEY", "")
        model = os.getenv("STABILITY_MODEL", "stable-diffusion-xl-1024-v1-0")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")
    else:
        raise ValueError(f"Unknown SKETCH_PROVIDER: {provider}")

    if not api_key:
        raise RuntimeError(f"API key not set for provider {provider}")

    return SketchGenerator(api_key=api_key, provider=provider, model=model)
