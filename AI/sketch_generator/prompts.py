"""
스케치 이미지 생성 프롬프트 템플릿
"""

SKETCH_PROMPT_TEMPLATE = """cute kawaii cartoon sketch of {name} ({scientific_name}),
simple clean line drawing, chibi style, adorable expression,
white background, no text, suitable for diving logbook sticker,
high quality, minimalist design, marine life illustration"""

SKETCH_NEGATIVE_PROMPT = """realistic, photo, photograph, text, watermark, signature,
blurry, low quality, deformed, ugly, scary, horror"""

def build_sketch_prompt(name: str, scientific_name: str) -> str:
    """어종 이름으로 스케치 생성 프롬프트 생성"""
    return SKETCH_PROMPT_TEMPLATE.format(
        name=name.strip(),
        scientific_name=scientific_name.strip() if scientific_name else "marine fish"
    )

def build_negative_prompt() -> str:
    """네거티브 프롬프트 반환"""
    return SKETCH_NEGATIVE_PROMPT
