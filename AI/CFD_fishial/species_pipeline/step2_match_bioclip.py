"""
Step 2: BioCLIP2 Taxonomy 매칭
    - 입력: outputs/01_filtered_species.json
    - 참조: BioCLIP TreeOfLife-200M taxonomy (Hugging Face)
    - 출력: outputs/02_bioclip_match.json
    - 구조: 필터링된 종 중 BioCLIP이 인식하는 종/누락 종 분리
"""
import json
import os
import urllib.request

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "outputs", "01_filtered_species.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "02_bioclip_match.json")

BIOCLIP_TAXONOMY_URL = (
    "https://huggingface.co/datasets/imageomics/TreeOfLife-200M"
    "/resolve/main/embeddings/txt_emb_species.json"
)


def fetch_bioclip_names() -> set:
    """BioCLIP TreeOfLife taxonomy에서 '속명 종명' 형태의 학명 세트를 반환한다."""
    print("  BioCLIP Taxonomy 다운로드 중 (Hugging Face)...")
    req = urllib.request.Request(BIOCLIP_TAXONOMY_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        raw = json.loads(response.read().decode("utf-8"))

    names = set()
    for entry in raw:
        # entry[0]: 분류 계층 리스트, 마지막 두 원소가 Genus + Species
        taxon = [t for t in entry[0] if t.strip()]
        if len(taxon) >= 2:
            names.add(f"{taxon[-2]} {taxon[-1]}".lower())
        elif len(taxon) == 1:
            names.add(taxon[0].lower())

    print(f"  BioCLIP 고유 학명 {len(names):,}종 로드 완료")
    return names


def main():
    print(f"[Step 2] 필터링 결과 로드: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    species_list = [item["학명"] for item in data.get("items", []) if item.get("학명")]
    print(f"  종 수: {len(species_list):,}")

    bioclip_names = fetch_bioclip_names()

    matched, missing = [], []
    for sp in species_list:
        (matched if sp.lower().strip() in bioclip_names else missing).append(sp)

    coverage = round(len(matched) / len(species_list) * 100, 2) if species_list else 0
    print(f"  매칭: {len(matched)}종 / 누락: {len(missing)}종 / 커버리지: {coverage}%")

    result = {
        "total_local":      len(species_list),
        "matched_count":    len(matched),
        "missing_count":    len(missing),
        "coverage_percent": coverage,
        "matched_species":  matched,
        "missing_species":  missing,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"  저장 완료: {OUTPUT_FILE}")

    # Sanity check
    assert coverage > 50.0, f"커버리지가 너무 낮습니다: {coverage}%"
    print("  [OK] Sanity check passed.")


if __name__ == "__main__":
    main()
