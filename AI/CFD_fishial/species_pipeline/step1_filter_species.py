"""
Step 1: 다이버 관점 종 필터링
    - 입력: data/result_bioinfo.json (MBRIS 전체 종 목록)
    - 출력: outputs/01_filtered_species.json
    - 기준: 어류, 문어류, 오징어류, 해파리류, 파충류, 포유류
"""
import json
import os

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data",    "result_bioinfo.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "01_filtered_species.json")

# ── 필터링 조건 정의 ────────────────────────────────────────────────────────────
def is_target_species(item: dict) -> bool:
    """다이버 가시성 기준으로 포함 여부를 판단한다."""
    sub_group   = item.get("세부분류군명", "").strip()
    order_val   = item.get("Order", "").strip()
    phylum_val  = item.get("Phylum", "").strip()
    korean_name = str(item.get("국명", ""))

    return (
        sub_group == "어류"                                             # 어류 전체
        or "Octopoda" in order_val                                      # 문어류
        or ("오징어" in korean_name and phylum_val in ["Mollusca", "Mullusca"])  # 오징어류
        or (sub_group == "자포동물" and "해파리" in korean_name)        # 해파리류
        or sub_group == "파충류"                                        # 파충류
        or sub_group == "포유류"                                        # 포유류
    )


def main():
    print(f"[Step 1] 원본 데이터 로드: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"  총 {len(items):,}건 확인")

    filtered = [item for item in items if is_target_species(item)]
    print(f"  필터링 완료 → {len(filtered):,}건")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_data = {"total_filtered_count": len(filtered), "items": filtered}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"  저장 완료: {OUTPUT_FILE}")

    # Sanity check
    assert len(filtered) > 0, "필터링 결과가 비어 있습니다."
    print("  [OK] Sanity check passed.")


if __name__ == "__main__":
    main()
