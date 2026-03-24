"""
Step 3: 누락 종 학명 동의어 정규화 (GBIF + WoRMS)
    - 입력: outputs/02_bioclip_match.json
    - 참조: GBIF Backbone Taxonomy API, WoRMS REST API
    - 출력: outputs/03_synonym_mapping.json
    - 목적: BioCLIP에서 텍스트 매칭에 실패한 종을 동의어(Synonym)로 복구

  흐름:
    1차) GBIF API로 각 미매칭 학명의 accepted name 후보 확보 (10 threads)
    2차) WoRMS AphiaRecordsByName + AphiaSynonymsByAphiaID로 추가 동의어 확보 (8 threads)
    양쪽에서 얻은 후보 이름들을 BioCLIP taxonomy와 비교 → 매칭 시 복구
"""
import json
import os
import urllib.request
import urllib.parse
import concurrent.futures

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "outputs", "02_bioclip_match.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "03_synonym_mapping.json")

BIOCLIP_TAXONOMY_URL = (
    "https://huggingface.co/datasets/imageomics/TreeOfLife-200M"
    "/resolve/main/embeddings/txt_emb_species.json"
)


# ── BioCLIP Taxonomy 로드 ──────────────────────────────────────────────────────
def fetch_bioclip_names() -> set:
    req = urllib.request.Request(BIOCLIP_TAXONOMY_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    names = set()
    for entry in raw:
        taxon = [t for t in entry[0] if t.strip()]
        if len(taxon) >= 2:
            names.add(f"{taxon[-2]} {taxon[-1]}".lower())
        elif len(taxon) == 1:
            names.add(taxon[0].lower())
    return names


# ── GBIF API ──────────────────────────────────────────────────────────────────
def get_gbif_candidates(name: str) -> set:
    """GBIF에서 학명의 accepted name 후보를 반환."""
    url = f"https://api.gbif.org/v1/species/match?name={urllib.parse.quote(name)}&class=Actinopterygii"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidates = set()
        if data.get("matchType") != "NONE":
            if "species" in data:
                candidates.add(data["species"].lower())
            if "accepted" in data:
                parts = data["accepted"].split()
                if len(parts) >= 2:
                    candidates.add(f"{parts[0]} {parts[1]}".lower())
        return candidates
    except Exception:
        return set()


# ── WoRMS API ─────────────────────────────────────────────────────────────────
def get_worms_candidates(name: str) -> set:
    """WoRMS에서 학명의 accepted name 및 synonym 목록을 반환."""
    candidates = set()

    for marine_only in ["true", "false"]:
        url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{urllib.parse.quote(name)}?like=false&marine_only={marine_only}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                records = json.loads(resp.read().decode("utf-8"))
            if records and isinstance(records, list):
                # valid_name 수집
                for record in records:
                    vn = record.get("valid_name", "")
                    if vn:
                        parts = vn.split()
                        if len(parts) >= 2:
                            candidates.add(f"{parts[0]} {parts[1]}".lower())

                # 동의어 목록 수집 (첫 번째 레코드의 AphiaID 사용)
                aphia_id = records[0].get("valid_AphiaID") or records[0].get("AphiaID")
                if aphia_id:
                    syn_url = f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}?offset=1"
                    try:
                        req_syn = urllib.request.Request(syn_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req_syn, timeout=10) as resp_syn:
                            synonyms = json.loads(resp_syn.read().decode("utf-8"))
                        if synonyms and isinstance(synonyms, list):
                            for s in synonyms:
                                sn = s.get("scientificname", "")
                                parts = sn.split()
                                if len(parts) >= 2:
                                    candidates.add(f"{parts[0]} {parts[1]}".lower())
                    except Exception:
                        pass
                break  # marine_only=true 에서 결과가 나왔으면 종료
        except Exception:
            continue

    return candidates


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[Step 3] BioCLIP 미매칭 종 로드: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
    missing = comp_data.get("missing_species", [])
    print(f"  미매칭 종 수: {len(missing)}")

    print("  BioCLIP Taxonomy 다운로드 중...")
    bioclip_names = fetch_bioclip_names()
    print(f"  BioCLIP 학명 {len(bioclip_names):,}종 로드 완료")

    # ── 1차: GBIF ──
    print(f"\n  [1차] GBIF API 병렬 조회 (10 threads)...")
    gbif_results: dict[str, set] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(get_gbif_candidates, sp): sp for sp in missing}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            sp = futures[f]
            gbif_results[sp] = f.result()
            if (i + 1) % 50 == 0:
                print(f"    [{i + 1}/{len(missing)}] 완료")
    print("  GBIF 조회 완료")

    recovered: dict[str, str] = {}
    still_missing: list[str] = []

    for sp in missing:
        matched = next((c for c in gbif_results.get(sp, set()) if c in bioclip_names), None)
        if matched:
            recovered[sp] = matched
        else:
            still_missing.append(sp)
    print(f"  GBIF 복구: {len(recovered)}종 / 남은 미매칭: {len(still_missing)}종")

    # ── 2차: WoRMS ──
    print(f"\n  [2차] WoRMS API 병렬 조회 (8 threads)...")
    worms_results: dict[str, set] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_worms_candidates, sp): sp for sp in still_missing}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            sp = futures[f]
            worms_results[sp] = f.result()
            if (i + 1) % 20 == 0:
                print(f"    [{i + 1}/{len(still_missing)}] 완료")
    print("  WoRMS 조회 완료")

    final_missing: list[str] = []
    for sp in still_missing:
        matched = next((c for c in worms_results.get(sp, set()) if c in bioclip_names), None)
        if matched:
            recovered[sp] = matched
        else:
            final_missing.append(sp)

    print(f"\n  최종 복구 합계: {len(recovered)}종 / 최종 미매칭: {len(final_missing)}종")

    result = {
        "recovered_count":    len(recovered),
        "still_missing_count": len(final_missing),
        "recovered_mapping":   recovered,
        "still_missing":       final_missing,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"  저장 완료: {OUTPUT_FILE}")

    # Sanity check
    assert len(recovered) > 0, "동의어로 복구된 종이 없습니다."
    print("  [OK] Sanity check passed.")


if __name__ == "__main__":
    main()
