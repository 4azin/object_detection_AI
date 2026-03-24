"""
Step 4: OBIS 해역별 관측 횟수 수집
    - 입력:
        outputs/01_filtered_species.json   (다이버 필터링 종 상세 정보)
        outputs/02_bioclip_match.json      (BioCLIP matched_species 목록)
        outputs/03_synonym_mapping.json    (동의어로 복구된 종 매핑)
    - 출력: outputs/04_obis_prior_db.json
    - 한반도 해역 기준 4개 구역 관측 횟수 수집 (전체 / 서해 / 남해 / 동해)

  핵심 로직:
    1. WoRMS API → valid_AphiaID 확보
         (marine_only=true → false → MatchNames 3단계 폴백)
    2. OBIS API → taxonid=AphiaID 로 조회 (가장 정확)
         AphiaID 없는 경우 → scientificname 텍스트 폴백
    3. overall==0인 동의어 복구 종은 accepted name 텍스트로 재조회
         (OBIS가 이명이 아닌 텍스트 기준으로 저장하는 케이스 대응)
    4. overall==0 종 최종 제거
"""
import os
import json
import requests
import urllib.request
import urllib.parse
import concurrent.futures

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
FILTERED_FILE    = os.path.join(BASE_DIR, "outputs", "01_filtered_species.json")
MATCH_FILE       = os.path.join(BASE_DIR, "outputs", "02_bioclip_match.json")
SYNONYM_FILE     = os.path.join(BASE_DIR, "outputs", "03_synonym_mapping.json")
OUTPUT_FILE      = os.path.join(BASE_DIR, "outputs", "04_obis_prior_db.json")

# ── 한반도 해역 WKT (GeoJSON Polygon) ─────────────────────────────────────────
REGIONS = {
    "overall": "POLYGON((124.0 32.0, 132.0 32.0, 132.0 39.0, 124.0 39.0, 124.0 32.0))",
    "west":    "POLYGON((124.0 34.0, 127.0 34.0, 127.0 39.0, 124.0 39.0, 124.0 34.0))",
    "south":   "POLYGON((124.0 32.0, 132.0 32.0, 132.0 34.0, 124.0 34.0, 124.0 32.0))",
    "east":    "POLYGON((127.0 34.0, 132.0 34.0, 132.0 39.0, 127.0 39.0, 127.0 34.0))",
}


# ────────────────────────────────────────────────────────────────────────────────
#  WoRMS 유틸
# ────────────────────────────────────────────────────────────────────────────────
def get_aphia_id(scientific_name: str) -> int | None:
    """WoRMS에서 학명의 valid AphiaID를 반환한다. 3단계 폴백.

    1단계: AphiaRecordsByName (marine_only=true)
    2단계: AphiaRecordsByName (marine_only=false)  ← 담수·기수 어류 대응
    3단계: AphiaRecordsByMatchNames                ← 철자 유사 매칭
    """
    for marine_only in ["true", "false"]:
        url = (
            f"https://www.marinespecies.org/rest/AphiaRecordsByName/"
            f"{urllib.parse.quote(scientific_name)}"
            f"?like=false&marine_only={marine_only}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                records = json.loads(resp.read().decode("utf-8"))
            if records and isinstance(records, list):
                return records[0].get("valid_AphiaID") or records[0].get("AphiaID")
        except Exception:
            continue

    # 3단계: Fuzzy match
    url = (
        f"https://www.marinespecies.org/rest/AphiaRecordsByMatchNames"
        f"?scientificnames[]={urllib.parse.quote(scientific_name)}&marine_only=true"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data and isinstance(data, list) and len(data[0]) > 0:
            return data[0][0].get("valid_AphiaID") or data[0][0].get("AphiaID")
    except Exception:
        pass

    return None


# ────────────────────────────────────────────────────────────────────────────────
#  OBIS 유틸
# ────────────────────────────────────────────────────────────────────────────────
def _obis_count(*, taxonid: int | None, scientificname: str, geometry: str) -> int:
    """단일 해역의 OBIS 관측 횟수를 반환. 실패 시 -1."""
    params = {"geometry": geometry, "size": 0}
    if taxonid:
        params["taxonid"] = taxonid
    else:
        params["scientificname"] = scientificname
    try:
        r = requests.get("https://api.obis.org/v3/occurrence", params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("total", 0)
    except Exception:
        pass
    return -1


def fetch_region_counts(*, taxonid: int | None, scientificname: str) -> dict:
    """한반도 4개 해역 관측 횟수를 병렬 조회한다."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fmap = {
            ex.submit(_obis_count, taxonid=taxonid, scientificname=scientificname, geometry=geom): name
            for name, geom in REGIONS.items()
        }
        for f in concurrent.futures.as_completed(fmap):
            results[fmap[f]] = f.result()
    return results


# ────────────────────────────────────────────────────────────────────────────────
#  종 단위 처리
# ────────────────────────────────────────────────────────────────────────────────
def process_species(item: dict, accepted_name: str | None = None) -> dict:
    """한 종에 대해 AphiaID 조회 + OBIS 관측 횟수 조회를 수행하고
    결과를 item에 추가하여 반환한다."""
    sci_name = str(item.get("학명") or "").strip()
    aphia_id = get_aphia_id(sci_name)

    counts = fetch_region_counts(taxonid=aphia_id, scientificname=sci_name)

    # overall==0 이고 동의어로 복구된 종이면
    # OBIS accepted name 텍스트로 재시도 (OBIS 내부 텍스트 기준 저장 대응)
    if counts.get("overall", 0) == 0 and accepted_name:
        alt_counts = fetch_region_counts(taxonid=None, scientificname=accepted_name)
        if alt_counts.get("overall", 0) > 0:
            counts = alt_counts
            item["obis_queried_as"] = accepted_name

    item["aphia_id"] = aphia_id
    item["obis_occurrences"] = counts
    return item


# ────────────────────────────────────────────────────────────────────────────────
#  메인
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # ── 데이터 로드 ──
    print("[Step 4] 데이터 로드")
    with open(FILTERED_FILE, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)
    with open(MATCH_FILE, "r", encoding="utf-8") as f:
        match_data = json.load(f)
    with open(SYNONYM_FILE, "r", encoding="utf-8") as f:
        syn_data = json.load(f)

    # BioCLIP 직접 매칭 + 동의어 복구 학명 세트
    matched_set = set(sp.lower().strip() for sp in match_data.get("matched_species", []))
    recovered_mapping: dict[str, str] = syn_data.get("recovered_mapping", {})
    for missing_sp in recovered_mapping:
        matched_set.add(missing_sp.lower().strip())

    # Title-case accepted name 복원 (소문자 저장→표시용)
    def to_title(name: str) -> str:
        parts = name.split()
        return f"{parts[0].capitalize()} {' '.join(parts[1:])}" if len(parts) >= 2 else name.capitalize()

    # 어류 이면서 BioCLIP 매칭 종만 추출 (중복 학명 제거)
    seen: set[str] = set()
    targets: list[tuple[dict, str | None]] = []  # (item, accepted_name_or_None)
    for item in filtered_data.get("items", []):
        if item.get("세부분류군명", "").strip() != "어류":
            continue
        sci = str(item.get("학명") or "").strip()
        if not sci or sci.lower() in seen:
            continue
        if sci.lower() in matched_set:
            accepted = recovered_mapping.get(sci)
            targets.append((item, to_title(accepted) if accepted else None))
            seen.add(sci.lower())

    print(f"  처리 대상 어류 종 수: {len(targets):,}")

    # ── 병렬 처리 ──
    print(f"\n  OBIS 관측 횟수 수집 시작 (10 threads)...")
    success = error = 0
    processed: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        fmap = {ex.submit(process_species, item, accepted): item for item, accepted in targets}
        for i, f in enumerate(concurrent.futures.as_completed(fmap)):
            result_item = f.result()
            processed.append(result_item)
            ok = -1 not in result_item.get("obis_occurrences", {}).values()
            if ok:
                success += 1
                name = result_item.get("국명", "")
                sci  = result_item.get("학명", "")
                obs  = result_item["obis_occurrences"].get("overall", 0)
                print(f"  ✅ {name}({sci}) overall:{obs}")
            else:
                error += 1
            if (i + 1) % 100 == 0:
                print(f"  진행: [{i + 1}/{len(targets)}]")

    # ── overall==0 제거 ──
    before = len(processed)
    final = [
        item for item in processed
        if isinstance(item.get("obis_occurrences"), dict)
        and item["obis_occurrences"].get("overall", 0) > 0
    ]
    print(f"\n  overall==0 제거: {before} → {len(final)}종 남음")

    # ── 저장 ──
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output = {"total_count": len(final), "items": final}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"  저장 완료: {OUTPUT_FILE}")
    print(f"\n  최종 통계 → 성공:{success} / 오류:{error} / 최종 종 수:{len(final)}")

    # Sanity check
    assert len(final) > 0, "결과 데이터가 비어 있습니다."
    print("  [OK] Sanity check passed.")


if __name__ == "__main__":
    main()
