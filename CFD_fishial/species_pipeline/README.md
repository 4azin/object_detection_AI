# Species Pipeline

한국 해역 다이버 기준 생물 종 데이터 수집 및 OBIS 관측 횟수 추출 파이프라인.

BioCLIP2 제로샷 분류 모델의 **종 후보군(Species Pool)** 을 구성하는 데 사용됩니다.

---

## 디렉토리 구조

```
species_pipeline/
├── README.md
├── step1_filter_species.py     # Step 1: MBRIS 종 데이터 다이버 기준 필터링
├── step2_match_bioclip.py      # Step 2: BioCLIP2 taxonomy와의 커버리지 분석
├── step3_resolve_synonyms.py   # Step 3: GBIF + WoRMS 동의어 정규화
├── step4_fetch_obis.py         # Step 4: OBIS 해역별 관측 횟수 수집
│
├── data/
│   └── result_bioinfo.json     # MBRIS 원본 종 데이터 (전체, Step 1 입력)
│
└── outputs/                    # 각 단계 산출물 (자동 생성)
    ├── 01_filtered_species.json
    ├── 02_bioclip_match.json
    ├── 03_synonym_mapping.json
    └── 04_obis_prior_db.json   # ← 최종 산출물
```

---

## 실행 방법

> **전제 조건:** `data/result_bioinfo.json` 파일이 존재해야 합니다.
> (MBRIS 포털에서 내려받은 전체 종 목록 JSON)

### 요구 패키지 설치

```bash
pip install requests
```

### 단계별 실행 (순서 중요)

```bash
cd species_pipeline

python step1_filter_species.py   # ~1초
python step2_match_bioclip.py    # ~30초 (Hugging Face 다운로드)
python step3_resolve_synonyms.py # ~3분  (GBIF + WoRMS API 병렬)
python step4_fetch_obis.py       # ~10분 (WoRMS + OBIS API 병렬)
```

각 단계는 독립적으로 실행 가능하며, 이전 단계의 `outputs/*.json`을 입력으로 사용합니다.

---

## 각 단계 설명

### Step 1 – 종 필터링 (`step1_filter_species.py`)

**입력:** `data/result_bioinfo.json` (MBRIS 전체)  
**출력:** `outputs/01_filtered_species.json`

MBRIS의 15,000+ 종에서 다이버가 수중 환경에서 육안으로 식별 가능한 종을 추출합니다.

| 필터 기준 | 조건 |
|---|---|
| 어류 | `세부분류군명` == "어류" |
| 문어류 | `Order` 에 "Octopoda" 포함 |
| 오징어류 | `국명`에 "오징어" 포함 AND Phylum == "Mollusca" |
| 해파리류 | `세부분류군명` == "자포동물" AND `국명`에 "해파리" 포함 |
| 파충류 | `세부분류군명` == "파충류" |
| 포유류 | `세부분류군명` == "포유류" |

---

### Step 2 – BioCLIP Taxonomy 매칭 (`step2_match_bioclip.py`)

**입력:** `outputs/01_filtered_species.json`  
**출력:** `outputs/02_bioclip_match.json`

BioCLIP2 (TreeOfLife-200M) 모델에서 실제로 인식 가능한 종인지 확인합니다.

- Hugging Face에서 `txt_emb_species.json`을 다운로드
- 분류 계층 배열의 마지막 두 원소를 조합해 이명법 학명으로 재구성
- `matched_species` / `missing_species` 분리

---

### Step 3 – 동의어 정규화 (`step3_resolve_synonyms.py`)

**입력:** `outputs/02_bioclip_match.json`  
**출력:** `outputs/03_synonym_mapping.json`

텍스트 매칭 실패 종(=이명·동의어 문제)을 외부 분류 DB로 복구합니다.

| 조회 순서 | API | 대상 |
|---|---|---|
| 1차 | GBIF Backbone Taxonomy | 담수·육상 포함 전체 생물 |
| 2차 | WoRMS (World Register of Marine Species) | 해양 생물 전문 동의어 |

> 예시: `Leptojulis poecilepterus` → WoRMS에서 `Parajulis poecilepterus`로 매핑

---

### Step 4 – OBIS 관측 횟수 수집 (`step4_fetch_obis.py`)

**입력:** `outputs/01~03_*.json`  
**출력:** `outputs/04_obis_prior_db.json`

한반도 해역 4개 구역 기준 관측 횟수를 각 종별로 수집합니다.

| 구역 | WKT 범위 |
|---|---|
| 전체 (overall) | 124~132°E, 32~39°N |
| 서해 (west) | 124~127°E, 34~39°N |
| 남해 (south) | 124~132°E, 32~34°N |
| 동해 (east) | 127~132°E, 34~39°N |

**정확도 보장 로직 (3단계):**
1. WoRMS `valid_AphiaID` 조회 → OBIS `taxonid` 기반 조회 (학명 이명 문제 해결)
2. AphiaID 없는 종 → scientificname 텍스트 폴백
3. overall==0이고 동의어 복구 종 → accepted name 텍스트로 재조회

최종적으로 관측 기록이 전혀 없는 종(overall==0)은 결과에서 제거합니다.

---

## 최종 산출물 스키마 (`outputs/04_obis_prior_db.json`)

```json
{
  "total_count": 795,
  "items": [
    {
      "학명": "Sebastes schlegelii",
      "국명": "조피볼락",
      "세부분류군명": "어류",
      "Phylum": "Chordata",
      "Order": "Perciformes",
      "Family": "Scorpaenidae Risso, 1827",
      "aphia_id": 274849,
      "obis_occurrences": {
        "overall": 304,
        "west":    169,
        "south":    30,
        "east":    105
      }
    }
  ]
}
```
