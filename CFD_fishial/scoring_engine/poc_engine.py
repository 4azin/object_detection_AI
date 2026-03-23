import pandas as pd

# -------------------------------------------------------------------
# [2.2] Prior Knowledge Base (사전 확률 DB - Mock 데이터)
# -------------------------------------------------------------------
mock_prior_data = [
    {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m", "Species": "자리돔", "P_prior": 0.9},
    {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m", "Species": "대구", "P_prior": 0.0}, # 한류성 (여름 제주 출현 X)
    {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m", "Species": "참돔", "P_prior": 0.5},
    {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m", "Species": "돌돔", "P_prior": 0.6},
    {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m", "Species": "아마존 담수어", "P_prior": 0.0}, # 담수어이므로 0
]

df_prior = pd.DataFrame(mock_prior_data)

# -------------------------------------------------------------------
# [Step 1] 입력 및 컨텍스트 추출 (F-1: 객관식 입력 인터페이스)
# -------------------------------------------------------------------
def get_context_interactive():
    print("=== [Step 1] 맥락 정보 입력 ===")
    regions = ["제주 남부", "동해 북부", "남해 동부", "서해 중부"]
    seasons = ["봄", "여름", "가을", "겨울"]
    depths = ["0-10m", "10-20m", "20-30m", "30m+"]

    print("해역: 1.제주 남부 2.동해 북부 3.남해 동부 4.서해 중부")
    region_idx = int(input("선택 (1-4): ")) - 1
    
    print("계절: 1.봄 2.여름 3.가을 4.겨울")
    season_idx = int(input("선택 (1-4): ")) - 1
    
    print("수심: 1.0-10m 2.10-20m 3.20-30m 4.30m+")
    depth_idx = int(input("선택 (1-4): ")) - 1

    return {
        "Region": regions[region_idx],
        "Season": seasons[season_idx],
        "Depth": depths[depth_idx]
    }

# -------------------------------------------------------------------
# [Step 2] 후보군 동적 필터링
# -------------------------------------------------------------------
def filter_candidates(context):
    print(f"\n=== [Step 2] 후보군 동적 필터링 ===")
    print(f"입력된 주변 컨텍스트: {context}")
    
    # 1. 선택된 컨텍스트에 해당하는 Prior 데이터 조회
    condition = (df_prior["Region"] == context["Region"]) & \
                (df_prior["Season"] == context["Season"]) & \
                (df_prior["Depth"] == context["Depth"])
    
    current_prior = df_prior[condition]
    
    if current_prior.empty:
        print("⚠️ 해당 컨텍스트에 대한 사전 DB 데이터가 없습니다. 필터링을 건너뜁니다.")
        return {}, []
    
    # 2. 해당 환경에서 출현 확률이 0인 종 추출
    zero_prob_species = current_prior[current_prior["P_prior"] == 0.0]["Species"].tolist()
    print(f"필터링 대상으로 판별된 종 (P_prior=0): {zero_prob_species}")
    
    # 종별 사전 확률 매핑 딕셔너리 생성
    prior_dict = dict(zip(current_prior["Species"], current_prior["P_prior"]))
    return prior_dict, zero_prob_species

# -------------------------------------------------------------------
# [Step 3] 가중치 기반 재정렬 (F-2, F-3)
# -------------------------------------------------------------------
def calculate_final_score(model_results, prior_dict, zero_prob_species, alpha=0.4):
    print("\n=== [Step 3] 가중치 기반 재정렬 ===")
    
    final_results = []
    
    for item in model_results:
        species = item["Species"]
        p_model = item["P_model"]
        
        # [F-3] 예외 처리: 희귀 종 보호 (모델 최고 확신 시)
        if p_model >= 0.95:
            print(f"⭐ 희귀 종 보호 발동! : '{species}' (P_model={p_model}) -> 컨텍스트 필터링 무시")
            p_prior = prior_dict.get(species, 0.0)
            final_score = p_model * (alpha * p_prior + (1 - alpha))
            final_results.append({"Species": species, "P_model": p_model, "P_prior": p_prior, "Final_Score": final_score})
            continue

        # [Step 2 연계] 출현 확률이 0인 종은 즉시 제외
        if zero_prob_species and species in zero_prob_species:
            print(f"🚫 강력 필터링됨 (Prior=0): '{species}'")
            continue
            
        # 알려지지 않은 종의 Prior 기본값 할당 (예: 0.1)
        p_prior = prior_dict.get(species, 0.1) 
        
        # 공식: Final_Score = P_model * (alpha * P_prior + (1 - alpha))
        final_score = p_model * (alpha * p_prior + (1 - alpha))
        
        final_results.append({
            "Species": species,
            "P_model": p_model,
            "P_prior": p_prior,
            "Final_Score": final_score
        })
        
    # [F-2] 내림차순 정렬하여 상위 리턴
    final_results.sort(key=lambda x: x["Final_Score"], reverse=True)
    return final_results

# -------------------------------------------------------------------
# [5] 테스트 및 검증 시나리오 (PoC)
# -------------------------------------------------------------------
def run_test_scenario_a():
    print("\n\n" + "="*50)
    print("▶ 테스트 A (유효성 검증): 제주도 여름 수심 10-20m")
    print("목표: 한류성 어종인 '대구' 점수 하락, '자리돔' 점수 상승 확인")
    
    context = {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m"}
    prior_dict, zero_prob_species = filter_candidates(context)

    # 비전 AI 모델의 순수 예측 결과 (모델이 비슷하게 헷갈렸다고 가정한 상태)
    model_results = [
        {"Species": "대구", "P_model": 0.82},   # 비전 모델은 이미지 특성으로 인해 대구를 82% 확신
        {"Species": "자리돔", "P_model": 0.78}, # 비전 모델은 자리돔을 78% 확신
        {"Species": "참돔", "P_model": 0.40},
    ]
    
    print("\n[비전 AI 모델의 초기 (Raw) 예측 확률]")
    for res in model_results: print(f" - {res['Species']}: {res['P_model']:.2f}")
        
    final_results = calculate_final_score(model_results, prior_dict, zero_prob_species, alpha=0.4)
    
    print("\n[최종 보정된 (Reranked) 결과]")
    for i, res in enumerate(final_results[:5], 1): 
        print(f"{i}순위: {res['Species']:10} | Raw:{res['P_model']:.2f} | Prior:{res['P_prior']:.2f} | Final:{res['Final_Score']:.4f}")

def run_test_scenario_b():
    print("\n\n" + "="*50)
    print("▶ 테스트 B (오탐 제거 및 희귀종 보호): 아마존 담수어 vs 희귀심해어")
    print("목표: 담수어 오탐 완전 배제 확인 및 확신(0.95+) 모델값의 보호 로직 작동 여부 확인")
    
    context = {"Region": "제주 남부", "Season": "여름", "Depth": "10-20m"}
    prior_dict, zero_prob_species = filter_candidates(context)

    model_results = [
        {"Species": "아마존 담수어", "P_model": 0.90}, # 단순 형태적 유사성으로 인한 오탐지
        {"Species": "희귀심해어", "P_model": 0.98},   # 모델이 특징을 완벽히 읽고 압도적으로 확신 (0.95 이상)
        {"Species": "돌돔", "P_model": 0.60},
    ]
    
    print("\n[비전 AI 모델의 초기 (Raw) 예측 확률]")
    for res in model_results: print(f" - {res['Species']}: {res['P_model']:.2f}")
        
    final_results = calculate_final_score(model_results, prior_dict, zero_prob_species, alpha=0.4)
    
    print("\n[최종 보정된 (Reranked) 결과]")
    for i, res in enumerate(final_results[:5], 1): 
        print(f"{i}순위: {res['Species']:10} | Raw:{res['P_model']:.2f} | Prior:{res['P_prior']:.2f} | Final:{res['Final_Score']:.4f}")

if __name__ == "__main__":
    run_test_scenario_a()
    run_test_scenario_b()
    print("\n" + "="*50)
    print("✅ PoC 스코어링 엔진 실행 완료")
