import os
import requests
from dotenv import load_dotenv

# .env 파일이 있다면 로드합니다.
load_dotenv()

def test_gms_llm():
    # 환경 변수에서 GMS_KEY를 가져옵니다.
    gms_key = os.getenv("GMS_KEY")
    if not gms_key:
        print("Error: GMS_KEY 환경 변수가 설정되지 않았습니다.")
        print("실행 예시: GMS_KEY=your_api_key python test_llm.py")
        print("또는 프로젝트 루트 혹은 GMS_test 폴더 내의 .env 파일에 GMS_KEY=your_api_key 를 추가하세요.")
        return

    # 엔드포인트 URL
    url = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions"
    
    # HTTP 헤더 설정
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gms_key}"
    }
    
    # 요청 페이로드 설정 (OpenAI API Reference 파라미터 적용)
    payload = {
        "model": "gpt-5-mini",
        "messages": [
            {
                "role": "developer",
                "content": "Answer in Korean"
            },
            {
                "role": "user",
                "content": "Summarize the 2024 total production data."
            }
        ],
        # ---------- 추가적인 파라미터 예시 ----------
        # temperature: 생성 텍스트의 다양성/창의성 조절 (0.0 ~ 2.0). 낮을수록 결정론적, 높을수록 창의적
        "temperature": 0.7,
        # max_tokens: 응답으로 받을 최대 토큰 수 제한
        "max_tokens": 1000,
        # top_p: nucleus sampling에서 다양한 단어 선택의 확률 질량 기준 조정 (0.0 ~ 1.0)
        "top_p": 1.0,
        # frequency_penalty: 이미 사용된 단어의 빈도수에 비례하여 패널티 부여 (반복 억제, -2.0 ~ 2.0)
        "frequency_penalty": 0.0,
        # presence_penalty: 이미 사용된 단어의 존재 여부에 따라 패널티 부여 (새로운 주제 유도, -2.0 ~ 2.0)
        "presence_penalty": 0.0
    }
    
    print("GMS API (gpt-5-mini) 호출 중...")
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # HTTP 응답 에러 상태 코드인 경우 예외 발생
        response.raise_for_status()
        
        # JSON 응답 파싱
        result = response.json()
        print("\n[API 응답 성공]")
        
        # 실제 텍스트 응답 추출 후 출력
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n--- AI 응답 (gpt-5-mini) ---")
            print(content)
            print("----------------------------")
        else:
            print("응답 구조에 choices 값이 없습니다:", result)
            
    except requests.exceptions.HTTPError as e:
        print("\n[HTTP 에러 발생]")
        print("상태 코드:", response.status_code)
        print("에러 내용:", response.text)
    except Exception as e:
        print("\n[에러 발생]")
        print(e)

if __name__ == "__main__":
    test_gms_llm()
