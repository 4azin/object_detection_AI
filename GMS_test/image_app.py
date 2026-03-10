import os
import base64
import requests
import gradio as gr
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from PIL import Image
import io

def encode_image_to_base64(image_path, max_size=(512, 512)):
    """이미지 파일을 읽어 크기를 줄인 후 Base64 문자열로 변환합니다."""
    try:
        with Image.open(image_path) as img:
            # 이미지 모드가 RGB가 아니면 변환 (예: RGBA -> RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 비율을 유지하며 최대 크기로 리사이징
            img.thumbnail(max_size)
            
            # 메모리 버퍼에 JPEG 형식으로 저장 (압축률 85)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            # Base64 변환
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 리사이징 중 오류: {e}")
        # 실패 시 원본 그대로 변환 시도
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_llm(image_filepath, user_prompt):
    # 환경 변수에서 GMS_KEY 가져오기
    gms_key = os.getenv("GMS_KEY")
    if not gms_key:
        return "Error: GMS_KEY 환경 변수가 설정되지 않았습니다. .env 파일이나 시스템 환경 변수를 확인하세요."
    
    if not image_filepath:
        return "오류: 이미지를 먼저 업로드해주세요."
        
    # 이미지를 base64로 인코딩
    try:
        base64_image = encode_image_to_base64(image_filepath)
    except Exception as e:
        return f"이미지 처리 중 오류 발생: {str(e)}"

    # GMS 엔드포인트 URL
    url = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gms_key}"
    }
    
    # 기본 프롬프트 설정 (사용자 입력이 없는 경우 기본 설명 요청)
    prompt_text = user_prompt if user_prompt and user_prompt.strip() else "이 이미지에 대해 자세히 설명해 주세요."
    
    # OpenAI Vision 모델 API 포맷에 맞게 메시지 구성 (텍스트 + 이미지)
    payload = {
        "model": "gpt-5-mini",
        "messages": [
            {
                "role": "developer",
                "content": "You are a helpful assistant that analyzes images and answers in Korean."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # JPEG, PNG 등에 상관없이 base64 데이터로 전송 구조 생성
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        # API 호출
        print("API 호출 시작...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("API 응답 수신:", result)
        
        # 응답 텍스트 추출
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("추출된 텍스트:", content)
            return content
        else:
            print("응답 구조 이상:", result)
            return f"응답 구조 이상 구조 확인 필요:\n{result}"
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러 발생:\n상태 코드: {response.status_code}\n에러 내용: {response.text}")
        return f"HTTP 에러 발생:\n상태 코드: {response.status_code}\n에러 내용: {response.text}"
    except Exception as e:
        return f"기타 에러 발생: {str(e)}"

# Gradio 기반 웹 UI 구성
with gr.Blocks(title="GMS Image & Text Analyzer") as demo:
    gr.Markdown("# 🤖 이미지 분석 AI (GMS gpt-5-mini)")
    gr.Markdown("이미지를 업로드하고 원하는 프롬프트를 입력하면, AI가 이미지를 분석하여 답변을 제공합니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 파일 경로 형태로 이미지를 받음
            image_input = gr.Image(type="filepath", label="이미지 업로드")
            
            # 추가 프롬프트를 입력받는 텍스트 박스
            prompt_input = gr.Textbox(
                label="추가 프롬프트 (선택사항)", 
                placeholder="예: 이 이미지에서 가장 특징적인 부분을 3가지로 요약해줘.",
                lines=3
            )
            submit_btn = gr.Button("분석 요청하기", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="AI 분석 결과", lines=15)
            
    # 버튼 클릭 시 analyze_image_with_llm 함수 매핑
    submit_btn.click(
        fn=analyze_image_with_llm,
        inputs=[image_input, prompt_input],
        outputs=output_text
    )

if __name__ == "__main__":
    # 외부 접속이 필요하다면 share=True 로 변경 가능
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

