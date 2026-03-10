import os
import io
import json
import base64
import requests
from datetime import datetime
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

# 환경변수 로딩
load_dotenv()

OPENAI_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://gms.ssafy.io/gmsapi/api.anthropic.com/v1/messages"

PROFILES = {
    "기본값: 가성비 모델 조합 (gpt-5-mini + o3-mini)": {
        "vision_model": "gpt-5-mini",
        "synthesis_model": "o3-mini"
    },
    "울트라값: 최고 성능 조합 (All GPT-5) [!!크레딧주의!!]": {
        "vision_model": "gpt-5",
        "synthesis_model": "gpt-5"
    },
    "프로값: 감성적 서술 조합 (Claude + GPT Hybrid) [!!크레딧주의!!]": {
        "vision_model": "claude-opus-4-5-20251101",
        "synthesis_model": "gpt-5.2"
    },
    "아마추어값: 분석적 서술 조합 (gpt-4o + o3-mini) [!!크레딧주의!!]": {
        "vision_model": "gpt-4o",
        "synthesis_model": "o3-mini"
    },
    
}

PERSONAS = {
    "기록형 다이버 (Record-keeper)": "당신은 전문 수중 객관적 기록관입니다. 사진 속의 해양 생물, 지형, 수중 상태(시야, 조류 등) 및 다이버의 스킬적 요소를 관찰 일지 형식으로 객관적이고 명확하게 묘사하세요. (반드시 200자 이내 핵심만 요약)",
    "예술가형 다이버 (Artist)": "당신은 예술적인 수중 사진 작가입니다. 빛의 산란, 물결의 질감, 산호와 생물들의 아름다운 색감 조화를 감각적이고 화려한 문체로 시적인 느낌이 나게 묘사하세요. (반드시 200자 이내 핵심만 묘사)",
    "실속형 다이버 (Pragmatist)": "당신은 효율성을 중시하는 다이빙 가이드입니다. 사진에 등장하는 핵심 타겟 생물 이름, 주요 지형지물 1개, 그리고 현재 다이버의 위치 정보 등 가장 중요한 팩트 3가지만을 불필요한 미사여구 없이 간결하게 나열하세요. (반드시 150자 이내)",
    "블로거형 다이버 (Blogger)": "당신은 트렌디한 스쿠버다이빙 인플루언서입니다. 사진 속 상황을 SNS에 올릴 귀엽고 재치 있는 말투(이모지 듬뿍)로 다이빙의 즐거운 순간을 생동감 있게 자랑하듯 묘사하세요. (반드시 200자 이내, 해시태그 포함)"
}

STEP2_PROMPT = "5개의 시계열 텍스트 소스를 바탕으로 다이빙 로그를 작성하세요. 각 단계는 시간 순서로 연결되어야 하며, 전체적인 다이빙의 흐름이 끊기지 않도록 연결하세요. 선택된 페르소나의 문체를 유지하여 하나의 다이빙 로그를 완성하세요."

def encode_image_to_base64(image_path, max_size=(512, 512)):
    """이미지 파일을 읽어 크기를 줄인 후 Base64 문자열로 변환합니다."""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 리사이징 중 오류: {e}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def call_openai_api(model_name, system_prompt, user_content):
    gms_key = os.getenv("GMS_KEY")
    if not gms_key:
        raise ValueError("GMS_KEY 환경 변수가 설정되지 않았습니다.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gms_key}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    }

    response = requests.post(OPENAI_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"OpenAI 응답 구조 에러: \n{result}")

def call_anthropic_api(model_name, system_prompt, user_content):
    gms_key = os.getenv("GMS_KEY")
    if not gms_key:
        raise ValueError("GMS_KEY 환경 변수가 설정되지 않았습니다.")
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": gms_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model_name,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content}
        ]
    }

    response = requests.post(ANTHROPIC_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    
    if "content" in result and len(result["content"]) > 0:
        return result["content"][0]["text"]
    else:
        raise ValueError(f"Anthropic 응답 구조 에러: \n{result}")

def analyze_frame(image_path, model_name, persona_prompt):
    if not image_path:
        return ""
    
    base64_image = encode_image_to_base64(image_path)
    system_prompt = persona_prompt
    
    if "claude" in model_name.lower():
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            },
            {
                "type": "text",
                "text": "제공된 이미지를 분석하세요."
            }
        ]
        return call_anthropic_api(model_name, system_prompt, user_content)
    else:
        user_content = [
            {"type": "text", "text": "제공된 이미지를 분석하세요."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        return call_openai_api(model_name, system_prompt, user_content)

def synthesize_logbook(texts, model_name):
    valid_texts = [txt for txt in texts if txt and txt.strip()]
    
    if not valid_texts:
        return "분석된 영상/이미지 텍스트가 없어 로그북을 생성할 수 없습니다."
        
    combined_texts = "\n\n".join([f"[Frame {i+1}]\n{txt}" for i, txt in enumerate(valid_texts)])
    
    system_prompt = "You are an expert dive logbook author."
    user_content = f"{STEP2_PROMPT}\n\n[입력 묘사 소스]\n{combined_texts}"
    
    if "claude" in model_name.lower():
        return call_anthropic_api(model_name, system_prompt, user_content)
    else:
        return call_openai_api(model_name, system_prompt, user_content)

def generate_pipeline(profile_choice, persona_choice, img1, img2, img3, img4, img5):
    try:
        models = PROFILES[profile_choice]
        vision_model = models["vision_model"]
        synthesis_model = models["synthesis_model"]
        persona_prompt = PERSONAS[persona_choice]
        
        images = [img1, img2, img3, img4, img5]
        descriptions = []
        
        # Step 1: 개별 프레임 분석
        print(f"--- [Step 1] Image-to-Text 분석 시작 (Model: {vision_model}, Persona: {persona_choice}) ---")
        for idx, img in enumerate(images):
            if img:
                print(f"프레임 {idx+1} 묘사 중...")
                res = analyze_frame(img, vision_model, persona_prompt)
            else:
                res = ""
            descriptions.append(res)
            
        # Step 2: 묘사 통합 및 로그북 생성
        print(f"--- [Step 2] Sequence-to-Logbook 생성 시작 (Model: {synthesis_model}) ---")
        final_logbook = synthesize_logbook(descriptions, synthesis_model)
        
        print("--- 파이프라인 수행 완료 ---")
        # 비어있는 응답에 대해 안내 메시지 추가
        display_desc = [d if d else "이미지가 등록되지 않았습니다." for d in descriptions]
        
        # --- 히스토리 자동 저장 로직 추가 ---
        os.makedirs("history", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_data = {
            "timestamp": timestamp,
            "profile": profile_choice,
            "persona": persona_choice,
            "step1_descriptions": display_desc,
            "final_logbook": final_logbook
        }
        
        history_filepath = os.path.join("history", f"logbook_history_{timestamp}.json")
        try:
            with open(history_filepath, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=4)
            save_msg = f"✅ (히스토리 저장 완료) 파일 위치: {history_filepath}"
        except Exception as file_e:
            save_msg = f"⚠️ 히스토리 저장 실패: {file_e}"
            print(save_msg)
        # ------------------------------------
        
        return display_desc[0], display_desc[1], display_desc[2], display_desc[3], display_desc[4], final_logbook, save_msg
        
    except Exception as e:
        err_msg = f"에러 발생: {str(e)}"
        print(err_msg)
        return err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, err_msg

# Gradio 기반 웹 UI 구성
with gr.Blocks(title="Divery AI Logbook Generation Pipeline") as demo:
    gr.Markdown("# 🤿 Divery AI Logbook Generation Pipeline")
    gr.Markdown("동영상에서 추출된 **5개의 주요 프레임(이미지)**을 시계열로 분석하여, 다이빙 전체 과정을 서사적으로 구성한 최종 로그북을 생성합니다.")
    
    with gr.Row():
        profile_dropdown = gr.Dropdown(
            choices=list(PROFILES.keys()), 
            value=list(PROFILES.keys())[0], 
            label="성능 프로파일 선택 (권장 프리셋)",
            info="Step 1(Vision) 및 Step 2(텍스트 합성) 모델의 조합을 선택하세요."
        )
        persona_dropdown = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value=list(PERSONAS.keys())[0],
            label="다이버 성향 (Persona)",
            info="텍스트의 톤앤매너와 길이를 결정합니다."
        )
        
    gr.Markdown("### Step 1: 시계열 이미지 5장 업로드")
    with gr.Row():
        img_inputs = [gr.Image(type="filepath", label=f"Frame {i+1}") for i in range(5)]
        
    btn = gr.Button("🚀 AI 다이빙 로그북 생성 파이프라인 시작", variant="primary")
    
    gr.Markdown("---")
    
    gr.Markdown("### 🔍 Step 1 Analysis Results (Image-to-Text)")
    with gr.Row():
        text_outputs = [gr.Textbox(label=f"Frame {i+1} 상세 묘사", lines=10) for i in range(5)]
        
    gr.Markdown("### 📝 Step 2 Final Logbook (Sequence-to-Logbook)")
    final_output = gr.Textbox(label="통합 다이빙 서사 로그북 및 구조화 데이터", lines=25)
    
    save_status_output = gr.Textbox(label="자동 저장 히스토리 안내", lines=1)
    
    btn.click(
        fn=generate_pipeline,
        inputs=[profile_dropdown, persona_dropdown] + img_inputs,
        outputs=text_outputs + [final_output, save_status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, theme=gr.themes.Soft())
