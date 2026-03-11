import gradio as gr
import os
import shutil
import time
from pathlib import Path

# HIGHLIGHT.py의 핵심 모듈 가져오기
from HIGHLIGHT import extract_frames_from_video, HighlightEngine

# image_app.py의 핵심 모듈 가져오기
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'GMS_test'))
from image_app import analyze_image_with_llm

def process_video_for_highlights(video_path):
    if not video_path:
        return "영상을 업로드해주세요.", []

    print(f"업로드된 비디오 경로: {video_path}")
    
    # 임시 폴더(더 이상 디스크 I/O 프레임 저장을 하지 않음) 및 결과 폴더 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    final_highlights_dir = os.path.join(base_dir, "gradio_highlights")
    
    # 이전 작업 생성물 초기화
    if os.path.exists(final_highlights_dir): shutil.rmtree(final_highlights_dir)
    Path(final_highlights_dir).mkdir(parents=True, exist_ok=True)

    nima_weights_path = os.path.join(base_dir, "weights", "nima_resnet50_finetuned.pth")

    status_message = "1. 영상에서 프레임을 메모리로 추출하고 있습니다 (초고속 In-Memory)...\n"
    print("프레임 추출 시작...")
    yield status_message, []

    # 1. 비디오 프레임 추출 (디스크에 저장하지 않고 RAM에 바로 올림)
    try:
        extracted_frames_in_memory = extract_frames_from_video(video_path, fps=1)
    except Exception as e:
        yield f"프레임 추출 중 에러 발생: {str(e)}", []
        return

    status_message += f"프레임 추출 완료! 총 {len(extracted_frames_in_memory)}장 확보\n"
    status_message += "2. 인공지능이 최고의 하이라이트 5장면을 선별하고 있습니다...\n"
    print("AI 하이라이트 판독 중...")
    yield status_message, []

    # 2. 엔진 가동 및 분석
    engine = HighlightEngine(nima_weights=nima_weights_path)
    top_highlights = engine.process_video_and_rank(extracted_frames_in_memory, k=5)
    
    status_message += "하이라이트 5장면 선별 완료!\n"
    status_message += "3. 각 장면에 대해 인공지능이 코멘트를 작성하고 있습니다...\n"
    print("거대 언어 모델(LLM) 코멘트 작성 시작...")
    yield status_message, []

    # 화면에 보여줄 갤러리/결과물 준비 목록
    final_results = []
    
    import cv2
    
    # AI 코멘트 작성 및 최종 저장
    for idx, res in enumerate(top_highlights, 1):
        clean_ts = res['timestamp'].replace(":", "")
        new_file_name = f"rank{idx}_{clean_ts}_score{res['final_score']}.jpg"
        save_path = os.path.join(final_highlights_dir, new_file_name)

        # 뽑힌 프레임을 최종 폴더로 기록 (디스크 저장)
        cv2.imwrite(save_path, res['frame_bgr'], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # LLM에게 2~3줄 코멘트 요청 (한국어 지시 명확화)
        prompt = f"이 사진은 수중 다이빙 영상의 가장 멋진 하이라이트 장면입니다. 탐지된 주요 객체는 {res['objects']} 입니다. 이 장면에 대해 관찰자가 감탄할 만한 2~3줄 길이의 한국어 코멘트를 예쁘게 작성해주세요."
        comment = analyze_image_with_llm(save_path, prompt)
        
        caption_text = f"{idx}위 ({res['timestamp']}) - {comment}"
        # Gradio 갤러리에 들어갈 (이미지경로, 설명 텍스트) 튜플 추가
        final_results.append((save_path, caption_text))
        
        # 중간 진행상황 업데이트
        current_status = status_message + f" - {idx}번째 장면 코멘트 작성 완료!\n"
        yield current_status, final_results

    # 3. 마무리
    final_status = "모든 작업이 완료되었습니다! 아래 생성된 하이라이트 영상과 코멘트를 확인하세요."
    yield final_status, final_results

# Gradio 웹 UI 구성
with gr.Blocks(title="AI Video Highlight & Commenter") as demo:
    gr.Markdown("# 🎬 AI 비디오 하이라이트 자동 추출 및 코멘트 생성기")
    gr.Markdown("영상을 업로드하시면 최고의 장면 5개를 뽑아 AI가 2~3줄짜리 찰떡 코멘트를 달아줍니다!")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="비디오 업로드 (mp4, avi 등)")
            submit_btn = gr.Button("하이라이트 및 코멘트 생성하기", variant="primary")
            
        with gr.Column(scale=1):
            status_output = gr.Textbox(label="진행 상태", lines=5, interactive=False)
            
    with gr.Row():
        # 결과 이미지를 갤러리 형태로 보여줌
        gallery_output = gr.Gallery(
            label="하이라이트 장면 5선 (클릭해서 확대 가능)", 
            show_label=True, 
            elem_id="gallery", 
            columns=5,  # 5개 이미지가 한 줄에 나오게 설정
            rows=1, 
            object_fit="contain", 
            height="auto"
        )
        
    # 버튼 클릭 시 process_video_for_highlights 함수 매핑
    submit_btn.click(
        fn=process_video_for_highlights,
        inputs=[video_input],
        outputs=[status_output, gallery_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
