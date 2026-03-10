"""
classify_app.py
===============
BioCLIP-2 종 분류 Gradio Web App
- Open-Domain: TreeOfLife-200M 전체 종 대상 분류
- Zero-Shot: 사용자 지정 종 목록 대상 분류
- Top-3 예측 + 신뢰도(%) 표시

사용법:
    python classify_app.py
    → http://localhost:7860 에서 브라우저로 접속
"""

import collections
import heapq
import json
import os
import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ── open_clip ──
try:
    from open_clip import create_model, get_tokenizer
except ImportError:
    print("❌ open_clip_torch가 설치되지 않았습니다.")
    print("   설치: pip install open_clip_torch")
    sys.exit(1)

# ── 로깅 ──
log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════════════
MODEL_STR = "hf-hub:imageomics/bioclip-2"
TOKENIZER_STR = "ViT-L-14"
HF_DATA_STR = "imageomics/TreeOfLife-200M"

TOP_K = 3  # Top-3 결과 표시
MIN_PROB = 1e-9

# ── 이미지 전처리 (CLIP ViT-L/14 표준) ──
preprocess_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

# ── 분류 계급 ──
RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")

# ── OpenAI ImageNet 프롬프트 템플릿 (BioCLIP-2 공식 사용) ──
TEMPLATES = [
    lambda c: f"a photo of a {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a photo of a small {c}.",
]

# ═══════════════════════════════════════════════════════════
#  디바이스 설정
# ═══════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🖥️  GPU: {gpu_name}")
else:
    device = torch.device("cpu")
    logger.info("⚠️  CUDA 사용 불가. CPU 모드로 실행합니다.")


# ═══════════════════════════════════════════════════════════
#  모델 로드
# ═══════════════════════════════════════════════════════════
logger.info(f"📦 BioCLIP-2 모델 로드 중... ({MODEL_STR})")
model = create_model(MODEL_STR, output_dict=True, require_pretrained=True)
model = model.to(device)
model.eval()
logger.info("✅ BioCLIP-2 모델 로드 완료")

tokenizer = get_tokenizer(TOKENIZER_STR)

# ── TreeOfLife-200M 종 텍스트 임베딩 (Open-Domain용) ──
logger.info("📥 TreeOfLife-200M 텍스트 임베딩 다운로드 중...")
txt_emb = torch.from_numpy(np.load(hf_hub_download(
    repo_id=HF_DATA_STR,
    filename="embeddings/txt_emb_species.npy",
    repo_type="dataset",
))).to(device)

with open(hf_hub_download(
    repo_id=HF_DATA_STR,
    filename="embeddings/txt_emb_species.json",
    repo_type="dataset",
), encoding="utf-8") as fd:
    txt_names = json.load(fd)

logger.info(f"✅ 텍스트 임베딩 로드 완료 ({txt_emb.shape[1]:,}종)")


# ═══════════════════════════════════════════════════════════
#  추론 함수
# ═══════════════════════════════════════════════════════════
def format_name(taxon, common):
    """분류 이름 포맷팅 (학명 + 일반명)"""
    taxon = " ".join(taxon)
    if not common:
        return taxon
    return f"{taxon} ({common})"


@torch.no_grad()
def get_txt_features(classnames, templates):
    """클래스 이름 목록의 텍스트 임베딩을 계산합니다."""
    all_features = []
    for classname in classnames:
        txts = [template(classname) for template in templates]
        txts = tokenizer(txts).to(device)
        txt_features = model.encode_text(txts)
        txt_features = F.normalize(txt_features, dim=-1).mean(dim=0)
        txt_features /= txt_features.norm()
        all_features.append(txt_features)
    all_features = torch.stack(all_features, dim=1)
    return all_features


@torch.no_grad()
def open_domain_classification(img, rank: int) -> dict[str, float]:
    """
    Open-Domain 분류: TreeOfLife-200M 전체 종 대상.
    이미지 임베딩과 사전 계산된 텍스트 임베딩 간 유사도를 계산합니다.
    """
    if img is None:
        return {}

    logger.info(f"🔬 Open-Domain 분류 시작 (계급: {RANKS[rank]})")

    img_tensor = preprocess_img(img).to(device)
    img_features = model.encode_image(img_tensor.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_emb).squeeze()
    probs = F.softmax(logits, dim=0)

    # Species 레벨
    if rank + 1 == len(RANKS):
        topk = probs.topk(TOP_K)
        result = {
            format_name(*txt_names[i]): float(prob)
            for i, prob in zip(topk.indices, topk.values)
        }
        logger.info(f"📊 결과: {result}")
        return result

    # 상위 계급: 종 확률을 합산
    output = collections.defaultdict(float)
    for i in torch.nonzero(probs > MIN_PROB).squeeze():
        output[" ".join(txt_names[i][0][: rank + 1])] += probs[i].item()

    topk_names = heapq.nlargest(TOP_K, output, key=output.get)
    result = {name: output[name] for name in topk_names}
    logger.info(f"📊 결과: {result}")
    return result


@torch.no_grad()
def zero_shot_classification(img, cls_str: str) -> dict[str, float]:
    """
    Zero-Shot 분류: 사용자가 지정한 종 목록 대상.
    텍스트 입력란에 줄바꿈으로 후보 종명을 입력합니다.
    """
    if img is None or not cls_str.strip():
        return {}

    classes = [cls.strip() for cls in cls_str.split("\n") if cls.strip()]
    logger.info(f"🔬 Zero-Shot 분류 시작 ({len(classes)}개 클래스)")

    txt_features = get_txt_features(classes, TEMPLATES)

    img_tensor = preprocess_img(img).to(device)
    img_features = model.encode_image(img_tensor.unsqueeze(0))
    img_features = F.normalize(img_features, dim=-1)

    logits = (model.logit_scale.exp() * img_features @ txt_features).squeeze()
    probs = F.softmax(logits, dim=0).to("cpu").tolist()

    # 스칼라인 경우 (클래스가 1개)
    if isinstance(probs, float):
        probs = [probs]

    result = {cls: prob for cls, prob in zip(classes, probs)}
    logger.info(f"📊 결과: {result}")
    return result


# ═══════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════
def change_output(choice):
    return gr.Label(num_top_classes=TOP_K, label=RANKS[choice], show_label=True, value=None)


def build_app() -> gr.Blocks:
    """Gradio 앱을 구성합니다."""

    custom_css = """
    .gradio-container {
        max-width: 960px !important;
        margin: 0 auto !important;
    }
    #header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    #header h1 {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    #header p {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    """

    with gr.Blocks(
        title="🐟 BioCLIP-2 종 분류기",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="emerald",
        ),
    ) as app:

        # ── 헤더 ──
        gr.HTML("""
        <div id="header">
            <h1>🐟 BioCLIP-2 종 분류기</h1>
            <p>수중 생물 이미지를 업로드하면 생물 종을 추론합니다 · Top-3 예측 + 신뢰도(%)</p>
        </div>
        """)

        # ══════════════════════════════════════════════
        #  탭 1: Open-Domain 분류
        # ══════════════════════════════════════════════
        with gr.Tab("🌊 Open-Domain (전체 종 분류)"):
            gr.Markdown(
                "**TreeOfLife-200M** 데이터셋의 전체 종을 대상으로 분류합니다. "
                "분류 계급을 선택하고 이미지를 업로드하세요."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    od_img = gr.Image(
                        label="🖼️ 이미지 업로드",
                        type="pil",
                        height=350,
                        sources=["upload", "clipboard"],
                    )
                    od_rank = gr.Dropdown(
                        label="분류 계급 (Taxonomic Rank)",
                        info="세밀할수록(Genus, Species) 더 도전적입니다.",
                        choices=list(RANKS),
                        value="Species",
                        type="index",
                    )
                    od_btn = gr.Button("🔍 분류 시작", variant="primary", size="lg")

                with gr.Column(scale=1):
                    od_output = gr.Label(
                        num_top_classes=TOP_K,
                        label="🏆 Top-3 예측 결과",
                        show_label=True,
                    )

            od_rank.change(fn=change_output, inputs=od_rank, outputs=od_output)
            od_btn.click(
                fn=open_domain_classification,
                inputs=[od_img, od_rank],
                outputs=od_output,
            )

        # ══════════════════════════════════════════════
        #  탭 2: Zero-Shot 분류
        # ══════════════════════════════════════════════
        with gr.Tab("🎯 Zero-Shot (사용자 지정 종)"):
            gr.Markdown(
                "후보 종명을 **줄바꿈**으로 구분하여 입력하고, 이미지를 업로드하세요. "
                "학명(예: *Amphiprion ocellaris*)이나 일반명(예: Clownfish) 모두 사용 가능합니다."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    zs_img = gr.Image(
                        label="🖼️ 이미지 업로드",
                        type="pil",
                        height=350,
                        sources=["upload", "clipboard"],
                    )
                    zs_classes = gr.Textbox(
                        label="후보 종 목록",
                        placeholder="Amphiprion ocellaris (Clownfish)\nParacanthurus hepatus (Blue Tang)\nZebrasoma flavescens (Yellow Tang)",
                        lines=5,
                        info="줄바꿈으로 구분. 학명 + (일반명) 형식 권장.",
                    )
                    zs_btn = gr.Button("🔍 분류 시작", variant="primary", size="lg")

                with gr.Column(scale=1):
                    zs_output = gr.Label(
                        num_top_classes=TOP_K,
                        label="🏆 Top-3 예측 결과",
                        show_label=True,
                    )

            zs_btn.click(
                fn=zero_shot_classification,
                inputs=[zs_img, zs_classes],
                outputs=zs_output,
            )

        # ── 푸터 ──
        gr.Markdown("""
        ---
        <center>
        <small>
        Model: <a href="https://huggingface.co/imageomics/bioclip-2" target="_blank">BioCLIP-2</a> (ViT-L/14) ·
        Data: <a href="https://huggingface.co/datasets/imageomics/TreeOfLife-200M" target="_blank">TreeOfLife-200M</a> ·
        Built with <a href="https://gradio.app" target="_blank">Gradio</a>
        </small>
        </center>
        """)

    return app


# ═══════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = build_app()
    app.queue(max_size=10)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
