# 📥 모델 가중치(Weights) 다운로드 및 배치 가이드

이 프로젝트(Divery Vision Pipeline)는 용량이 큰 여러 AI 모델 가중치(`.pt`, `.ckpt` 등)를 사용합니다.
GitHub/GitLab 등 코드 저장소의 용량 제한 및 버전 관리 효율성을 위해, 가중치 파일들은 `git` 추적에서 제외(`.gitignore`)되어 배포됩니다.

따라서 코드를 처음 클론하신 팀원분들께서는 아래 안내에 따라 가중치 파일을 직접 다운로드하여 지정된 경로에 배치해 주셔야 파이프라인이 정상 작동합니다.

---

## 📂 1. 필요한 모델 가중치 목록 및 배치 경로

다운로드하신 가중치 파일들은 반드시 아래 표에 명시된 **정확한 경로**에 배치해 주세요. 파일명도 변경하시면 안 됩니다.

| 파일명                 | 배치해야 할 경로 (프로젝트 루트 기준) | 용량 (약) | 역할 및 설명                                                                                    |
| :--------------------- | :------------------------------------ | :-------- | :---------------------------------------------------------------------------------------------- |
| `cfd-yolov12x-1.00.pt` | `CFD_fishial/`                        | 114 MB    | **CFD (Custom Fish Detector)**<br>수중 생물 탐지를 위한 메인 YOLOv12x 모델입니다.               |
| `yolo11n.pt`           | `CFD_fishial/`                        | 5.4 MB    | **YOLOv11 Nano**<br>빠른 기본 객체 탐지를 위한 경량 모델입니다.                                 |
| `model.ckpt`           | `CFD_fishial/fishial_model/`          | 331 MB    | **(Deprecated) Fishial.AI 체급/종 분류 모델**<br>어종의 특징을 추출하고 분류하는 구버전 핵심 모델 체크포인트입니다. |
| `database.pt`          | `CFD_fishial/fishial_model/`          | 137 MB    | **(Deprecated) Fishial.AI kNN 임베딩 DB**<br>미리 계산된 755종의 어종 임베딩 데이터베이스입니다.             |

> 💡 **최신 BioCLIP-2 관련 안내**<br>
> 최신 버전의 파이프라인은 분류 단계에서 `BioCLIP-2 (ViT-L-14 / ViT-H-14)` 모델을 사용합니다. 해당 모델 가중치 파일들은 로컬에 직접 배치할 필요 없이, 파이프라인 최초 실행 시 **Hugging Face Hub를 통해 사용자 환경의 캐시 폴더 (`~/.cache/huggingface/hub`) 로 자동 다운로드** 됩니다. (초기 실행 시 인터넷 연결 필수)

---

## 🔗 2. 다운로드 링크 (Google Drive / NAS)

> **팀장/관리자 필수 작성란**: <br>
> 팀원들이 모델을 다운로드할 수 있는 공유 링크나 서버 접속 정보 등을 아래에 기입해 주세요.

- **Google Drive 통합 폴더**: `https://drive.google.com/drive/folders/1aNvrQEgMkFUmkXU-SRXuY-RfWLroaxfF?usp=sharing`
<!-- - **사내 NAS 접근 경로**: `smb://192.168.x.x/models/` (예시) -->

---

## ✅ 3. 검증 방법

파일 배치를 모두 완료하셨다면, 다음 명령어를 통해 파일들이 올바른 위치에 있는지 간단히 확인해 볼 수 있습니다. (Linux / Mac / Git Bash 환경)

```bash
ls -lh CFD_fishial/*.pt
ls -lh CFD_fishial/fishial_model/
```

터미널 상에 `cfd-yolov12x-1.00.pt`, `model.ckpt` 등의 파일과 용량이 정상적으로 출력된다면 환경 구성이 완료된 것입니다. 이제 `CFD_fishial/web_app.py` 등을 실행하여 파이프라인을 테스트해 보세요!
