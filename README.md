# VTON_fit: Virtual Try-On for Custom Fit Dataset

DCI-VTON(Disentangled Clothing Images Virtual Try-On) 기반 가상 피팅 파이프라인.
자체 수집한 의류 데이터셋(Fit Dataset)에 맞게 전처리 및 학습/추론 코드를 구성하였습니다.

## Project Structure

```
VTON_fit/
├── main.py                  # 학습 메인 스크립트 (PyTorch Lightning)
├── test.py                  # 추론 스크립트 (DDIM/PLMS Sampler)
├── train.sh                 # 학습 실행 스크립트
├── test.sh                  # 추론 실행 스크립트
├── environment.yaml         # Conda 환경 설정
├── configs/
│   ├── viton512.yaml        # 512x512 학습/추론 config
│   └── viton512_v2.yaml     # v2 config
├── ldm/                     # Latent Diffusion Model 코어
├── src/
│   ├── clip/                # CLIP 모듈
│   └── taming-transformers/ # VQGAN 모듈
├── warp/                    # Warping 네트워크 (TPS 기반)
├── PF-AFN/                  # Parser-Free Appearance Flow Network
│   ├── PF-AFN_test/         # PF-AFN 추론
│   └── PF-AFN_train/        # PF-AFN 학습
└── preprocessing/           # 데이터 전처리 코드
    ├── Self-Correction-Human-Parsing/  # Human Parsing (세그멘테이션)
    └── DensePose/                      # DensePose 추출 (Detectron2)
```

## Pipeline Overview

```
Input Image + Clothing Image
        │
        ├── 1. Human Parsing (Self-Correction-Human-Parsing)
        │   └── 사람 이미지에서 신체 부위별 세그멘테이션 맵 생성
        │
        ├── 2. DensePose (Detectron2)
        │   └── 사람 이미지에서 UV 좌표 기반 밀집 자세 추정
        │
        ├── 3. Warping (PF-AFN)
        │   └── 의류 이미지를 사람 체형에 맞게 변형
        │
        └── 4. Try-On Generation (DCI-VTON, Latent Diffusion)
            └── Warped 의류 + 사람 이미지 → 가상 피팅 결과 생성
```

## Setup

### Environment
```bash
conda env create -f environment.yaml
conda activate dci-vton
```

### Prerequisites
- Python 3.8
- PyTorch 1.11.0
- CUDA 11.3
- Detectron2 (DensePose용)

## Data Preprocessing

### 1. Human Parsing
```bash
cd preprocessing/Self-Correction-Human-Parsing
python simple_extractor.py --dataset lip --model-restore <checkpoint_path> --input-dir <image_dir> --output-dir <output_dir>
```

### 2. DensePose
```bash
cd preprocessing/DensePose/detectron2/projects/DensePose
python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml <model_weights> <input_dir> --output <output_dir>
```

## Training

### PF-AFN (Warping Network)
```bash
cd PF-AFN/PF-AFN_train
bash train_VITON.sh
```

### DCI-VTON (Diffusion Model)
```bash
bash train.sh
# python -u main.py --logdir models/dci-vton \
#   --pretrained_model checkpoints/model.ckpt \
#   --base configs/viton512.yaml \
#   --scale_lr False
```

## Inference

```bash
bash test.sh
# python test.py --plms --gpu_id 0 \
#   --ddim_steps 100 \
#   --outdir results/viton \
#   --config configs/viton512.yaml \
#   --dataroot <data_path> \
#   --ckpt <checkpoint_path> \
#   --n_samples 8 \
#   --seed 23 \
#   --scale 1 \
#   --H 512 --W 512 \
#   --unpaired
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ddim_steps` | 100 | DDIM sampling steps |
| `--n_samples` | 8 | Batch size for inference |
| `--scale` | 1 | Guidance scale |
| `--H / --W` | 512 / 512 | Output resolution |
| `--unpaired` | flag | Unpaired try-on mode |
| `--plms` | flag | Use PLMS sampler (faster) |

## Model Architecture

- **Backbone**: Latent Diffusion Model (Stable Diffusion 기반)
- **Warping**: TPS(Thin Plate Spline) + Appearance Flow (PF-AFN)
- **Conditioning**: CLIP image encoder + Human parsing + DensePose
- **Resolution**: 512 x 512

## References

- [DCI-VTON: Disentangled Clothing Images Virtual Try-On](https://github.com/bcmi/DCI-VTON-Virtual-Try-On)
- [PF-AFN: Parser-Free Virtual Try-On via Distilling Appearance Flows](https://github.com/geyuying/PF-AFN)
- [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)
- [DensePose (Detectron2)](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
