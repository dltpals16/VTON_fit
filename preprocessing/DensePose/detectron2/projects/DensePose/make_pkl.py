import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import glob
import torch
torch.cuda.empty_cache()
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config


# ====== 경로 설정 ======
input_dir = "/mnt/aix23904/아르포아/virtual_tryon/008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)/01-1.정식개방데이터/Validation/01.원천데이터/VS_상품_상의_vest"
output_dir = "pkl_output/validation/vest"
os.makedirs(output_dir, exist_ok=True)

config_file = "configs/densepose_rcnn_R_50_FPN_s1x.yaml"  # 수정 필요시 경로 지정
model_weights = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# ====== DensePose 모델 로드 ======
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = model_weights
cfg.freeze()
predictor = DefaultPredictor(cfg)

# ====== '_wear'가 포함된 이미지 파일만 리스트업 (jpg, jpeg, png) ======
img_patterns = ["*_wear*.jpg", "*_wear*.jpeg", "*_wear*.png", "*_wear*.JPG", "*_wear*.JPEG", "*_wear*.PNG"]
image_paths = []
for pat in img_patterns:
    image_paths.extend(glob.glob(os.path.join(input_dir, pat)))
image_paths = sorted(image_paths)

print(f"총 {len(image_paths)}개의 '_wear' 이미지가 발견되었습니다.")

for img_path in image_paths:
    print(f"처리중: {img_path}")
    img = read_image(img_path, format="BGR")
    with torch.no_grad():
        outputs = predictor(img)["instances"]

    result = {"file_name": img_path}
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
            from densepose.vis.extractor import DensePoseResultExtractor, DensePoseOutputsExtractor

            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()
            else:
                extractor = None
            if extractor is not None:
                result["pred_densepose"] = extractor(outputs)[0]

    # ====== 결과 저장 ======
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_pkl = os.path.join(output_dir, f"{base_name}.pkl")
    torch.save([result], out_pkl)  # DumpAction과 동일하게 리스트 형태로 저장
    print(f"저장됨: {out_pkl}")

print("모든 이미지 처리 완료!")
