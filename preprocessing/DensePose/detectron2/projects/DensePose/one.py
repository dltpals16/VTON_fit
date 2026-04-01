# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import torch
# torch.cuda.empty_cache()
# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.engine.defaults import DefaultPredictor
# from densepose import add_densepose_config
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # ====== 경로 설정 ======
# img_path = "/mnt/aix23904/아르포아/virtual_tryon/008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)/01-1.정식개방데이터/Validation/01.원천데이터/VS_상품_상의_blouse/01_sou_015180_075898_wear_02top_01blouse_woman.jpg"
# output_dir = "pkl_output/validation/blouse"
# os.makedirs(output_dir, exist_ok=True)

# config_file = "configs/densepose_rcnn_R_50_FPN_s1x.yaml"    
# model_weights = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# # ====== DensePose 모델 로드 ======
# cfg = get_cfg()
# add_densepose_config(cfg)
# cfg.merge_from_file(config_file)
# cfg.MODEL.WEIGHTS = model_weights
# cfg.freeze()
# predictor = DefaultPredictor(cfg)

# print(f"처리중: {img_path}")
# img = read_image(img_path, format="BGR")
# with torch.no_grad():
#     outputs = predictor(img)["instances"]

# result = {"file_name": img_path}
# if outputs.has("scores"):
#     result["scores"] = outputs.get("scores").cpu()
# if outputs.has("pred_boxes"):
#     result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
#     if outputs.has("pred_densepose"):
#         from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
#         from densepose.vis.extractor import DensePoseResultExtractor, DensePoseOutputsExtractor

#         if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
#             extractor = DensePoseResultExtractor()
#         elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
#             extractor = DensePoseOutputsExtractor()
#         else:
#             extractor = None
#         if extractor is not None:
#             result["pred_densepose"] = extractor(outputs)[0]

# # ====== 결과 저장 ======
# base_name = os.path.splitext(os.path.basename(img_path))[0]
# out_pkl = os.path.join(output_dir, f"{base_name}.pkl")
# torch.save([result], out_pkl)  # 리스트 형태로 저장
# print(f"저장됨: {out_pkl}")
# print("처리 완료!")

import os
import torch
import numpy as np
import cv2

# ====== 경로 및 파일 설정 ======
input_pkl = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/pkl_output/validation/blouse/01_sou_015180_075898_wear_02top_01blouse_woman.pkl"
output_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/mask/validation"
os.makedirs(output_dir, exist_ok=True)

file_name = os.path.basename(input_pkl)
out_color_png = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")

try:
    print(f"로드 중: {input_pkl}")
    data = torch.load(input_pkl, map_location='cpu')
    entry = data[0]
    densepose_list = entry.get("pred_densepose")
    if densepose_list is None or len(densepose_list) == 0:
        print("[!] DensePose 결과가 없습니다 (빈 파일)")
    else:
        dp = densepose_list[0]
        labels = dp.labels.cpu().numpy().astype(np.uint8)
        colored = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(out_color_png, colored)
        print(f"[✔] 저장 완료: {out_color_png}")
except Exception as e:
    print(f"[!] 에러 발생: {e}")
