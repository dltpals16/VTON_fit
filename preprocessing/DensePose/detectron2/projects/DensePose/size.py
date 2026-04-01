# import os
# import glob
# import cv2

# input_dir = "/mnt/aix23904/아르포아/virtual_tryon/008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)/01-1.정식개방데이터/Training/01.원천데이터/TS_상품_상의_blouse"

# img_patterns = ["*_wear*.jpg", "*_wear*.jpeg", "*_wear*.png", "*_wear*.JPG", "*_wear*.JPEG", "*_wear*.PNG"]
# image_paths = []
# for pat in img_patterns:
#     image_paths.extend(glob.glob(os.path.join(input_dir, pat)))
# image_paths = sorted(image_paths)

# print(f"총 {len(image_paths)}개의 '_wear' 이미지가 있습니다.")

# # 임계값(예시): 너무 큰 이미지와 너무 작은 이미지 기준
# MAX_SIZE = 3000
# MIN_SIZE = 128

# for img_path in image_paths:
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[!] 읽기 실패: {img_path}")
#         continue
#     h, w = img.shape[:2]
#     msg = ""
#     if max(h, w) > MAX_SIZE:
#         msg = f"  <-- [너무 큼!]"
#     if min(h, w) < MIN_SIZE:
#         msg = f"  <-- [너무 작음!]"
#     print(f"{os.path.basename(img_path)}\t{w} x {h}{msg}")


import torch; print(torch.__version__); print(torch.version.cuda)
import detectron2; print(detectron2.__version__)
import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))
import torch
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.is_available())
