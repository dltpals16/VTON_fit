import os

files = os.listdir("/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/pkl_output/validation/t-shirt")
whole = os.listdir("/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/mask/validation")
wear = [file for file in whole if "t-shirt" in file]
# images = os.listdir("/home/aix23904/virtual_tryon/LIP_JPPNet/datasets/t-shirt/images")

print(len(files), len(wear))


# 경로 설정
pkl_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/pkl_output/validation/t-shirt"
mask_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/mask/validation"

# .pkl 파일 목록 (확장자 제거)
pkl_files = [os.path.splitext(f)[0] for f in os.listdir(pkl_dir) if f.endswith('.pkl')]

# mask_dir에서 "blouse"가 포함된 이미지 파일만 추출 (확장자 제거)
blouse_images = [
    os.path.splitext(f)[0] for f in os.listdir(mask_dir)
    if ("t-shirt" in f.lower()) and (f.endswith('.png') or f.endswith('.jpg'))
]

# .pkl에는 있는데, 이미지로는 없는 파일 찾기
missing_images = [f for f in pkl_files if f not in blouse_images]

# 결과 출력
print(f"[.pkl은 있는데 이미지가 없는 blouse 관련 파일 수]: {len(missing_images)}")
for fname in missing_images:
    print(fname + ".pkl")

