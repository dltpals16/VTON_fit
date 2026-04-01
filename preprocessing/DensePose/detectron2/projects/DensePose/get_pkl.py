# import os
# import pickle
# import numpy as np
# import cv2

# import torch
# # 경로 설정
# pkl_path = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/results2.pkl"
# output_dir = "./mask"
# os.makedirs(output_dir, exist_ok=True)
# def save_uv_heatmap(channel_map, save_path):
#     norm = np.clip(channel_map * 255, 0, 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
#     cv2.imwrite(save_path, heatmap)

# # ===== dump 결과 로드 =====
# data = torch.load(pkl_path)

# img_id, instance_id = 0, 0
# entry = data[img_id]
# fname = os.path.splitext(os.path.basename(entry["file_name"]))[0]

# densepose_list = entry.get("pred_densepose")
# if densepose_list is None or len(densepose_list) <= instance_id:
#     raise ValueError("DensePose 결과가 없습니다.")

# dp = densepose_list[instance_id]

# # ===== dp_segm 마스크 저장 =====
# labels = dp.labels.cpu().numpy().astype(np.uint8)
# cv2.imwrite(os.path.join(output_dir, f"{fname}_dp_segm.png"), labels)
# colored = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
# cv2.imwrite(os.path.join(output_dir, f"{fname}_dp_segm_color.png"), colored)

# # ===== U/V 맵 =====
# uv = dp.uv  # shape: [2, H, W]
# u_map = uv[0].cpu().numpy()
# v_map = uv[1].cpu().numpy()

# # ===== 마스크 영역 외 제거 (선택) =====
# mask = labels > 0
# u_map[~mask] = 0
# v_map[~mask] = 0

# # ===== 저장 =====
# save_uv_heatmap(u_map, os.path.join(output_dir, f"{fname}_u_map.png"))
# save_uv_heatmap(v_map, os.path.join(output_dir, f"{fname}_v_map.png"))

# uv_rgb = np.zeros((u_map.shape[0], u_map.shape[1], 3), dtype=np.uint8)
# uv_rgb[..., 0] = (u_map * 255).astype(np.uint8)
# uv_rgb[..., 1] = (v_map * 255).astype(np.uint8)
# cv2.imwrite(os.path.join(output_dir, f"{fname}_uv_combined.png"), uv_rgb)

# print(f"[✔] 저장 완료: {output_dir}")

###########################################################################

# # .pkl 결과 로드# === 시각화 유틸 ===
# def save_uv_heatmap(channel_map, save_path):
#     norm = np.clip(channel_map * 255, 0, 255).astype(np.uint8)
#     heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
#     cv2.imwrite(save_path, heatmap)

# # === 결과 로드 ===
# results = torch.load(pkl_path)

# # === 원하는 이미지 및 인스턴스 선택 ===
# img_id = 0         # 첫 번째 이미지
# instance_id = 0    # 첫 번째 인스턴스

# entry = results[img_id]
# fname = os.path.splitext(os.path.basename(entry["file_name"]))[0]

# densepose_list = entry.get("pred_densepose")
# if densepose_list is None or len(densepose_list) <= instance_id:
#     raise ValueError("DensePose 결과가 없습니다.")

# dp = densepose_list[instance_id]

# # === dp_segm ===
# labels = dp.labels.cpu().numpy().astype(np.uint8)
# cv2.imwrite(os.path.join(output_dir, f"{fname}_dp_segm.png"), labels)

# # === U/V 맵 ===
# uv = dp.uv  # shape: [2, H, W]
# u_map = uv[0].cpu().numpy()
# v_map = uv[1].cpu().numpy()

# # === 마스크 적용 (선택)
# mask = labels > 0
# u_map[~mask] = 0
# v_map[~mask] = 0

# # === 저장
# save_uv_heatmap(u_map, os.path.join(output_dir, f"{fname}_u_map.png"))
# save_uv_heatmap(v_map, os.path.join(output_dir, f"{fname}_v_map.png"))

# # === U+V 혼합 RGB 이미지
# uv_rgb = np.zeros((u_map.shape[0], u_map.shape[1], 3), dtype=np.uint8)
# uv_rgb[..., 0] = (u_map * 255).astype(np.uint8)  # U → R
# uv_rgb[..., 1] = (v_map * 255).astype(np.uint8)  # V → G
# # B 채널은 0
# cv2.imwrite(os.path.join(output_dir, f"{fname}_uv_combined.png"), uv_rgb)

# print(f"[✔] 저장 완료: {output_dir}")

# print('results', results[0]['pred_densepose'][0]['uv'])
# print('len', len(results))
# for idx, entry in enumerate(results):
#     # print('here')
#     instances = entry.get("instances", None)
#     if instances is None:
#         continue

#     # pred_densepose에서 segm (dp_segm) 추출
#     densepose_data = instances.get("pred_densepose", None)
#     if not densepose_data:
#         continue

#     segm = densepose_data.get("segm", None)
#     print('segm',segm)
#     if segm is None:
#         continue

#     # 마스크 저장
#     mask = np.array(segm, dtype=np.uint8)  # 0~24: body part class
#     save_path = os.path.join(output_dir, f"mask_{idx:03d}.png")
#     cv2.imwrite(save_path, mask)
# #     print(f"[✔] 저장 완료: {save_path}")
# for idx, entry in enumerate(results):
#     fname = os.path.splitext(os.path.basename(entry["file_name"]))[0]
#     densepose_list = entry.get("pred_densepose")
#     boxes = entry.get("pred_boxes_XYXY")

#     if densepose_list is None or boxes is None:
#         continue

#     for i, dp in enumerate(densepose_list):
#         if not hasattr(dp, "labels"):
#             print(f"[SKIP] instance {i}: labels 없음")
#             continue

#         segm = dp.labels.cpu().numpy().astype(np.uint8)  # ← 이게 dp_segm!

#         save_path = os.path.join(output_dir, f"{fname}_instance{i}.png")
#         cv2.imwrite(save_path, segm)
#         print(f"[✔] 저장 완료: {save_path}")

##############################################################################

# import os
# import torch
# import numpy as np
# import cv2

# # ====== 설정 ======
# input_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/pkl_output/blouse"
# output_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/mask/blouse"
# os.makedirs(output_dir, exist_ok=True)

# # ====== .pkl 파일 리스트업 ======
# pkl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])
# total = len(pkl_files)

# print(f"총 {total}개의 .pkl 파일이 있습니다.")

# done = 0

# # ====== 전체 파일 반복 ======
# for idx, file_name in enumerate(pkl_files, 1):
#     pkl_path = os.path.join(input_dir, file_name)
#     out_color_png = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")

#     # ===== dump 결과 로드 =====
#     data = torch.load(pkl_path)
#     entry = data[0]
#     densepose_list = entry.get("pred_densepose")
#     if densepose_list is None or len(densepose_list) == 0:
#         print(f"[{idx}/{total}] [!] DensePose 결과가 없습니다: {file_name}")
#         continue
#     dp = densepose_list[0]

#     # ===== 컬러 마스크 저장 =====
#     labels = dp.labels.cpu().numpy().astype(np.uint8)
#     colored = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
#     cv2.imwrite(out_color_png, colored)

#     done += 1
#     print(f"[{idx}/{total}] [✔] 저장 완료: {out_color_png}")

# print(f"\n[전체 완료] 총 {done}개의 마스크가 {output_dir}에 저장됨.")


import os
import torch
import numpy as np
import cv2
from multiprocessing import Pool
from tqdm import tqdm

# ====== 설정 ======
input_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/pkl_output/validation/t-shirt"
output_dir = "/mnt/aix23904/아르포아/virtual_tryon/DensePose/detectron2/projects/DensePose/mask/validation"
os.makedirs(output_dir, exist_ok=True)

# ====== .pkl 파일 리스트업 ======
pkl_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pkl")])
total = len(pkl_files)

print(f"총 {total}개의 .pkl 파일이 있습니다.")

def process_file(args):
    idx, file_name = args
    pkl_path = os.path.join(input_dir, file_name)
    out_color_png = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".png")
    try:
        data = torch.load(pkl_path, map_location='cpu')
        entry = data[0]
        densepose_list = entry.get("pred_densepose")
        if densepose_list is None or len(densepose_list) == 0:
            return (file_name, False, "[!] DensePose 결과가 없습니다")
        dp = densepose_list[0]
        labels = dp.labels.cpu().numpy().astype(np.uint8)
        colored = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(out_color_png, colored)
        return (file_name, True, "[✔] 저장 완료")
    except Exception as e:
        return (file_name, False, f"[!] 에러: {str(e)}")

if __name__ == "__main__":
    args_list = list(enumerate(pkl_files, 1))
    done = 0

    with Pool(processes=16) as pool:
        # tqdm 진행바로 퍼센트 표시
        for result in tqdm(pool.imap_unordered(process_file, args_list), total=total, ncols=80):
            file_name, success, msg = result
            if success:
                done += 1
            # 진행 도중에도 필요하면 msg/log 찍기
            # print(f"{file_name}: {msg}")

    print(f"\n[전체 완료] 총 {done}개의 마스크가 {output_dir}에 저장됨.")
