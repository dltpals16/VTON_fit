import os

file_path = '/mnt/aix23904/아르포아/virtual_tryon/008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)/01-1.정식개방데이터/Validation/01.원천데이터/VS_상품_상의_t-shirt/01_sou_027610_138048_wear_02top_02t-shirt_woman.jpg'

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} 파일이 성공적으로 삭제되었습니다.")
else:
    print(f"{file_path} 파일이 존재하지 않습니다.")
