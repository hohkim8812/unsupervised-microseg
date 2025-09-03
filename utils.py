import os
from datetime import datetime

def make_filename(out_dir, img_path):
    date = datetime.now().strftime("%y%m%d")
    name = os.path.splitext(os.path.basename(img_path))[0]  # 입력 이미지 이름
    folder = os.path.join(out_dir, name)  # 이미지별 폴더 생성
    os.makedirs(folder, exist_ok=True)

    i = 0
    while os.path.exists(f"{folder}/{date}_{i}.png"):
        i += 1
    return f"{folder}/{date}_{i}"