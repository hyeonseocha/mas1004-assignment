import os
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ResNet18 정규화에 사용되는 기본 상수값
RESIZE = 256
CROP = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def prepare_image(image: Image.Image) -> np.ndarray:
    """
    1. RGB 변환
    2. 짧은 축 기준 RESIZE(256) 크기로 리사이즈 (Bilinear)
    3. CenterCrop (224x224)
    4. ToTensor (0~1 범위 변환 및 Channel-first (3, 224, 224) 변경)
    5. Normalize (MEAN, STD 적용)
    6. numpy array (float32)로 변환하여 반환
    """
    # Step 1: RGB 변환
    img_rgb = image.convert("RGB")
    
    # Step 2~5: torchvision transforms 이용
    transform = T.Compose([
        T.Resize(RESIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(CROP),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    
    tensor_img = transform(img_rgb)
    
    # Step 6: numpy float32 배열로 반환
    return tensor_img.numpy().astype(np.float32)


def load_folder(root):
    root_path = Path(root)
    
    # 클래스 폴더 탐색 및 정렬
    class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_folders]
    
    X_list = []
    y_list = []
    paths = []
    
    for class_idx, class_dir in enumerate(class_folders):
        # 파일 순서 고정을 위해 정렬
        file_paths = sorted(class_dir.iterdir())
        
        for file_path in file_paths:
            # 확장자 검사 (대소문자 구분 없음)
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            
            # 깨진 이미지 예외 처리 및 이미지 로드
            try:
                with Image.open(file_path) as img:
                    img.verify() # 이미지 손상 여부 검증
                
                with Image.open(file_path) as img:
                    prepared_x = prepare_image(img)
                    
                X_list.append(prepared_x)
                y_list.append(class_idx)
                paths.append(file_path)
            except Exception:
                # 손상되었거나 열 수 없는 파일은 건너뜀
                continue

    if not X_list:
        X = np.empty((0, 3, CROP, CROP), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        
    return X, y, class_names, paths

import numpy as np


def split_train_test(X, y, paths, test_ratio=0.2, seed=42):
    """X, y, paths 데이터를 train과 test 세트로 분할합니다."""
    np.random.seed(seed)
    n_samples = len(X)

    if n_samples == 0:
        return X, y, paths, X, y, paths

    # 계층적 분할 (Stratified Split: 각 클래스 비율 유지)
    unique_classes = np.unique(y)
    train_indices, test_indices = [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_test = int(np.ceil(len(cls_indices) * test_ratio))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    paths_list = list(paths)
    paths_train = [paths_list[i] for i in train_indices]
    paths_test = [paths_list[i] for i in test_indices]

    return X_train, y_train, paths_train, X_test, y_test, paths_test