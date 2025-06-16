import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import random

def create_kmeans_mask(image: np.ndarray, k: int = 4) -> np.ndarray:
    """단일 이미지(Numpy 배열)를 입력받아 K-means 기반의 학습용 마스크를 생성합니다."""
    pixel_data = image.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(pixel_data)
    brightness = np.mean(kmeans.cluster_centers_, axis=1)
    sorted_indices = np.argsort(brightness)
    normalized_labels = np.zeros_like(kmeans.labels_)
    for new_label, original_label_idx in enumerate(sorted_indices):
        normalized_labels[kmeans.labels_ == original_label_idx] = new_label
    final_mask = normalized_labels.reshape(image.shape[0], image.shape[1])
    return final_mask.astype(np.uint8)

def batch_create_and_verify(
    base_img_dir: str, 
    list_file_path: str, 
    output_data_dir: str, 
    num_verification_samples: int,
    k_value: int):
    """
    이미지 목록을 기반으로 마스크를 생성하고, 결과물 중 일부를 검증합니다.
    """
    # --- 1단계: 마스크 생성 ---
    print("--- 단계 1: 학습용 마스크 생성 시작 ---")
    os.makedirs(output_data_dir, exist_ok=True)

    try:
        with open(list_file_path, 'r') as f:
            image_entries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"오류: 이미지 목록 파일 '{list_file_path}'를 찾을 수 없습니다.")
        return

    for entry in tqdm(image_entries, desc="마스크 생성 중"):
        image_path = os.path.join(base_img_dir, entry)
        try:
            original_image = cv2.imread(image_path)
            if original_image is None: continue
            
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            mask_data = create_kmeans_mask(original_image_rgb, k=k_value)
            
            class_name, filename = os.path.split(entry)
            filename_without_ext = os.path.splitext(filename)[0]
            output_class_dir = os.path.join(output_data_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            data_output_path = os.path.join(output_class_dir, f"{filename_without_ext}_mask.png")
            cv2.imwrite(data_output_path, mask_data)
        except Exception as e:
            print(f"오류 발생: '{image_path}' 처리 중 문제 발생 - {e}")
    print("--- 단계 1: 마스크 생성 완료 ---")


    # --- 2단계: 생성된 마스크 자동 검증 ---
    print("\n--- 단계 2: 생성된 마스크 자동 검증 시작 ---")
    verification_dir = os.path.join(os.path.dirname(output_data_dir), "verification_samples")
    os.makedirs(verification_dir, exist_ok=True)
    print(f"검증용 시각화 샘플은 '{verification_dir}' 폴더에 저장됩니다.")
    
    all_created_masks = glob.glob(os.path.join(output_data_dir, '**', '*.png'), recursive=True)
    if not all_created_masks:
        print("검증할 마스크가 없습니다. 생성이 제대로 되었는지 확인하세요.")
        return
        
    # 생성된 마스크 중 일부를 무작위로 선택
    samples_to_verify = random.sample(all_created_masks, min(num_verification_samples, len(all_created_masks)))

    for i, mask_path in enumerate(samples_to_verify):
        mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask_data)

        print(f"\n[샘플 {i+1} 검증] 파일: {os.path.basename(mask_path)}")
        print(f"  > 데이터 값 확인: {unique_vals}")
        
        if set(unique_vals).issubset(set(range(k_value))):
             print(f"  > ✅ 데이터 정상: 0부터 {k_value-1}까지의 값으로 구성되어 있습니다.")
        else:
             print(f"  > ⚠️ 데이터 경고: 예상치 못한 값이 포함되어 있습니다.")

        # 시각화용 이미지 생성 및 저장
        visual_mask = (mask_data * (255.0 / (k_value - 1))).astype(np.uint8)
        visual_path = os.path.join(verification_dir, f"visual_{os.path.basename(mask_path)}")
        cv2.imwrite(visual_path, visual_mask)
    
    print("\n--- 단계 2: 검증 완료 ---")
    print(f"'{output_data_dir}' 폴더에 있는 데이터로 학습을 시작하시면 됩니다.")

# ===== 메인 실행 부분 =====
if __name__ == '__main__':
    # --- 설정 ---
    BASE_IMG_DIR = '/home/users/astar/ares/yoosehwa/scratch/pathology'
    LIST_FILE_PATH = './good_images.txt'
    # AI 학습에 사용할 데이터가 저장될 폴더
    OUTPUT_DATA_DIR = './generated_masks_data'
    # K-means 클러스터 개수
    K_VALUE = 4
    # 생성 후 검증할 샘플의 수
    NUM_VERIFICATION_SAMPLES = 5

    batch_create_and_verify(
        base_img_dir=BASE_IMG_DIR,
        list_file_path=LIST_FILE_PATH,
        output_data_dir=OUTPUT_DATA_DIR,
        num_verification_samples=NUM_VERIFICATION_SAMPLES,
        k_value=K_VALUE
    )