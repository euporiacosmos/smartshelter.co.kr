import os
import csv
import cv2
import time
import glob
import albumentations as A
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from pathlib import Path

class SmartShelterPipeline:
    def __init__(self, raw_data_dir=os.path.join('dataset', 'raw'), aug_data_dir=os.path.join('dataset', 'augmented')):
        self.raw_data_dir = raw_data_dir
        self.aug_data_dir = aug_data_dir
        self.model_path = 'yolov8n.pt' # 무료 배포 가능한 Nano 모델
        self.best_model = None

    def augment_data(self, augment_factor=10):
        """
        [STEP 1] 234장의 빈약한 데이터를 2340장으로 뻥튀기하는 데이터 증강 모듈
        """
        print("이미지 증강을 시작합니다... (로드뷰 악조건 시뮬레이션)")
        os.makedirs(self.aug_data_dir, exist_ok=True)
        
        # 로드뷰 특성을 반영한 변형 (왜곡, 밝기 변화, 노이즈 등)
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.4), # 렌즈 왜곡
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # 화질 저하
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        raw_dir = Path(self.raw_data_dir)
        raw_images_dir = Path(self.raw_data_dir) / 'images'
        raw_labels_dir = Path(self.raw_data_dir) / 'labels'
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        
        images = [p for p in raw_images_dir.glob("*") if p.suffix in valid_extensions]
        
        for img_path in images:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filename = img_path.stem
            
            label_path = raw_labels_dir / f"{filename}.txt"
            
            bboxes = []
            class_labels = []
            
            # YOLO 라벨 파일 읽기
            if label_path.exists():
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            # Albumentations 내 bboxes 리스트는 순수 좌표값(float)만 요구함
                            bbox = [float(x) for x in parts[1:]]
                            bboxes.append(bbox)
                            class_labels.append(cls_id)
            
            for i in range(augment_factor):
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{self.aug_data_dir}/{filename}_aug_{i}.jpg", aug_img)
                aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                aug_bboxes = augmented['bboxes']
                
                # 증강된 이미지 저장
                aug_img_name = f"{filename}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(self.aug_data_dir, aug_img_name), aug_img)
                
                # 증강된 바운딩 박스 좌표를 YOLO 포맷으로 변환하여 .txt 파일로 저장
                aug_lbl_name = f"{filename}_aug_{i}.txt"
                with open(os.path.join(self.aug_data_dir, aug_lbl_name), 'w', encoding='utf-8') as f_out:
                    for box, cls in zip(aug_bboxes, class_labels):
                        # box: [x_center, y_center, width, height]
                        f_out.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        
        print(f"증강 완료! {len(images)}장의 이미지가 {len(images) * augment_factor}장으로 늘어났습니다.")

    def train_model(self, yaml_path='shelter_data.yaml', epochs=50):
        """
        [STEP 2] YOLOv8 Nano 모델에 전이 학습(Fine-tuning) 진행
        """
        print("YOLOv8 전이 학습을 시작합니다. GPU를 갈굴 시간입니다.")
        # Pre-trained 모델 로드
        model = YOLO(self.model_path)
        
        # 증강된 데이터셋으로 학습 시작
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='shelter_detector'
        )
        self.best_model = YOLO('runs/detect/shelter_detector/weights/best.pt')
        print("학습 완료! 최고 성능의 가중치가 저장되었습니다.")

    def scan_roadview(self, lat, lng, station_id, station_code, file):
        """
        [STEP 3] 셀레니움으로 카카오맵 로드뷰 진입 -> 스크린샷 -> AI 판독
        """
        kakao_screenshot_path = f"kakao_rv_{station_id}.png"
        naver_screenshot_path = f"naver_rv_{station_id}.png"
        
        if not self.best_model:
            # 학습된 모델이 없다면 저장된 best 모델 로드
            self.best_model = YOLO('runs/detect/shelter_detector/weights/best.pt')

        print(f"[{station_id}] 정류장 로드뷰 탐색 시작... (좌표: {lat}, {lng})")
        
        options = webdriver.ChromeOptions()
        options.add_argument('--window-size=1280,720')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        try:
            # 카카오맵 로드뷰 URL 조합 (카카오맵 API 방식에 따라 URL 구조는 변경될 수 있음)
            # 여기서는 예시로 은평구 일대의 특정 위경도를 타겟팅합니다.
            kakaomap_url = f"https://map.kakao.com/link/roadview/{lat},{lng}"
            driver.get(kakaomap_url)
            time.sleep(4.5) # 로딩 대기 (네트워크 상태에 따라 조절)

            # 로드뷰 화면 캡처
            driver.save_screenshot(kakao_screenshot_path)

            # 캡처한 이미지를 YOLO 모델에 던져서 스마트쉼터 찾기
            results = self.best_model(kakao_screenshot_path)
            
            is_shelter_exist_in_kakaomap = False
            for r in results:
                if len(r.boxes) > 0: # 탐지된 객체가 있다면!
                    is_shelter_exist_in_kakaomap = True
                    break

            navermap_url = f"https://map.naver.com/p?c={lng},{lat},17,0,0,0,dh"
            driver.get(navermap_url)
            time.sleep(3)
            is_shelter_exist_in_navermap = False

            try:
                # 우측 레이아웃 툴바에서 '거리뷰(사람 모양)' 버튼 강제 클릭
                # 네이버 UI 요소의 클래스명을 타겟팅합니다.
                street_view_btn = driver.find_element(By.XPATH, "//*[@id=\"app-layout\"]/div[3]/div/div[3]/div[3]/div[1]/div[2]/button[4]")
                street_view_btn.click()
                time.sleep(1.5)  # 지도에 보라색 거리뷰 노선이 활성화될 때까지 대기
                
                street_fullscreen_view_input = driver.find_element(By.XPATH, "//*[@id=\"root\"]/div/dialog/div/label[2]")
                street_fullscreen_view_save = driver.find_element(By.XPATH, "//*[@id=\"root\"]/div/dialog/button[2]")
                
                street_fullscreen_view_input.click()
                time.sleep(0.5)
                street_fullscreen_view_save.click()
                time.sleep(0.5)

                # 화면 정중앙(우리가 입력한 좌표 지점)을 강제로 쿵 찍어서 거리뷰 내부로 진입시키기
                # 브라우저 창의 정중앙 좌표를 계산하여 ActionChains로 클릭을 날립니다.
                window_size = driver.get_window_size()
                center_x = window_size['width'] / 2
                center_y = window_size['height'] / 2

                try:
                    # 1차 진입 시도
                    actions = ActionChains(driver)
                    actions.move_by_offset(center_x, center_y).click().perform()
                    time.sleep(3)
                except Exception as first_err:
                    print(f"⚠️ 1차 클릭 실패, 재시도합니다... 사유: {first_err}")
                    # 오프셋 초기화를 위한 초기화 액션 후 재시도
                    actions = ActionChains(driver)
                    actions.move_to_element(driver.find_element(By.TAG_NAME, "body")).move_by_offset(center_x, center_y).click().perform()
                    time.sleep(3)
                
                street_close_notice_button = driver.find_element(By.XPATH, "//*[@id=\"app-layout\"]/div[2]/div/div[2]/div/dialog/button")
                street_close_notice_button.click()
                driver.save_screenshot(naver_screenshot_path)
                results = self.best_model(naver_screenshot_path) # [핵심] 캡처한 이미지를 YOLO 모델에 던져서 스마트쉼터 찾기

                for r in results:
                    if len(r.boxes) > 0: # 탐지된 객체가 있다면!
                        is_shelter_exist_in_navermap = True
                        break

            except Exception as e:
                print(f"❌ 네이버 거리뷰 진입 실패: {e}")

            if is_shelter_exist_in_kakaomap or is_shelter_exist_in_navermap:
                print(f"🟢 빙고! [{station_id}] 정류장에서 스마트쉼터가 발견되었습니다.")
                """
                TODO: 현재는 정류장 정보만 기록하지만,
                나중에는 [전국 스마트쉼터](https://smartshelter.co.kr/) 앱의 백엔드 DB로 해당 좌표를 INSERT/UPDATE 하는 로직까지 추가하기 때문에 클래스 이름도 Pipeline이라고 지었다

                백엔드 DB 테이블의 구성은 다음과 같다
                | id | createdAt | imageUrl | author | address | location | enabled |
                """
                file.write(f"{station_code},{station_id},{lat},{lng}\n")
            else:
                print(f"🔴 [{station_id}] 정류장에는 일반 정류장만 존재합니다.")

        finally:
            driver.quit()
            for path in (kakao_screenshot_path, naver_screenshot_path):
                if os.path.exists(path):
                    os.remove(path) # 임시 파일 삭제

if __name__ == "__main__":
    pipeline = SmartShelterPipeline()
    startup_time = time.time()
    
    # 1. 초기 1회만 실행: 데이터 증강
    pipeline.augment_data(augment_factor=10)
    augment_data_spend_time = time.time()-startup_time
    print(f"데이터 증강 소요시간: {augment_data_spend_time:.3f}초")

    # 2. 초기 1회만 실행: YOLO 학습 (이 과정에서 GPU가 사용됨)
    pipeline.train_model(epochs=50)
    train_model_spend_time = time.time()-startup_time
    print(f"YOLO 학습 완료\n총 소요시간: {train_model_spend_time:.3f}초")
    
    csv_stations = open("국토교통부_전국 버스정류장 위치정보_20251031.csv", "r", encoding="cp949")
    reader = csv.reader(csv_stations)
    next(reader) # 헤더 스킵
    f_shelters_location = open("data.txt", "w")

    for station in reader:
        pipeline.scan_roadview(lat=station[2], lng=station[3], station_id=station[1], station_code=station[0], file=f_shelters_location)
    scan_spend_time = time.time()-startup_time
    print(f"스캔 완료\n총 소요시간: {scan_spend_time:.3f}초")

    print(f"\n데이터 증강 소요시간: {augment_data_spend_time:.2f}초\nYOLO 학습 소요시간: {train_model_spend_time-augment_data_spend_time:.2f}초\n스캔 소요시간: {scan_spend_time-train_model_spend_time:.2f}초")
    csv_stations.close()
    f_shelters_location.close()