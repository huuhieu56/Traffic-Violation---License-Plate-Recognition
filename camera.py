import cv2
import torch
import time
import utils.helper as helper
import csv
import numpy as np
from datetime import datetime
import pathlib
import subprocess
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# load model
yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/lp_vn_det_yolov5n.pt', force_reload=True)
# yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/lp_vn_ocr_yolov5s_final.pt', force_reload=True)
# yolo_license_plate.conf = 0.8

prev_frame_time = 0
new_frame_time = 0

# CSV file paths
license_plate_file = "violation_data/license_plate.csv"
valid_plate_file = "violation_data/valid_plate.csv"

# Video stream setup
vid = cv2.VideoCapture("./test_video/tra.mp4")
# vid = cv2.VideoCapture(0)

start_time = time.time()
plate_detection_duration = 30  # 30 seconds for detection
cycle_duration = 60

prev_frame_time = 0
new_frame_time = 0

points = np.array([[1000, 700], [1700, 700], [1900,1078], [800, 1078]], np.int32)
points = points.reshape((-1, 1, 2))

flag = 0

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    ret, frame = vid.read()
    if not ret:
        break

    if elapsed_time % cycle_duration < plate_detection_duration:
        # License plate detection (First 30 seconds)
        temp_frame = frame
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        masked_image = cv2.bitwise_and(frame, mask)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 215, 255), thickness=2)  
        plates = yolo_LP_detect(masked_image, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        for plate in list_plates:
            flag = 0
            x1 = int(plate[0]) 
            y1 = int(plate[1]) 
            x2 = int(plate[2]) 
            y2 = int(plate[3]) 
            crop_img = helper.crop_expanded_plate(plate_xyxy=(x1, y1, x2, y2), img=temp_frame, expand_ratio=0.15)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color = (0,0,225), thickness = 1)
            crop_img = helper.preprocess_image(crop_img)
            cv2.imwrite("crop_plate.jpg", crop_img)
                  
            lp = helper.read_plate_ppocr(crop_img)
            if lp != "unknown":
                list_read_plates.add(lp)
                helper.draw_text(img=frame, text=lp, pos=(x1, y1),  font_thickness=2, text_color=(0,255,0))
                        
        
            # Save detected plates to CSV
            with open(license_plate_file, mode='a', newline='') as file:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                name = lp + str(current_time)

                writer = csv.writer(file)
                for lp in list_read_plates:
                    writer.writerow([lp, dt_string, name])
                    cv2.imwrite("violation_data/img/"+name+".jpg", frame)
                    flag = 1

    else:
        # Data processing (Next 30 seconds)
        # Read from license_plate.csv, filter data, and save to valid_plate.csv
        # try:
        #     with open(license_plate_file, mode='r') as infile, open(valid_plate_file, mode='a', newline='') as outfile:
        #         reader = csv.reader(infile)
        #         writer = csv.writer(outfile)
        #         for row in reader:
        #             # Apply filtering logic here
                    
        #             # if not helper.check_valid_plate(row[0]): continue
        #             writer.writerow(row)
            
        #     # Clear the license_plate.csv by overwriting it with nothing
        #     with open(license_plate_file, mode='w', newline='') as infile:
        #         pass  # This will empty the file
        
        # except FileNotFoundError:
        #     print("CSV file not found, skipping processing...")
        if flag: 
            subprocess.run(['python', 'clean_and_update_db.py'])
            with open(license_plate_file, mode='w', newline='') as infile:
                pass 
            flag = 0

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 0
    if (new_frame_time-prev_frame_time)!=0: fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # cv2.polylines(frame, [points], isClosed=True, color=(0, 215, 255), thickness=2)
    cv2.imshow('frame', helper.set_hd_resolution(frame))
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
vid.release()
cv2.destroyAllWindows()
