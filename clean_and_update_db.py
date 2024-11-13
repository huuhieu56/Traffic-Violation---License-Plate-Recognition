import pandas as pd
from pymongo import MongoClient
import os
import base64
from datetime import datetime

def check_valid_plate(plate: str) -> bool:
    if (len(plate) <= 7): return False
    parts = plate.split('-')
    unknown_plate = ["13", "42", "44", "45", "46", "87", "91", "96"]
    if (len(parts)<=1) or len(parts[0])<2: return False
    if not (parts[0][0].isdigit() and parts[0][1].isdigit()): return False
    if (plate[0:2] in unknown_plate): return False
    
    if (len(parts)==2):
        if (len(parts[0])!=3): return False
        if (not parts[0][2].isalpha()): return False
        
    elif (len(parts)==3):
        if (len(parts[0])!=2): return False
    if (len(parts[-1])==4):
            for c in parts[-1]:
                if not c.isdigit(): return False
    if (len(parts[-1])==6 and (parts[-1][3]!='.')):
            for i in range(6):
                if (i==3): continue
                if (not parts[-1][i].isdigit()): return False
    if (len(parts[-1])<4 or len(parts[-1])>6): return False
    return True

# Đọc tệp license_plate.csv
df = pd.read_csv('violation_data/license_plate.csv', names=['license_plate', 'time_violation', 'name_img'], header=None)

# Loại bỏ các dòng có biển số hợp lệ
df = df[df['license_plate'].apply(check_valid_plate)]

# Tính tần suất xuất hiện của mỗi biển số
license_plate_counts = df['license_plate'].value_counts()
total_count = len(df)
license_plate_percentage = (license_plate_counts / total_count) * 100

# Lấy danh sách biển số có tần suất xuất hiện lớn hơn 5%
frequent_plates = license_plate_percentage[license_plate_percentage > 5].index

# Lọc DataFrame chỉ chứa các biển số thường xuyên xuất hiện
df_filtered = df[df['license_plate'].isin(frequent_plates)]

# Với mỗi license_plate, chọn hàng có time_violation lớn nhất
df_final = df_filtered.sort_values('time_violation').drop_duplicates('license_plate', keep='last')

# Ghi kết quả vào valid_plate.csv
df_final.to_csv('violation_data/valid_plate.csv', mode='a', index=False, header=False)

# Copy dataFrame
df = df_final

# Connect to MongoDB
client = MongoClient('mongodb://192.168.1.35:27017/')
db = client['traffic_violations']
collection = db['violations']

# Process each row and update to MongoDB
for index, row in df.iterrows():
    try:
        # Convert 'time_violation' from string to datetime
        violation_time = datetime.strptime(row['time_violation'], '%d/%m/%Y %H:%M:%S')
        
        # Image path
        image_path = os.path.join('violation_data/img', f"{row['name_img']}.jpg")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Read and encode image data
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare data
        data = {
            'license_plate': row['license_plate'],
            'violation_time': violation_time,
            'image_data': image_data
        }
        
        # Insert into MongoDB
        collection.insert_one(data)
        print(f"Inserted record for license plate: {row['license_plate']}")
    
    except Exception as e:
        print(f"Error processing row {index}: {e}")

print("Data update complete.")