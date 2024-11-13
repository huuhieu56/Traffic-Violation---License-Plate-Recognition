import pandas as pd

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