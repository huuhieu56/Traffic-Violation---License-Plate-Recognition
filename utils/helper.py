import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# from paddleocr import PaddleOCR
import numpy as np
import cv2
from paddleocr import PaddleOCR

def set_hd_resolution(image):
    """
    Set video resolution (for displaying only)
    Arg:
        image (OpenCV image): video frame read by cv2
    """
    height, width, _ = image.shape
    ratio = height / width
    image = cv2.resize(image, (1280, int(1280 * ratio)))
    return image

def draw_text(img, text,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255)):
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
def delete_file(path):
    """
    Delete generated file during inference
    """
    if os.path.exists(path):
        os.remove(path)

def crop_expanded_plate(plate_xyxy, img, expand_ratio=0.1):
    # Original coordinates
    x_min, y_min, x_max, y_max = plate_xyxy

    # Calculate the width and height of the original cropping area
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the expansion amount (10% of the width and height by default)
    expand_x = int(expand_ratio * width)
    expand_y = int(expand_ratio * height)

    # Calculate the new coordinates with expansion
    new_x_min = max(x_min - expand_x, 0)
    new_y_min = max(y_min - expand_y, 0)
    new_x_max = min(x_max + expand_x, img.shape[1])
    new_y_max = min(y_max + expand_y, img.shape[0])

    # Crop the expanded area
    cropped_plate = img[new_y_min:new_y_max, new_x_min:new_x_max, :]

    return cropped_plate

# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    return(math.isclose(y_pred, y, abs_tol = 3))

def check_valid_plate(plate: str) -> bool:
    if (len(plate) <= 7): return False
    parts = plate.split('-')
    unknown_plate = ["13", "42", "44", "45", "46", "87", "91", "96"]
    if (len(parts)<=1) or len(parts[0])<2: return False
    if not (parts[0][0].isdigit() and parts[0][1].isdigit()): return False
    if (plate[0:2] in unknown_plate): return False
    
    if (len(parts)==2):
        if (len(parts[0])<2): return False
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
    


ocr = PaddleOCR(lang="en")

def read_plate_ppocr(plate_path) -> str:
    result = ocr.ocr(plate_path)[0]
    if (result==None): return "unknown"
    print(result)
    text = ""
    cnt = 0
    
    for r in result:
        print("OCR"+str(cnt), r)
        scores = r[1][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 80:
            if cnt==0: text += r[1][0]
            else: text += "-" + r[1][0]  
            cnt += 1
        else:
            return "unknown"
        
    # pattern = re.compile('[\W]')
    # text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    # if cnt!=len(result): return "unknown"
    if (text[2]=='8'):
        text = text[:2] + 'B' + text[3:]
    if (text[2]=='-' and text[3]=='8'):
        text = text[:2] + 'B' + text[3:]
    if (text[2]=='6'):
        text = text[:2] + 'G' + text[3:]
    if (text[2]=='-' and text[3]=='6'):
        text = text[:2] + 'G' + text[3:]
    return str(text)

def read_plate(yolo_license_plate, im):
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()
    
    if not (7 <= len(bb_list) <= 10):
        return "unknown"
    
    center_list = [[(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, bb[-1]] for bb in bb_list]
    y_mean = sum(c[1] for c in center_list) / len(center_list)
    
    l_point = min(center_list, key=lambda c: c[0])
    r_point = max(center_list, key=lambda c: c[0])
    
    LP_type = "1"
    if l_point[0] != r_point[0]:
        for c in center_list:
            if not check_point_linear(c[0], c[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                LP_type = "2"
                break
    
    line_1 = [c for c in center_list if c[1] <= y_mean]
    line_2 = [c for c in center_list if c[1] > y_mean]
    
    license_plate = ""
    if LP_type == "2":
        license_plate = "".join(str(c[2]) for c in sorted(line_1, key=lambda x: x[0]))
        license_plate += "-"
        license_plate += "".join(str(c[2]) for c in sorted(line_2, key=lambda x: x[0]))
    else:
        license_plate = "".join(str(c[2]) for c in sorted(center_list, key=lambda x: x[0]))
        license_plate = license_plate[:3]+license_plate[3:]
        
    return license_plate

def upscale_image (image, scale=2.0):
    height, width = image.shape[:2]
    new_dimensions = (int(width*scale), int(height*scale))
    upscale_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
    return upscale_image

def denoise_image (image):
    gray_image = image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    return denoised_image

def adjust_contrast (image):
    if (len(image.shape)) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_image = clahe.apply(gray_image)
    return contrast_image

def preprocess_image (image):
    upscaled_image = upscale_image(image)
    
    denoised_image = denoise_image(upscaled_image)
    
    contrast_adjusted_image = adjust_contrast(denoised_image)
    
    return contrast_adjusted_image
    