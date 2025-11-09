import numpy as np, cv2, math

def clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

def compute_area(mask):
    return int(np.sum(mask > 0))

def compute_perimeter(mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    return float(cv2.arcLength(c, True))

def compute_compactness(mask):
    a = compute_area(mask)
    p = compute_perimeter(mask)
    return 0.0 if a == 0 else (p**2) / (4 * math.pi * a)

def opening_ratio(mask):
    h, w = mask.shape
    return compute_area(mask) / float(h*w)

def extract_features(mask):
    mask = clean_mask(mask)
    feats = {
        "area_px": compute_area(mask),
        "perimeter_px": round(compute_perimeter(mask), 2),
        "compactness": round(compute_compactness(mask), 3),
        "opening_ratio": round(opening_ratio(mask), 3)
    }
    return feats