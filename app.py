import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis

#load model and processors 
model    = joblib.load('rf_model.pkl')
scaler   = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

FIXED_SIZE = (256, 256)

#helper functions
def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr).astype(np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return float(val)
    except:
        return float(default)


def create_green_leaf_mask(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 20, 20], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    return mask


def create_lesion_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    cond1 = ((h >= 5) & (h <= 40) & (s >= 40) & (v >= 40))
    cond2 = (b >= 140)
    cond3 = (a >= 128)

    lesion = np.where((cond1 & cond2) | (cond1 & cond3), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)

    return lesion

#feature extraction functions
def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}

    rgb_names = ['r', 'g', 'b']
    for i, ch_name in enumerate(rgb_names):
        ch = img_rgb[:, :, i]
        feats[f'rgb_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'rgb_std_{ch_name}']  = float(np.std(ch))
        feats[f'rgb_skew_{ch_name}'] = safe_stat(skew, ch)

    hsv_names = ['h', 's', 'v']
    for i, ch_name in enumerate(hsv_names):
        ch = img_hsv[:, :, i]
        feats[f'hsv_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'hsv_std_{ch_name}']  = float(np.std(ch))

    lab_names = ['l', 'a', 'b']
    for i, ch_name in enumerate(lab_names):
        ch = img_lab[:, :, i]
        feats[f'lab_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'lab_std_{ch_name}']  = float(np.std(ch))

    feats['gray_mean']     = float(np.mean(img_gray))
    feats['gray_std']      = float(np.std(img_gray))
    feats['gray_skew']     = safe_stat(skew, img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)

    hist_density, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    feats['gray_entropy'] = float(-np.sum(hist_density * np.log2(hist_density + 1e-12)))

    bins = 8
    for i, ch_name in enumerate(rgb_names):
        hist = cv2.calcHist([img_rgb], [i], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        for j, val in enumerate(hist):
            feats[f'rgb_hist_{ch_name}_{j}'] = float(val)

    return feats


def extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = float(np.mean(values))
        feats[f'glcm_{prop}_std']  = float(np.std(values))

    return feats


def extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab, radius=1, n_points=8):
    feats = {}
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = float(val)

    feats['lbp_mean'] = float(np.mean(lbp))
    feats['lbp_std']  = float(np.std(lbp))

    return feats


def extract_shape_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    mask = create_green_leaf_mask(img_rgb)

    labeled = label(mask > 0)
    props = regionprops(labeled)

    if len(props) == 0:
        keys = [
            'leaf_area', 'leaf_perimeter', 'leaf_bbox_w', 'leaf_bbox_h',
            'leaf_aspect_ratio', 'leaf_extent', 'leaf_solidity',
            'leaf_equiv_diameter', 'leaf_eccentricity'
        ]
        for k in keys:
            feats[k] = 0.0
        return feats

    region = max(props, key=lambda x: x.area)

    minr, minc, maxr, maxc = region.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc

    feats['leaf_area']            = float(region.area)
    feats['leaf_perimeter']       = float(region.perimeter)
    feats['leaf_bbox_w']          = float(bbox_w)
    feats['leaf_bbox_h']          = float(bbox_h)
    feats['leaf_aspect_ratio']    = float(bbox_w / (bbox_h + 1e-12))
    feats['leaf_extent']          = float(region.extent)
    feats['leaf_solidity']        = float(region.solidity)
    feats['leaf_equiv_diameter']  = float(region.equivalent_diameter_area)
    feats['leaf_eccentricity']    = float(region.eccentricity)

    return feats


def extract_hog_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}

    hog_vec = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    for i, val in enumerate(hog_vec):
        feats[f'hog_{i}'] = float(val)

    grad_x   = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    feats['grad_mean'] = float(np.mean(grad_mag))
    feats['grad_std']  = float(np.std(grad_mag))

    return feats


def extract_lesion_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    lesion_mask = create_lesion_mask(img_rgb)
    leaf_mask   = create_green_leaf_mask(img_rgb)

    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)

    labeled = label(lesion_mask > 0)
    props   = regionprops(labeled)

    leaf_area   = np.sum(leaf_mask > 0)
    lesion_area = np.sum(lesion_mask > 0)

    feats['lesion_area']  = float(lesion_area)
    feats['lesion_ratio'] = float(lesion_area / (leaf_area + 1e-12))
    feats['lesion_count'] = float(len(props))

    if len(props) == 0:
        feats['lesion_mean_area']      = 0.0
        feats['lesion_largest_area']   = 0.0
        feats['lesion_perimeter_sum']  = 0.0
    else:
        areas      = [p.area for p in props]
        perimeters = [p.perimeter for p in props]
        feats['lesion_mean_area']     = float(np.mean(areas))
        feats['lesion_largest_area']  = float(np.max(areas))
        feats['lesion_perimeter_sum'] = float(np.sum(perimeters))

    return feats


#combined feature extractor
def extract_all_features(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    feats = {}
    feats.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_shape_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_hog_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_lesion_features(img_rgb, img_gray, img_hsv, img_lab))
    return list(feats.values())

st.set_page_config(page_title="Tomato Leaf Classifier", page_icon="🍅", layout="centered")

st.title("Tomato Leaf Disease Classifier")
st.write("Upload or capture a tomato leaf image to detect **Bacterial Spot** or **Healthy**.")

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Take Photo"])

img = None

with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

with tab_camera:
    captured = st.camera_input("Point camera at a tomato leaf and take a photo")
    if captured:
        img = Image.open(captured).convert("RGB")
        st.image(img, caption="Captured Image", width=300)

if img is not None:
    img_np = np.array(img)
    img_np = cv2.resize(img_np, FIXED_SIZE)

    with st.spinner("Analyzing leaf..."):
        features          = extract_all_features(img_np)
        features_scaled   = scaler.transform([features])
        features_selected = selector.transform(features_scaled)
        prediction        = model.predict(features_selected)[0]
        proba             = model.predict_proba(features_selected)[0]

    label_map = {
        # string labels
        'Tomato___Bacterial_spot': ('bacterial_spot', 'Bacterial Spot Detected on Tomato Leaf'),
        'Tomato___healthy':        ('healthy',        'Tomato Leaf is Healthy'),
        # numeric labels (0 = bacterial spot, 1 = healthy)
        0: ('bacterial_spot', 'Bacterial Spot Detected on Tomato Leaf'),
        1: ('healthy',        'Tomato Leaf is Healthy'),
    }

    confidence = max(proba) * 100

    result_type, result_label = label_map.get(prediction, ('unknown', f'Unknown Result: {prediction}'))

    st.subheader("Result:")
    if result_type == 'healthy':
        st.success(result_label)
    elif result_type == 'bacterial_spot':
        st.error(result_label)
    else:
        st.warning(result_label)

    st.write(f"Confidence: **{confidence:.1f}%**")
    st.progress(int(confidence))