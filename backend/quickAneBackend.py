#!/usr/bin/env python3
"""
HemaScope v3 — RBC Analysis Backend
- skimage local_maxima watershed: separates overlapping cells
- MobileNetV3Large (Chula-RBC) DL model for cell type classification
- Classification crops are rebuilt as tight, centroid-centred, CLAHE-enhanced
  96x96 windows (matching the training pipeline exactly) instead of using
  the raw watershed bounding box — this was the root cause of unreliable
  predictions, not the preprocessing math or class index mapping.
- Steps: Upload → Preprocess → Segment → Classify → Report
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, json, base64, math, time, tempfile
from pathlib import Path
import numpy as np
import cv2
from skimage import segmentation, measure, morphology, color, feature, filters
from scipy import ndimage as ndi
import tensorflow as tf

# ─────────────────────────────────────────────
# DL MODEL  (MobileNetV2 trained on Chula-RBC)
# ─────────────────────────────────────────────
_MODEL = None
_MAP   = None

# Model file — MobileNetV3Large trained on Chula-RBC-12 (80.81% val acc)
MODEL_FILE   = 'rbc_mobilenetv3_hemascope.h5'
MAP_FILE     = 'rbc_class_map.json'

def _load_dl_model():
    global _MODEL, _MAP
    if _MODEL is None:
        if not Path(MODEL_FILE).exists():
            raise FileNotFoundError(
                f"Model file '{MODEL_FILE}' not found. "
                "Place rbc_mobilenetv3_hemascope.h5 and rbc_class_map.json "
                "in the same folder as this script."
            )
        try:
            _MODEL = tf.keras.models.load_model(MODEL_FILE)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                "Failed to load the .h5 model — this is almost always a "
                "Keras VERSION MISMATCH between Colab (where you trained) "
                "and this machine (where you're deploying), not a problem "
                "with the model itself.\n"
                f"Underlying error: {e}\n\n"
                "Fix: run `pip show tensorflow` in this Colab notebook's "
                "first cell, note the version, then `pip install "
                "tensorflow==<that exact version>` here before starting "
                "the backend. Keras 3 (TF >= 2.16 by default) cannot "
                "reliably read .h5 files saved by older Keras 2 / TF "
                "MobileNetV3 configs."
            ) from e

        with open(MAP_FILE) as f:
            data = json.load(f)

        # idx_to_hemascope: "0"→"Normocytic", "1"→"Macrocytic", ...
        _MAP = {int(k): v for k, v in data['idx_to_hemascope'].items()}
        print(f"[HemaScope] ✅ MobileNetV3Large loaded — "
              f"{data['num_classes']} classes | "
              f"val_acc={data.get('val_accuracy_pct','?')}%")
    return _MODEL, _MAP


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def build_training_style_crop(source_img_bgr, cx, cy, crop_size=96):
    """
    Reproduces the EXACT crop the model was trained on (notebook Cell 4):
      - a fixed square window centred on the cell's centroid (cx, cy)
      - reflect-padding if the window falls off the image edge
      - CLAHE contrast enhancement on the L channel

    This is the #1 fix for "predictions look random": the original backend
    cropped a watershed bounding box (variable size, off-centre, sometimes
    containing neighbouring cells) and ran the DISPLAY-oriented preprocessing
    pipeline (denoise/sharpen/contrast) on it. The model never saw images
    that looked like that — it only ever saw tightly centred, CLAHE-enhanced
    96x96 single-cell crops. Matching that format here is what makes the
    model's real ~80% accuracy show up at inference time.
    """
    half = crop_size // 2
    H, W = source_img_bgr.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))

    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(W, cx + half), min(H, cy + half)
    crop = source_img_bgr[y1:y2, x1:x2]

    pt, pb = max(0, half - cy), max(0, (cy + half) - H)
    pl, pr = max(0, half - cx), max(0, (cx + half) - W)
    if pt or pb or pl or pr:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REFLECT)
    crop = crop[:crop_size, :crop_size]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return crop


def classify_rbc_cell_dl(cell_crop_bgr, use_tta=True):
    """
    Classifies a single RBC crop using MobileNetV3Large.
    Input : BGR crop, ALREADY built via build_training_style_crop()
    Output: (class_label: str, confidence: float 0-1)

    Preprocessing matches training Cell 5 in HemaScope_RBC_v3.ipynb:
      img = tf.cast(img, tf.float32) / 127.5 - 1.0   →  range [-1, 1]

    use_tta: average predictions over the 4 90-degree rotations. The model
    was trained with random 90-degree rotation augmentation (Cell 5), and
    RBCs have no canonical "up" orientation, so averaging over rotations
    reduces prediction variance — it does NOT fix a wrong crop, only
    stabilises an already-correct one.
    """
    model, class_map = _load_dl_model()

    rgb = cv2.cvtColor(cell_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (96, 96)).astype(np.float32)
    img = img / 127.5 - 1.0                      # [-1, 1] — matches training

    if use_tta:
        variants = [np.rot90(img, k) for k in range(4)]
        arr = np.stack(variants, axis=0)          # (4, 96, 96, 3)
        probs_all = model.predict(arr, verbose=0)
        probs = probs_all.mean(axis=0)
    else:
        arr = np.expand_dims(img, axis=0)
        probs = model.predict(arr, verbose=0)[0]

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    return class_map.get(top_idx, 'Mixed/Unclassified'), confidence


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_TEMP  = Path(tempfile.gettempdir()) / "hemascope_v3"
UPLOAD_DIR = BASE_TEMP / "uploads"
OUTPUT_DIR = BASE_TEMP / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# BGR annotation colors — strictly the 10 output classes from hemascope_map:
#   Normal→Normocytic | Macrocyte→Macrocytic | Microcyte→Microcytic
#   Spherocyte→Spherocyte | Target_Cell→Target Cell | Ovalocyte→Elliptocyte
#   Teardrop→Tear Drop | Schistocyte→Schistocyte | Hypochromia→Hypochromic
#   Stomatocyte+Burr_Cell→Mixed/Unclassified
TYPE_COLORS_BGR = {
    "Normocytic":         (0,   200, 0),    # Green
    "Microcytic":         (0,   165, 255),  # Orange
    "Macrocytic":         (255, 165, 0),    # Blue
    "Hypochromic":        (0,   215, 255),  # Yellow
    "Spherocyte":         (0,   0,   255),  # Red
    "Target Cell":        (0,   128, 255),  # Amber
    "Elliptocyte":        (255, 255, 0),    # Cyan
    "Tear Drop":          (128, 128, 0),    # Teal
    "Schistocyte":        (128, 0,   128),  # Purple
    "Mixed/Unclassified": (128, 128, 128),  # Gray (Stomatocyte + Burr Cell)
}

def img_to_b64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img


# ─────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files      = request.files.getlist('images')
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir()

    images_info = []
    for i, f in enumerate(files):
        if f.filename == '':
            continue
        img_id   = f"img_{i}_{int(time.time())}"
        ext      = Path(f.filename).suffix or '.jpg'
        save_path = session_dir / f"{img_id}{ext}"
        f.save(save_path)

        img   = read_image(save_path)
        thumb = cv2.resize(img, (256, int(256 * img.shape[0] / img.shape[1])))

        images_info.append({
            "id":        img_id,
            "name":      f.filename,
            "thumbnail": img_to_b64(thumb),
            "path":      str(save_path)
        })

    with open(session_dir / "meta.json", "w") as j:
        json.dump(images_info, j)

    return jsonify({"session_id": session_id, "images": images_info})


# ─────────────────────────────────────────────
# STEP 2 — PREPROCESS
# ─────────────────────────────────────────────
@app.route('/api/preprocess', methods=['POST'])
def preprocess_image():
    req = request.json
    sid = req.get('session_id')
    iid = req.get('image_id')
    p   = req.get('params', {})

    meta_path = UPLOAD_DIR / sid / "meta.json"
    if not meta_path.exists():
        return jsonify({"error": "Session not found"}), 404
    meta     = json.loads(meta_path.read_text())
    img_info = next((x for x in meta if x['id'] == iid), None)
    if not img_info:
        return jsonify({"error": "Image not found"}), 404

    img   = read_image(img_info['path'])
    steps = []

    # 1. Background Removal (Giemsa-aware)
    if p.get('background_removal', True):
        lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch     = lab[:, :, 0]
        _, bg    = cv2.threshold(l_ch, 218, 255, cv2.THRESH_BINARY)
        bg       = cv2.bitwise_not(bg)
        img      = cv2.bitwise_and(img, img, mask=bg)
        img[bg == 0] = [235, 235, 235]
        steps.append("Giemsa background removal")

    # 2. Denoising
    if p.get('noise_removal', True):
        s   = p.get('noise_strength', 5)
        img = cv2.fastNlMeansDenoisingColored(img, None, s, s, 7, 21)
        steps.append(f"Non-local means denoising (strength {s})")

    # 3. Contrast & Brightness
    if p.get('brightness_fix', True):
        alpha = p.get('contrast', 1.15)
        beta  = p.get('brightness', 5)
        img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        steps.append(f"Brightness/Contrast (α={alpha}, β={beta})")

    # 4. Sharpen
    if p.get('sharpen', True):
        g   = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.5, g, -0.5, 0)
        steps.append("Gaussian unsharp mask")

    out_path = OUTPUT_DIR / f"{sid}_{iid}_pre.jpg"
    cv2.imwrite(str(out_path), img)

    return jsonify({
        "original":  img_info['thumbnail'],
        "processed": img_to_b64(img),
        "steps":     steps
    })


# ─────────────────────────────────────────────
# STEP 3 — SEGMENT
# ─────────────────────────────────────────────
@app.route('/api/segment', methods=['POST'])
def segment_image():
    req = request.json
    sid = req.get('session_id')
    iid = req.get('image_id')

    pre_path = OUTPUT_DIR / f"{sid}_{iid}_pre.jpg"
    if not pre_path.exists():
        return jsonify({"error": "Preprocessed image not found"}), 404

    img  = cv2.imread(str(pre_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    green_ch = img[:, :, 1]

    thresh = filters.threshold_otsu(green_ch)
    binary = green_ch < thresh

    # FIX — over-segmentation root cause: a normal RBC's pale center often
    # crosses back above the Otsu threshold, splitting ONE cell's silhouette
    # into a ring/crescent of disconnected fragments before watershed even
    # runs. A SMALL closing radius bridges these pallor-induced gaps so each
    # real cell becomes one solid blob again.
    # IMPORTANT: radius must stay small (2px). At radius=4 this was found to
    # also bridge gaps between separate, merely-nearby cells, fusing entire
    # clusters of 8-10 real cells into one ~41,000px blob (verified on a
    # real test image — closing(disk=4) ballooned the largest blob from a
    # sane 4,617px to 41,183px). radius=2 still cuts raw fragmentation
    # roughly in half (100->53 components on a 25-cell test image) without
    # the runaway merging.
    binary = morphology.closing(binary, morphology.disk(2))
    binary = ndi.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=200)

    # Smoothing the distance transform before peak-finding prevents the same
    # pallor texture from creating two spurious peaks inside one solid cell.
    distance = ndi.distance_transform_edt(binary)
    distance_smooth = ndi.gaussian_filter(distance, sigma=6)
    coords = feature.peak_local_max(distance_smooth, min_distance=30, labels=binary)
    mask     = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels     = segmentation.watershed(-distance, markers, mask=binary)

    # POST-MERGE — small leftover fragments (slivers/debris from the closing
    # step) get absorbed into a touching neighbor, but ONLY if the neighbor
    # is itself still a plausible single cell. Without this cap, a fragment
    # could glue onto an already-correctly-sized cell and create an
    # oversized multi-cell blob — verified this was happening (one region
    # ballooned to 7499px², ~5x a normal single-cell area, and its centroid
    # landed 30px from the nearest true cell center, corrupting that
    # cell's classification crop).
    MERGE_BELOW_AREA = 500
    MAX_MERGED_AREA  = 3200   # ~2.2x a typical single-cell area in this dataset
    props_pre = measure.regionprops(labels)
    areas_by_label = {p.label: p.area for p in props_pre}
    small_labels = [p.label for p in props_pre if p.area < MERGE_BELOW_AREA]
    for sl in small_labels:
        region_mask = labels == sl
        dilated = morphology.dilation(region_mask, morphology.disk(3))
        neighbor_vals = labels[dilated & ~region_mask]
        neighbor_vals = neighbor_vals[neighbor_vals != 0]
        if len(neighbor_vals) == 0:
            continue
        vals, counts = np.unique(neighbor_vals, return_counts=True)
        # try neighbors in order of shared-border size, skip any that
        # would become implausibly large once merged
        order = np.argsort(-counts)
        frag_area = areas_by_label.get(sl, region_mask.sum())
        for v in vals[order]:
            if v not in areas_by_label:
                continue
            if areas_by_label[v] + frag_area <= MAX_MERGED_AREA:
                labels[region_mask] = v
                areas_by_label[v] += frag_area
                break
        # if no neighbor qualifies, the fragment is left as its own region
        # and will simply be dropped later by the area<300 filter if tiny

    props     = measure.regionprops(labels, intensity_image=gray)
    cells     = []
    display_id = 1
    annotated  = img.copy()
    water_map  = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)
    water_map[labels == 0] = [0, 0, 0]

    for prop in props:
        area = prop.area
        if area < 300 or area > 10000:
            continue

        perim   = prop.perimeter
        circ    = (4 * math.pi * area) / (perim ** 2) if perim > 0 else 0
        bbox    = prop.bbox
        h_box   = bbox[2] - bbox[0]
        w_box   = bbox[3] - bbox[1]
        hull_a  = prop.convex_area
        solidity = area / hull_a if hull_a > 0 else 1.0
        maj     = prop.major_axis_length
        mn      = prop.minor_axis_length
        ar      = maj / mn if mn > 0 else 1.0

        cell_mask_2d = (labels == prop.label).astype(np.uint8)
        y0, x0, y1, x1 = bbox

        roi    = img[y0:y1, x0:x1]
        rmask  = cell_mask_2d[y0:y1, x0:x1]

        dist_roi = ndi.distance_transform_edt(rmask)
        max_d    = np.max(dist_roi)
        inner    = (dist_roi > max_d * 0.5).astype(np.uint8)
        outer    = ((dist_roi <= max_d * 0.5) & (rmask > 0)).astype(np.uint8)

        inner_L  = cv2.mean(roi, mask=inner)[0] if inner.any() else 200
        outer_L  = cv2.mean(roi, mask=outer)[0] if outer.any() else 150
        pallor   = inner_L / outer_L if outer_L > 5 else 1.0

        lab_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L_roi    = lab_roi[:, :, 0]
        cy, cx   = int(prop.centroid[0]), int(prop.centroid[1])
        lcx, lcy = cx - x0, cy - y0

        center_L    = float(L_roi[lcy, lcx]) if (0 <= lcy < h_box and 0 <= lcx < w_box) else float(inner_L)
        central_peak = bool(center_L < inner_L - 5)

        cells.append({
            "id":            display_id,
            "label_id":      int(prop.label),
            "area_px":       round(float(area), 1),
            "perimeter_px":  round(float(perim), 1),
            "circularity":   round(float(circ), 3),
            "solidity":      round(float(solidity), 3),
            "aspect_ratio":  round(float(ar), 3),
            "major_axis_px": round(float(maj), 1),
            "minor_axis_px": round(float(mn), 1),
            "pallor_ratio":  round(float(pallor), 3),
            "central_peak":  central_peak,
            "cell_type":     "Unclassified",
            "centroid":      [cx, cy],
            "bbox":          [int(x0), int(y0), int(w_box), int(h_box)]
        })
        display_id += 1

        contours, _ = cv2.findContours(
            (rmask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            cnt[:, 0, 0] += x0
            cnt[:, 0, 1] += y0
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 1)

    out_water = OUTPUT_DIR / f"{sid}_{iid}_water.jpg"
    out_seg   = OUTPUT_DIR / f"{sid}_{iid}_seg.jpg"
    cv2.imwrite(str(out_water), water_map)
    cv2.imwrite(str(out_seg), annotated)

    with open(OUTPUT_DIR / f"{sid}_{iid}_cells.json", "w") as f:
        json.dump(cells, f)

    return jsonify({
        "original":    img_to_b64(img),
        "watershed":   img_to_b64(water_map),
        "segmented":   img_to_b64(annotated),
        "cell_count":  len(cells),
        "cells":       cells
    })


# ─────────────────────────────────────────────
# STEP 4 — AI CLASSIFY  (replaces /api/convert)
# ─────────────────────────────────────────────
@app.route('/api/classify', methods=['POST'])
def classify_cells():
    """
    For every segmented cell crop, run classify_rbc_cell_dl() and return
    class label + confidence.  No calibration math needed.
    """
    req = request.json
    sid = req.get('session_id')
    iid = req.get('image_id')

    cells_path = OUTPUT_DIR / f"{sid}_{iid}_cells.json"
    if not cells_path.exists():
        return jsonify({"error": "Segmentation data not found"}), 404
    cells = json.loads(cells_path.read_text())

    img_path = OUTPUT_DIR / f"{sid}_{iid}_pre.jpg"
    if not img_path.exists():
        return jsonify({"error": "Preprocessed image not found"}), 404

    source_img    = cv2.imread(str(img_path))
    annotated_img = source_img.copy()

    classified  = []
    type_counts = {}

    for c in cells:
        x0, y0, w, h = c["bbox"]
        cx, cy = c.get("centroid", [x0 + w // 2, y0 + h // 2])

        # Guard cells whose centroid maps outside the image (shouldn't normally happen)
        if source_img is None or source_img.size == 0:
            ctype, conf = "Mixed/Unclassified", 0.0
        else:
            cell_crop = build_training_style_crop(source_img, cx, cy, crop_size=96)
            ctype, conf = classify_rbc_cell_dl(cell_crop, use_tta=True)

        type_counts[ctype] = type_counts.get(ctype, 0) + 1

        classified.append({
            "id":           c["id"],
            "cell_type":    ctype,
            "confidence":   round(conf * 100, 1),
            "circularity":  c["circularity"],
            "solidity":     c["solidity"],
            "aspect_ratio": c["aspect_ratio"],
            "pallor_ratio": c["pallor_ratio"],
            "central_peak": c["central_peak"],
            "area_px":      c["area_px"],
            "bbox":         c["bbox"]
        })

        # Draw colored bounding box on annotation image
        color = TYPE_COLORS_BGR.get(ctype, (128, 128, 128))
        cv2.rectangle(annotated_img, (x0, y0), (x0 + w, y0 + h), color, 1)
        # Small confidence label
        label_txt = f"{ctype[:3]} {conf*100:.0f}%"
        cv2.putText(annotated_img, label_txt, (x0, max(y0 - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    with open(OUTPUT_DIR / f"{sid}_{iid}_classified.json", "w") as f:
        json.dump(classified, f)

    return jsonify({
        "classified_image": img_to_b64(annotated_img),
        "total_cells":      len(classified),
        "type_distribution": type_counts,
        "cells":            classified
    })


# ─────────────────────────────────────────────
# STEP 5 — REPORT
# ─────────────────────────────────────────────
@app.route('/api/report', methods=['POST'])
def generate_report():
    req     = request.json
    sid     = req.get('session_id')
    iid     = req.get('image_id')
    patient = req.get('patient', {})

    cls_path = OUTPUT_DIR / f"{sid}_{iid}_classified.json"
    if not cls_path.exists():
        return jsonify({"error": "Classification data missing"}), 404
    cells = json.loads(cls_path.read_text())

    tot = len(cells) or 1
    td  = {}
    for c in cells:
        td[c['cell_type']] = td.get(c['cell_type'], 0) + 1

    def pct(t):
        return (td.get(t, 0) / tot) * 100

    # ── Findings ──
    findings = []
    for ctype, count in sorted(td.items(), key=lambda x: -x[1]):
        findings.append(f"{ctype}: {count} cells ({pct(ctype):.1f}%)")

    # Confidence summary
    confs = [c['confidence'] for c in cells if c.get('confidence') is not None]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0
    findings.insert(0, f"Model mean confidence: {mean_conf}%.")

    # ── Diagnosis logic — covers all 10 trained model output classes ──
    # Classes: Normocytic | Microcytic | Macrocytic | Hypochromic | Spherocyte
    #          Target Cell | Elliptocyte | Tear Drop | Schistocyte | Mixed/Unclassified
    diagnosis = "Mixed Morphology"
    severity  = "Mild"
    subtype   = "Review smear with haematologist"

    if pct("Normocytic") > 60:
        diagnosis = "Normocytic Range"
        severity  = "Normal"
        subtype   = "Morphology within normal clinical bounds"
    elif pct("Schistocyte") > 2:
        diagnosis = "Schistocytosis Pattern"
        severity  = "Severe"
        subtype   = "Suspect Microangiopathic Haemolytic Anaemia (MAHA)"
    elif pct("Spherocyte") > 5:
        diagnosis = "Spherocytosis Pattern"
        severity  = "Moderate"
        subtype   = "Suspect Hereditary Spherocytosis / Haemolysis"
    elif pct("Microcytic") > 20 and pct("Hypochromic") > 10:
        diagnosis = "Microcytic Hypochromic Anaemia"
        severity  = "Moderate"
        subtype   = "Suspect Iron Deficiency Anaemia"
    elif pct("Microcytic") > 20:
        diagnosis = "Microcytic Anaemia"
        severity  = "Moderate"
        subtype   = "Suspect Iron Deficiency / Thalassaemia"
    elif pct("Macrocytic") > 20:
        diagnosis = "Macrocytic Anaemia"
        severity  = "Moderate"
        subtype   = "Suspect Vitamin B12 / Folate Deficiency"
    elif pct("Hypochromic") > 20:
        diagnosis = "Hypochromic Anaemia"
        severity  = "Moderate"
        subtype   = "Suspect Iron Deficiency / Anaemia of Chronic Disease"
    elif pct("Target Cell") > 5:
        diagnosis = "Target Cell (Codocyte) Morphology"
        severity  = "Mild"
        subtype   = "Suspect Thalassaemia / Liver Disease / HbC Disease"
    elif pct("Tear Drop") > 5:
        diagnosis = "Teardrop Cell (Dacryocyte) Pattern"
        severity  = "Moderate"
        subtype   = "Suspect Myelofibrosis / Extramedullary Haematopoiesis"
    elif pct("Elliptocyte") > 10:
        diagnosis = "Elliptocytosis (Ovalocyte) Pattern"
        severity  = "Mild"
        subtype   = "Suspect Hereditary Elliptocytosis"

    recommendations = [
        "Correlate AI morphological findings with full CBC indices.",
        "Assess iron studies (Ferritin, TIBC) if microcytic/hypochromic pattern predominates.",
        "Assess Vitamin B12 and Folate levels if macrocytic pattern predominates.",
        "Manual review of smear by haematologist is advised for any severe pattern.",
        "Clinical review required if patient is symptomatic."
    ]

    return jsonify({
        "report_id":    f"RPT-{uuid.uuid4().hex[:8].upper()}",
        "image_id":     iid,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "patient":      patient,
        "specimen": {
            "type":              "Peripheral Blood Smear",
            "stain":             "Giemsa/Wright-Leishman",
            "total_rbc_counted": tot,
            "classifier":        "MobileNetV3Large — Chula-RBC-12 (val acc 80.81%)"
        },
        "model_performance": {
            "mean_confidence_pct": mean_conf
        },
        "type_distribution":   td,
        "findings":            findings,
        "clinical_impression": diagnosis,
        "anaemia_subtype":     subtype,
        "severity":            severity,
        "recommendations":     recommendations
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)