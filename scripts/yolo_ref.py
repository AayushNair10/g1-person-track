"""Python reference of the robot's C++ detection path (letterbox -> model -> decode -> un-letterbox).

Each function mirrors Ultralytics 8.4 (LetterBox, non_max_suppression, scale_boxes) and the C++ code in
src/ (preprocess_math.hpp, postprocess.hpp), so the three can be compared number for number.
Laptop only (numpy + OpenCV from the benchmark venv).
"""
import cv2
import numpy as np

PAD_VALUE = 114  # Ultralytics LetterBox padding grey
PERSON = 0


def letterbox_params(w, h, in_w, in_h):
    """Ultralytics LetterBox(auto=False, center=True) geometry: (new_w, new_h, left, top)."""
    r = min(in_h / h, in_w / w)
    new_w, new_h = round(w * r), round(h * r)  # Python round = half to even (C++: std::nearbyint)
    dw, dh = (in_w - new_w) / 2, (in_h - new_h) / 2
    return new_w, new_h, round(dw - 0.1), round(dh - 0.1)


def letterbox(img, in_w, in_h, exact=False):
    """BGR uint8 frame -> BGR uint8 in_h x in_w image, exactly as Ultralytics does it (cv2 INTER_LINEAR).

    exact=True uses cv2.INTER_LINEAR_EXACT, OpenCV's bit-exact bilinear that gives the same result on every
    platform. Plain INTER_LINEAR depends on the build: opencv-python 5.0 on Apple silicon rounds 3-channel
    2x downscales differently from 1-channel ones (up to 1 grey level). The robot's CUDA kernel follows the
    bit-exact result at 1/2 and 1/4 scale ((a + b + c + d + 2) >> 2) and is within 1 level elsewhere."""
    h, w = img.shape[:2]
    new_w, new_h, left, top = letterbox_params(w, h, in_w, in_h)
    if (w, h) != (new_w, new_h):
        interp = cv2.INTER_LINEAR_EXACT if exact else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    out = np.full((in_h, in_w, 3), PAD_VALUE, np.uint8)
    out[top:top + new_h, left:left + new_w] = img
    return out


def to_tensor(lb_bgr):
    """BGR uint8 HWC -> RGB float32 1x3xHxW in [0, 1]."""
    return np.ascontiguousarray(lb_bgr[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def unletterbox(boxes, w, h, in_w, in_h):
    """Ultralytics scale_boxes + clip_boxes: input-pixel xyxy -> frame-pixel xyxy."""
    new_w, new_h, left, top = letterbox_params(w, h, in_w, in_h)
    gx, gy = new_w / w, new_h / h
    b = boxes.astype(np.float64).copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - left) / gx).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - top) / gy).clip(0, h)
    return b


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-7)


def decode_e2e(out, conf=0.25, cls=PERSON):
    """YOLO26 NMS-free output (1, K, 6) [x1 y1 x2 y2 score class] -> (N, 5) [x1 y1 x2 y2 score], input pixels."""
    p = out[0]
    keep = (p[:, 4] > conf) & (p[:, 5] == cls)
    return p[keep, :5].astype(np.float64)


def nms(boxes, scores, iou_thr):
    """Greedy NMS, highest score first (torchvision.ops.nms order). Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    suppressed = np.zeros(len(order), bool)
    ious = iou_matrix(boxes[order], boxes[order])
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        suppressed |= ious[i] > iou_thr
    return np.array(keep, int)


def decode_raw(out, conf=0.25, iou=0.7, cls=PERSON, max_det=300):
    """Raw one-to-many output (1, 4+nc, N) [cx cy w h scores...] -> NMS -> (N, 5), like Ultralytics
    non_max_suppression(classes=[cls], multi_label=False): a box counts only if `cls` is its best class."""
    p = out[0]
    scores = p[4:]
    best = scores.argmax(0)
    best_score = scores.max(0)
    keep = (best_score > conf) & (best == cls)
    if not keep.any():
        return np.zeros((0, 5))
    cx, cy, w, h = p[0, keep], p[1, keep], p[2, keep], p[3, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float64)
    s = best_score[keep].astype(np.float64)
    k = nms(boxes, s, iou)[:max_det]
    return np.concatenate([boxes[k], s[k, None]], 1)


def decode(out, conf=0.25, iou=0.7):
    """Pick the decoder from the output shape (same rule as the C++ app)."""
    if out.ndim == 3 and out.shape[2] == 6:
        return decode_e2e(out, conf)
    return decode_raw(out, conf, iou)


def match(a, b, thr=0.5):
    """Greedy one-to-one match by IoU. Returns list of (i, j, iou)."""
    m = iou_matrix(a[:, :4], b[:, :4]) if len(a) and len(b) else np.zeros((len(a), len(b)))
    pairs = []
    used_a, used_b = set(), set()
    for flat in np.argsort(-m, axis=None):
        i, j = divmod(int(flat), m.shape[1])
        if m[i, j] < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, float(m[i, j])))
    return pairs


def read_frames(path, n=None, stride=1, size=(1280, 720)):
    """Frames from a video, resized to the robot stream size (1280x720), every `stride`-th, at most n."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = None
    if n is not None:
        want = set(np.linspace(0, total - 1, n).astype(int).tolist())
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (want is None and i % stride == 0) or (want is not None and i in want):
            if size and (f.shape[1], f.shape[0]) != size:
                f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    cap.release()
    return frames
