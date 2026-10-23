import numpy as np
import cv2
import pickle

MBV3_CONFIG = [
    dict(in_c=16,  kernel=3, exp_c=16,  out_c=16,  se=False, act="RE", stride=1),
    dict(in_c=16,  kernel=3, exp_c=64,  out_c=24,  se=False, act="RE", stride=2),
    dict(in_c=24,  kernel=3, exp_c=72,  out_c=24,  se=False, act="RE", stride=1),
    dict(in_c=24,  kernel=5, exp_c=72,  out_c=40,  se=True,  act="RE", stride=2),
    dict(in_c=40,  kernel=5, exp_c=120, out_c=40,  se=True,  act="RE", stride=1),
    dict(in_c=40,  kernel=5, exp_c=120, out_c=40,  se=True,  act="RE", stride=1),
    dict(in_c=40,  kernel=3, exp_c=240, out_c=80,  se=False, act="HS", stride=2),
    dict(in_c=80,  kernel=3, exp_c=200, out_c=80,  se=False, act="HS", stride=1),
    dict(in_c=80,  kernel=3, exp_c=184, out_c=80,  se=False, act="HS", stride=1),
    dict(in_c=80,  kernel=3, exp_c=184, out_c=80,  se=False, act="HS", stride=1),
    dict(in_c=80,  kernel=3, exp_c=480, out_c=112, se=True,  act="HS", stride=1),
    dict(in_c=112, kernel=3, exp_c=672, out_c=112, se=True,  act="HS", stride=1),
    dict(in_c=112, kernel=5, exp_c=672, out_c=160, se=True,  act="HS", stride=2),
    dict(in_c=160, kernel=5, exp_c=960, out_c=160, se=True,  act="HS", stride=1),
    dict(in_c=160, kernel=5, exp_c=960, out_c=160, se=True,  act="HS", stride=1),
]

def crop_black_borders(img_rgb, tol=7):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img_rgb
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_rgb[y0:y1, x0:x1]

def np_conv2d(x, weight, bias, stride, padding, groups=1):
    C, H, W = x.shape
    out_c, in_c_per_group, kh, kw = weight.shape
    xp = np.pad(x, ((0,0),(padding,padding),(padding,padding)), mode="constant")
    Hp, Wp = xp.shape[1], xp.shape[2]
    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1
    out = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    group_in_c = C // groups
    group_out_c = out_c // groups
    for g in range(groups):
        x_g = xp[g*group_in_c:(g+1)*group_in_c]
        w_g = weight[g*group_out_c:(g+1)*group_out_c].reshape(group_out_c, -1)
        cols = np.empty((group_in_c*kh*kw, out_h*out_w), dtype=np.float32)
        idx = 0
        for c in range(group_in_c):
            for i in range(kh):
                for j in range(kw):
                    cols[idx] = x_g[c, i:i+stride*out_h:stride, j:j+stride*out_w:stride].reshape(-1)
                    idx += 1
        out[g*group_out_c:(g+1)*group_out_c] = (w_g @ cols).reshape(group_out_c, out_h, out_w)
    if bias is not None:
        out += bias.reshape(-1,1,1)
    return out

def np_batchnorm(x, weight, bias, mean, var, eps=1e-5):
    scale = weight / np.sqrt(var + eps)
    shift = bias - mean * scale
    return x * scale.reshape(-1,1,1) + shift.reshape(-1,1,1)

def np_relu(x):
    return np.maximum(x, 0)

def np_hardswish(x):
    return x * (np.clip(x + 3, 0, 6) / 6)

def np_hardsigmoid(x):
    return np.clip(x + 3, 0, 6) / 6

def np_squeeze_excite(x, fc1_w, fc1_b, fc2_w, fc2_b):
    s = x.mean(axis=(1,2))
    s = fc1_w.reshape(fc1_w.shape[0], fc1_w.shape[1]) @ s + fc1_b
    s = np_relu(s)
    s = fc2_w.reshape(fc2_w.shape[0], fc2_w.shape[1]) @ s + fc2_b
    s = np_hardsigmoid(s)
    return x * s.reshape(-1,1,1)

def np_inverted_residual(x, W, i, cfg):
    input_x = x
    prefix = f"features.{i}.block"
    j = 0
    expand_exists = cfg["exp_c"] != cfg["in_c"]
    act_fn = np_hardswish if cfg["act"] == "HS" else np_relu
    if expand_exists:
        x = np_conv2d(x, W[f"{prefix}.{j}.0.weight"], None, stride=1, padding=0, groups=1)
        x = np_batchnorm(x, W[f"{prefix}.{j}.1.weight"], W[f"{prefix}.{j}.1.bias"],
                          W[f"{prefix}.{j}.1.running_mean"], W[f"{prefix}.{j}.1.running_var"])
        x = act_fn(x)
        j += 1
    pad = (cfg["kernel"] - 1) // 2
    x = np_conv2d(x, W[f"{prefix}.{j}.0.weight"], None, stride=cfg["stride"], padding=pad, groups=x.shape[0])
    x = np_batchnorm(x, W[f"{prefix}.{j}.1.weight"], W[f"{prefix}.{j}.1.bias"],
                      W[f"{prefix}.{j}.1.running_mean"], W[f"{prefix}.{j}.1.running_var"])
    x = act_fn(x)
    j += 1
    if cfg["se"]:
        x = np_squeeze_excite(x, W[f"{prefix}.{j}.fc1.weight"], W[f"{prefix}.{j}.fc1.bias"],
                               W[f"{prefix}.{j}.fc2.weight"], W[f"{prefix}.{j}.fc2.bias"])
        j += 1
    x = np_conv2d(x, W[f"{prefix}.{j}.0.weight"], None, stride=1, padding=0, groups=1)
    x = np_batchnorm(x, W[f"{prefix}.{j}.1.weight"], W[f"{prefix}.{j}.1.bias"],
                      W[f"{prefix}.{j}.1.running_mean"], W[f"{prefix}.{j}.1.running_var"])
    if cfg["stride"] == 1 and cfg["in_c"] == cfg["out_c"]:
        x = x + input_x
    return x

def np_mobilenetv3_large_forward(x, W):
    x = np_conv2d(x, W["features.0.0.weight"], None, stride=2, padding=1, groups=1)
    x = np_batchnorm(x, W["features.0.1.weight"], W["features.0.1.bias"],
                      W["features.0.1.running_mean"], W["features.0.1.running_var"])
    x = np_hardswish(x)
    for i, cfg in enumerate(MBV3_CONFIG, start=1):
        x = np_inverted_residual(x, W, i, cfg)
    x = np_conv2d(x, W["features.16.0.weight"], None, stride=1, padding=0, groups=1)
    x = np_batchnorm(x, W["features.16.1.weight"], W["features.16.1.bias"],
                      W["features.16.1.running_mean"], W["features.16.1.running_var"])
    x = np_hardswish(x)
    pooled = x.mean(axis=(1,2))
    h = W["classifier.0.weight"] @ pooled + W["classifier.0.bias"]
    h = np_hardswish(h)
    logits = W["classifier.3.weight"] @ h + W["classifier.3.bias"]
    return logits

def np_softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def np_binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def preprocess_for_numpy_model(img_rgb, img_size, mean, std):
    cropped = crop_black_borders(img_rgb)
    resized = cv2.resize(cropped, (img_size, img_size)).astype(np.float32) / 255.0
    normed = (resized - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return normed.transpose(2, 0, 1).astype(np.float32)

def load_model_package(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def predict_from_package(img_rgb, package):
    W = package["weights"]
    x = preprocess_for_numpy_model(img_rgb, package["image_size"], package["normalize_mean"], package["normalize_std"])
    logits = np_mobilenetv3_large_forward(x, W)
    probs = np_softmax(logits)
    p1 = float(probs[1])
    ent = float(np_binary_entropy(np.array([p1]))[0])
    if ent > package["referral_entropy_threshold"]:
        return {
            "status": "uncertain",
            "predicted_class": 1,
            "predicted_label": package["label_map"][1],
            "probability_referable": p1,
            "entropy": ent,
            "is_high_uncertainty": True,
            "message": "Uncertain. Please consult an ophthalmologist.",
        }
    predicted_class = 1 if p1 >= package["decision_threshold"] else 0
    return {
        "status": "ok",
        "predicted_class": predicted_class,
        "predicted_label": package["label_map"][predicted_class],
        "probability_referable": p1,
        "entropy": ent,
        "is_high_uncertainty": False,
    }