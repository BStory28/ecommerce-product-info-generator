#!/usr/bin/env python3
"""
电商爆款视频基图生成器（本地版）
根据产品图片文件夹，自动识别产品类别、凝练结构化卖点，
通过 OpenCV 本地抠图得到产品透明底图，输出管线下游所需数据结构。

输出: product_layer.png（透明底产品图）+ selling_points.json + base_layers.json

使用方式:
  python generate_base_image.py --folder ./产品图片 --country 泰国 --output ./output
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import requests
except ImportError:
    requests = None

_DESKTOP_SCRIPT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "AI视频脚本")

def _default_output_dir() -> str:
    path = _DESKTOP_SCRIPT_DIR
    os.makedirs(path, exist_ok=True)
    return path

# 自动加载脚本目录下 jeecg_auth
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
try:
    from jeecg_auth import JeecgAuth
except ImportError:
    JeecgAuth = None

# 自动查找 Tesseract OCR 可执行文件
_TESSERACT_CMD = None
for _p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
    if Path(_p).is_file():
        _TESSERACT_CMD = _p
        break

MARKET_STYLE_PRESETS = {
    "china": "chinese-aesthetic",
    "north-america": "north-american-real",
    "us": "north-american-real",
    "europe": "european-minimal",
    "japan": "japanese-healing",
    "korea": "korean-glossy",
    "southeast-asia": "tropical-vibrant",
    "brazil": "brazilian-warm",
}

# ============================================================
# 产品抠图（本地 OpenCV）
# ============================================================

def extract_product(input_path: str, output_path: str, fallback_copy: bool = True) -> str:
    """
    从图片中提取产品主体，去除背景，保存为透明底 PNG。
    策略: 白底→阈值, 深色→阈值, 复杂→GrabCut, 降级→复制
    """
    if cv2 is None:
        if fallback_copy:
            shutil.copy2(input_path, output_path)
            return output_path
        raise RuntimeError("OpenCV 未安装，无法抠图")

    # Use imdecode for Unicode path support
    try:
        with open(input_path, "rb") as f:
            buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None:
        if fallback_copy:
            shutil.copy2(input_path, output_path)
            return output_path
        raise RuntimeError(f"无法读取图片: {input_path}")

    h, w = img.shape[:2]
    has_alpha = img.shape[2] == 4

    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        if np.mean(alpha > 0) > 0.05 and np.mean(alpha < 255) > 0.05:
            _imwrite_unicode(output_path, img)
            return output_path
    else:
        bgr = img

    bg_type = _classify_color_domain(bgr)
    mask = None

    if bg_type in ("white", "transparent"):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif bg_type == "dark":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        try:
            rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
            gm = np.zeros((h, w), np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gm, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        except Exception:
            mask = None

    if mask is None:
        if fallback_copy:
            shutil.copy2(input_path, output_path)
            return output_path
        raise RuntimeError("抠图失败")

    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    _imwrite_unicode(output_path, result)

    print(f"  抠图完成: {Path(output_path).name} (背景: {bg_type})")
    return output_path


def _imwrite_unicode(path: str, img):
    """cv2.imwrite 的 Unicode 路径替代方案"""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    Path(path).write_bytes(buf.tobytes())


# ============================================================
# Seedream 5.0 Lite API 白底图生成
# ============================================================

SEEDREAM_IMAGE_MODEL = os.environ.get("AIGC_IMAGE_MODEL", "2049087333668446209")
SEEDREAM_IMAGE_QUALITY = "high"
SEEDREAM_RESOLUTION = "2k"
SEEDREAM_ASPECT_RATIO = "9:16"


def _img_to_data_uri(path: str) -> str:
    """本地图片 → base64 data URI"""
    import base64, mimetypes
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resize_image_for_upload(input_path: str, output_path: str, max_size: int = 256):
    """缩小图片为小尺寸 JPEG 以减小 data URI 大小"""
    if cv2 is not None:
        try:
            with open(input_path, "rb") as f:
                buf = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                scale = min(max_size / max(h, w), 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                ok, jpg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    Path(output_path).write_bytes(jpg_buf.tobytes())
                    return
        except Exception:
            pass
    # fallback: PIL
    if Image is not None:
        try:
            img = Image.open(input_path)
            img.thumbnail((max_size, max_size))
            img.save(output_path, "JPEG", quality=70)
            return
        except Exception:
            pass
    # final fallback: copy
    shutil.copy2(input_path, output_path)


def _img_to_base64(path: str) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def generate_product_with_seedream(input_path: str, output_path: str,
                                    product_name: str = "", product_category: str = "",
                                    ocr_texts: list = None) -> bool:
    """
    通过 Seedream 5.0 Lite API 生成产品白底图（图生图模式）。
    输入: 产品原图 → base64 → API 图生图 → 下载保存
    输出白底图后, 用 OpenCV 将白色背景转为透明。
    """
    if requests is None:
        print("  [seedream] requests 未安装，降级为本地抠图")
        return False
    if JeecgAuth is None:
        print("  [seedream] jeecg_auth 导入失败，降级为本地抠图")
        return False
    try:
        auth = JeecgAuth()
        auth.validate()
    except ValueError as e:
        print(f"  [seedream] 认证失败: {e}")
        return False

    # 原图 base64
    if not Path(input_path).is_file():
        print(f"  [seedream] 输入文件不存在: {input_path}")
        return False
    img_b64 = _img_to_base64(input_path)
    print(f"  原图 base64: {len(img_b64)} 字符")

    # 构建 prompt
    prompt_parts = ["电商产品白底图"]
    if product_name:
        prompt_parts.append(product_name)
    if product_category:
        prompt_parts.append(f"{product_category}类产品")
    prompt_parts.append("纯白背景，产品主体居中展示，商业产品摄影，高清质感，完整保留全部产品细节")
    prompt_parts.append("无阴影，无文字水印，无背景杂物，产品边缘清晰")
    prompt_text = "，".join(prompt_parts)

    payload = {
        "modelId": SEEDREAM_IMAGE_MODEL,
        "prompt": prompt_text,
        "image": img_b64,
        "imageQuality": SEEDREAM_IMAGE_QUALITY,
        "resolution": SEEDREAM_RESOLUTION,
        "aspectRatio": SEEDREAM_ASPECT_RATIO,
        "count": 1,
    }

    print(f"  调用 Seedream API (图生图)...")
    try:
        resp = requests.post(auth.image_submit_url(), json=payload,
                             headers=auth.get_headers(), timeout=120)
    except Exception as e:
        print(f"  [seedream] 请求异常: {e}")
        return False

    if resp.status_code != 200:
        print(f"  [seedream] API 返回 {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()
    if not data.get("success"):
        err_msg = data.get("message", "") or (data.get("result", {}) or {}).get("message", "")
        print(f"  [seedream] API 失败: {err_msg}")
        return False

    result_data = data.get("result", {})
    image_urls = result_data.get("imageUrls", [])
    if not image_urls:
        print(f"  [seedream] 未返回图片 URL")
        return False

    print(f"  下载结果图片...")
    try:
        img_resp = requests.get(image_urls[0], timeout=60)
        if img_resp.status_code != 200:
            print(f"  下载失败: {img_resp.status_code}")
            return False
        Path(output_path).write_bytes(img_resp.content)
    except Exception as e:
        print(f"  下载异常: {e}")
        return False

    # 将白底转为透明
    try:
        with open(output_path, "rb") as f:
            buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is not None and (img.shape[2] == 3 or img.shape[2] == 4):
            if img.shape[2] == 3:
                rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            else:
                rgba = img
            gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            rgba[:, :, 3] = mask
            _imwrite_unicode(output_path, rgba)
            print(f"  白底图已保存（透明底）: {Path(output_path).name}")
    except Exception:
        pass

    return True


# ============================================================
# 背景环境描述（文本，供下游 Skill2/Skill3 使用）
# ============================================================

def build_background_description(market: str, aesthetic: str, color_palette: str,
                                  scene: str = "", lighting: str = "") -> str:
    """构建背景环境描述文本（下游 Skill3 根据此文本自动渲染背景）"""
    parts = ["电商产品展示背景"]
    parts.append(f"{market}市场风格")
    if aesthetic:
        parts.append(f"{aesthetic}审美")
    if color_palette:
        parts.append(f"{color_palette}色调")
    if scene:
        parts.append(f"{scene}场景")
    if lighting:
        parts.append(f"{lighting}光影")
    parts.extend(["柔化背景", "模糊景深", "无产品", "无人物", "干净底板"])
    return "，".join(parts)


# ============================================================
# 图像合成（同原始版本）
# ============================================================

def composite_layers(product_path: str, background_path: str,
                     people_path: str = None,
                     product_scale: float = 0.4,
                     product_position: tuple = None,
                     people_position: tuple = None,
                     output_path: str = None) -> str:
    if Image is None:
        raise ImportError("PIL (Pillow) 未安装，请执行: pip install Pillow")
    bg = Image.open(background_path).convert("RGBA")
    product = Image.open(product_path).convert("RGBA")
    target_w = int(bg.width * product_scale)
    target_h = int(product.height * (target_w / product.width))
    product_resized = product.resize((target_w, target_h), Image.LANCZOS)
    if product_position is None:
        px = (bg.width - product_resized.width) // 2 + int(bg.width * 0.1)
        py = (bg.height - product_resized.height) // 2
    else:
        px, py = product_position
    canvas = bg.copy()
    canvas.paste(product_resized, (px, py), product_resized)
    if people_path and os.path.exists(people_path):
        people = Image.open(people_path).convert("RGBA")
        pw = int(bg.width * 0.35)
        ph = int(people.height * (pw / people.width))
        people_resized = people.resize((pw, ph), Image.LANCZOS)
        if people_position is None:
            ppx = px - int(people_resized.width * 0.3)
            ppy = py + product_resized.height - people_resized.height + int(people_resized.height * 0.2)
        else:
            ppx, ppy = people_position
        canvas.paste(people_resized, (ppx, ppy), people_resized)
    if output_path:
        canvas.save(output_path, "PNG")
        print(f"合成完成: {output_path}")
    return output_path


# ============================================================
# 图片信息提取（智能识别产品名称/类别/卖点）
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

WHITE_BG_CANDIDATES = [
    re.compile(r"1\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"white.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"白底.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"main.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"主图.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"product.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
    re.compile(r"产品.*\.(png|jpg|jpeg|webp)$", re.IGNORECASE),
]

CATEGORY_WHITE_BG_KEYWORDS = {
    "美妆护肤": ["精华", "面霜", "乳液", "化妆水", "面膜", "口红", "粉底", "眼影", "护肤", "化妆品", "彩妆", "防晒", "美白", "抗皱"],
    "食品零食": ["食品", "零食", "饮料", "茶", "咖啡", "饼干", "巧克力", "糖果", "奶粉", "调味", "食材", "养生", "保健"],
    "家居日用": ["家居", "清洁", "收纳", "日用", "厨具", "餐具", "毛巾", "地毯", "窗帘", "洗衣", "纸巾", "洗漱", "粘", "胶", "滚筒"],
    "服饰鞋包": ["服装", "服饰", "鞋子", "运动鞋", "包包", "女装", "男装", "童装", "内衣", "袜子", "帽子", "围巾", "皮带"],
    "数码3C": ["手机", "电脑", "耳机", "充电", "数据线", "数码", "电子", "智能", "蓝牙", "音箱", "摄像头", "手表", "平板"],
}

CATEGORY_DETAIL_KEYWORDS = {
    "美妆护肤": ["精华", "面霜", "补水", "保湿", "美白", "抗皱", "防晒", "护肤", "化妆品", "彩妆", "粉底", "口红", "眼影", "腮红", "卸妆", "洁面", "爽肤", "乳液", "面膜", "精华液", "烟酰胺", "玻尿酸", "胶原", "修护"],
    "食品零食": ["食品", "零食", "饮料", "茶", "咖啡", "饼干", "巧克力", "糖果", "奶粉", "调味", "食材", "养生", "保健", "维生素", "蛋白", "坚果", "水果", "即食", "烘焙"],
    "家居日用": ["家居", "清洁", "收纳", "日用", "厨具", "餐具", "毛巾", "地毯", "窗帘", "洗衣", "纸巾", "洗漱", "除螨", "除菌", "去污", "防霉", "粘", "胶", "滚筒", "撕", "除尘", "滚", "挂钩", "置物", "拖把", "扫把", "刷", "垃圾桶"],
    "服饰鞋包": ["服装", "服饰", "鞋子", "运动鞋", "包包", "女装", "男装", "童装", "内衣", "袜子", "帽子", "围巾", "皮带", "面料", "尺码", "穿搭"],
    "数码3C": ["手机", "电脑", "耳机", "充电", "数据线", "数码", "电子", "智能", "蓝牙", "音箱", "摄像头", "手表", "平板", "充电器", "快充", "无线", "电池", "屏幕", "内存", "USB"],
}


def _scan_images(folder: str) -> list:
    """扫描文件夹中所有图片"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
    return images


def _img_to_cv2(img_path: str):
    """用OpenCV读取图片，支持中文路径，返回(BGR np.array, alpha_np.array)"""
    try:
        if cv2 is None:
            return None, None
        file_bytes = np.frombuffer(Path(img_path).read_bytes(), dtype=np.uint8)
        mat = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if mat is None:
            return None, None
        if mat.shape[2] == 4:
            return mat[:, :, :3], mat[:, :, 3]
        return mat, None
    except Exception:
        return None, None


def _classify_color_domain(bgr_img, alpha=None) -> str:
    """基于OpenCV的像素分析，判断是白底/深色底/复杂场景"""
    if bgr_img is None:
        return "unknown"
    h, w = bgr_img.shape[:2]
    # 边缘采样区域
    margin = max(5, min(h, w) // 20)
    regions = {
        "tl": bgr_img[:margin, :margin],
        "tr": bgr_img[:margin, -margin:],
        "bl": bgr_img[-margin:, :margin],
        "br": bgr_img[-margin:, -margin:],
        "tc": bgr_img[:margin, w // 2 - margin:w // 2 + margin],
        "bc": bgr_img[-margin:, w // 2 - margin:w // 2 + margin],
    }
    white_count = 0
    for name, roi in regions.items():
        if roi.size == 0:
            continue
        mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
        if mean_r > 220 and mean_g > 220 and mean_b > 220:
            white_count += 1

    # 透明检测
    if alpha is not None:
        alpha_ratio = np.sum(alpha < 30) / alpha.size
        if alpha_ratio > 0.3:
            return "transparent"

    # 如果超过一半的边缘区域是白色，判为白底
    total_regions = len([r for r in regions.values() if r.size > 0])
    if total_regions > 0 and white_count / total_regions >= 0.4:
        return "white"

    # 检查整体亮度均值和方差
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    # 低方差 + 高均值 = 白底
    if mean_val > 200 and std_val < 50:
        return "white"
    # 低方差 + 低均值 = 深色纯底
    if mean_val < 60 and std_val < 40:
        return "dark"
    return "scene"


def _edge_density(bgr_img) -> float:
    """Canny边缘密度 — 高密度表示细节丰富（详情图/场景图）"""
    if bgr_img is None:
        return 0.0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / edges.size)


def _contour_count(bgr_img) -> int:
    """提取轮廓数量 — 越多表示物体越复杂"""
    if bgr_img is None:
        return 0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def _detect_text_regions(bgr_img) -> int:
    """用MSER检测图片中的文字区域数 — >0表示可能有文字"""
    if bgr_img is None:
        return 0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        # 过滤太小的区域
        valid = sum(1 for r in regions if r.shape[0] > 20)
        return valid
    except Exception:
        return 0


def _dominant_colors_cv2(bgr_img, k=5) -> list:
    """用OpenCV K-Means提取主色调"""
    if bgr_img is None or cv2 is None or np is None:
        return []
    try:
        pixels = bgr_img.reshape(-1, 3).astype(np.float32)
        _, labels, centers = cv2.kmeans(pixels, k, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten().astype(np.int32))
        total = np.sum(counts)
        colors = []
        for i in range(k):
            b, g, r = centers[i]
            colors.append({
                "r": int(r), "g": int(g), "b": int(b),
                "ratio": round(float(counts[i]) / total, 4),
            })
        colors.sort(key=lambda c: -c["ratio"])
        return colors
    except Exception:
        return []


def is_mostly_white_or_transparent(img_path: str) -> bool:
    """用OpenCV检测是否白底/透明底"""
    if cv2 is None:
        return False
    bgr, alpha = _img_to_cv2(img_path)
    if bgr is None:
        return False
    domain = _classify_color_domain(bgr, alpha)
    return domain in ("white", "transparent")


def _score_white_bg_cv2(bgr_img) -> float:
    """返回白底图可信度 0~1，越高越像干净白底图"""
    if bgr_img is None:
        return 0.0
    h, w = bgr_img.shape[:2]
    domain = _classify_color_domain(bgr_img)
    edge_den = _edge_density(bgr_img)
    n_contours = _contour_count(bgr_img)
    text_regions = _detect_text_regions(bgr_img)

    score = 0.0
    # 背景类型
    if domain == "white":
        score += 0.35
    elif domain == "transparent":
        score += 0.40

    # 极低边缘密度 → 纯色背景
    if edge_den < 0.02:
        score += 0.20
    elif edge_den < 0.05:
        score += 0.10

    # 轮廓数少 → 单个/少物体
    if n_contours < 5:
        score += 0.20
    elif n_contours < 15:
        score += 0.10

    # 无文字区域 → 干净产品图
    if text_regions < 5:
        score += 0.25
    elif text_regions < 20:
        score += 0.10
    else:
        score -= 0.10  # 文本过多 → 详情图

    return max(0.0, min(score, 1.0))


def _score_detail_cv2(bgr_img, file_size: int) -> float:
    """返回详情图可信度 0~1"""
    if bgr_img is None:
        return 0.0
    h, w = bgr_img.shape[:2]
    domain = _classify_color_domain(bgr_img)
    edge_den = _edge_density(bgr_img)
    n_contours = _contour_count(bgr_img)
    text_regions = _detect_text_regions(bgr_img)

    score = 0.0
    # 场景/深色底 → 非白底图
    if domain in ("scene", "dark"):
        score += 0.20

    # 高边缘密度
    if edge_den > 0.10:
        score += 0.20
    elif edge_den > 0.05:
        score += 0.10

    # 多轮廓
    if n_contours > 50:
        score += 0.15
    elif n_contours > 10:
        score += 0.08

    # 有文字区域
    if text_regions > 50:
        score += 0.25
    elif text_regions > 10:
        score += 0.15

    # 大文件
    if file_size > 60000:
        score += 0.10

    # 大尺寸
    if max(w, h) > 1000:
        score += 0.10

    return max(0.0, min(score, 1.0))


def classify_image(img_path: str, all_images: list = None) -> str:
    """基于OpenCV + 文件名的图片分类：white_bg / main / detail"""
    path = Path(img_path)
    name = path.stem.lower()

    # 1. 文件名模式匹配（最高优先级）
    for pattern in WHITE_BG_CANDIDATES:
        if pattern.match(path.name):
            return "white_bg"

    if re.search(r"detail|详情|desc|展示|show|desc|info", name):
        return "detail"

    file_size = path.stat().st_size

    # 2. OpenCV 评分
    if cv2 is not None:
        bgr, alpha = _img_to_cv2(img_path)
        if bgr is not None:
            bg_score = _score_white_bg_cv2(bgr)
            detail_score = _score_detail_cv2(bgr, file_size)

            # 白底图: 高白底分 + 低详情分
            if bg_score > 0.55 and detail_score < 0.35:
                return "white_bg"

            # 详情图: 高详情分
            if detail_score > 0.40:
                return "detail"

    # 3. 降级：按文件大小排序
    if all_images:
        sorted_by_size = sorted(all_images, key=lambda p: p.stat().st_size, reverse=True)
        mid = len(sorted_by_size) // 2
        if path in sorted_by_size[:mid]:
            return "detail"

    return "main"


def extract_image_metadata(img_path: str) -> dict:
    """提取图片元数据（OpenCV优先，PIL降级）"""
    meta = {
        "path": img_path,
        "filename": Path(img_path).name,
        "file_size": Path(img_path).stat().st_size,
        "width": 0, "height": 0, "format": "", "mode": "", "aspect_ratio": 1.0,
        "dominant_colors": [],
        "edge_density": 0.0,
        "contour_count": 0,
        "text_region_count": 0,
        "background_type": "",
    }

    if cv2 is not None:
        bgr, alpha = _img_to_cv2(img_path)
        if bgr is not None:
            h, w = bgr.shape[:2]
            meta["width"] = w
            meta["height"] = h
            meta["format"] = Path(img_path).suffix.lstrip(".").upper()
            meta["mode"] = "RGBA" if alpha is not None else "BGR"
            meta["aspect_ratio"] = round(w / h, 4) if h > 0 else 1.0
            meta["dominant_colors"] = _dominant_colors_cv2(bgr)
            meta["edge_density"] = round(_edge_density(bgr), 4)
            meta["contour_count"] = _contour_count(bgr)
            meta["text_region_count"] = _detect_text_regions(bgr)
            meta["background_type"] = _classify_color_domain(bgr, alpha)
    elif Image is not None:
        try:
            img = Image.open(img_path)
            meta["width"], meta["height"] = img.size
            meta["format"] = img.format or ""
            meta["mode"] = img.mode
            meta["aspect_ratio"] = round(img.size[0] / img.size[1], 4) if img.size[1] > 0 else 1.0
        except Exception:
            pass

    return meta


def extract_text_from_image(img_path: str, preprocess: bool = True) -> str:
    """
    OCR提取图片文字（OpenCV预处理 + pytesseract）
    """
    try:
        import pytesseract
        if _TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    except ImportError:
        return ""

    try:
        if preprocess and cv2 is not None:
            # OpenCV 预处理管线
            file_bytes = np.frombuffer(Path(img_path).read_bytes(), dtype=np.uint8)
            mat = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if mat is not None:
                if len(mat.shape) == 3 and mat.shape[2] >= 3:
                    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                else:
                    gray = mat
                # 去噪
                denoised = cv2.fastNlMeansDenoising(gray, h=20)
                # 自适应二值化（比全局阈值更适合文字）
                binary = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 12,
                )
                # 放大（小文字识别提升）
                h, w = binary.shape
                if min(h, w) < 800:
                    scale = max(800 / min(h, w), 1.0)
                    binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # OpenCV 结果转 PIL
                pil_img = Image.fromarray(binary)
                text = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
                if text.strip():
                    return re.sub(r"\s+", " ", text).strip()

        # 降级：直接 PIL 识别
        if Image is None:
            return ""
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def score_category_by_text(text: str, keywords: dict) -> tuple:
    """根据文本关键词给品类打分"""
    if not text:
        return ("", 0)
    text_lower = text.lower()
    best_cat = ""
    best_score = 0
    for cat, words in keywords.items():
        score = 0
        for w in words:
            if w.lower() in text_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat
    return (best_cat, best_score)


def analyze_images(folder: str) -> dict:
    """
    综合分析（OpenCV + OCR），返回产品信息
    """
    result = {
        "product_name": "",
        "category": "",
        "selling_points": "",
        "white_bg_path": "",
        "detail_images": [],
        "ocr_texts": [],
        "analysis_summary": "",
        "confidence": 0.0,
    }

    images = _scan_images(folder)
    if not images:
        return result

    # 1. 分类所有图片
    classified = {"white_bg": [], "main": [], "detail": []}
    for img in images:
        cls = classify_image(str(img), images)
        classified[cls].append(img)

    # 2. 确定白底图（优先选文件名含"白底"的，其次按大小）
    white_bg_candidates = classified["white_bg"] or classified["main"]
    if white_bg_candidates:
        pattern_matched = [p for p in white_bg_candidates if any(pt.match(p.name) for pt in WHITE_BG_CANDIDATES)]
        if pattern_matched:
            white_bg_candidates = pattern_matched
        # 优先：文件名含"白底" > 文件最小
        white_bg_candidates.sort(key=lambda p: (0 if "白底" in p.stem else 1, p.stat().st_size))
        result["white_bg_path"] = str(white_bg_candidates[0])

    # 3. 详情图（detail + main，排除白底图）
    detail_images = classified["detail"] + [
        img for img in classified["main"] if str(img) != result["white_bg_path"]
    ]
    result["detail_images"] = [str(p) for p in detail_images]

    # 4. 元数据
    all_metadata = [extract_image_metadata(str(p)) for p in detail_images]
    if result["white_bg_path"]:
        _ = extract_image_metadata(result["white_bg_path"])

    # 5. OCR（取前8张进行文字识别）
    all_texts = []
    for img_path in detail_images[:8]:
        text = extract_text_from_image(str(img_path))
        if text and len(text) > 5:
            all_texts.append(text)
    result["ocr_texts"] = all_texts
    combined_text = " ".join(all_texts)
    filenames_text = " ".join(p.stem for p in images[:10])

    # 6. 推测产品名（从OCR中提取最可能的产品名称）
    product_name_from_ocr = ""
    if combined_text:
        # 去除字间空格
        compact = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", combined_text)
        # 提取 3~20 字的中文词组（排除过长段落/句子）
        candidates = re.findall(r"[\u4e00-\u9fff]{3,20}", compact)
        # 过滤词
        stopwords = {"升级设计", "适用方便", "轻松", "强劲", "顺滑", "高效",
                     "不留", "残胶", "不会", "可以", "采用", "使用",
                     "的时候", "每天", "传统", "不是", "就是",
                     "防尘设计", "壁挂设计", "收纳方便",
                     "助吊", "一一", "一不留",
                     "自带防尘盖", "透视防尘盖", "不误粘带"}
        candidates = [c for c in candidates
                      if c not in stopwords and len(c) >= 4 and len(c) <= 16]
        # 评分: 强产品名特征 + 适中长度
        # 核心词(权重5): 器/机/仪/剂 — 强烈指示这是产品名
        # 中义词(权重2): 纸/毛/刷/筒/膜/胶/包/瓶/装
        # 泛义词(权重1): 盖/带/扣/管/套/粒/挂钩
        strong_kw = re.compile(r"[器机仪剂]")
        mid_kw = re.compile(r"[纸毛刷筒膜胶包瓶装]")
        weak_kw = re.compile(r"[挂号盖带巾扣管套粒挂钩]")
        scored = []
        for c in candidates:
            strong = len(strong_kw.findall(c))
            mid = len(mid_kw.findall(c))
            weak = len(weak_kw.findall(c))
            has_any = strong + mid + weak > 0
            if not has_any:
                continue
            feature_score = strong * 5 + mid * 2 + weak * 1
            length_penalty = abs(len(c) - 7)
            total = feature_score * 10 - length_penalty
            scored.append((-total, len(c), c))
        scored.sort(key=lambda x: (x[0], x[1]))
        if scored:
            product_name_from_ocr = scored[0][2]
    result["product_name"] = product_name_from_ocr or (Path(folder).stem if folder else "")

    # 7. 品类推断
    cat_bg_text = " ".join(
        p.stem for p in white_bg_candidates + [Path(result["white_bg_path"])]
        if result["white_bg_path"]
    )
    cat_from_bg = score_category_by_text(cat_bg_text, CATEGORY_WHITE_BG_KEYWORDS)
    compact = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", combined_text)
    cat_from_detail = score_category_by_text(compact + " " + filenames_text, CATEGORY_DETAIL_KEYWORDS)
    result["category"] = cat_from_detail[0] if cat_from_detail[1] > cat_from_bg[1] else cat_from_bg[0]

    # 8. 卖点提取
    markers = ["%", "倍", "效", "保湿", "补水", "美白", "抗皱", "修护", "清洁",
               "去污", "除菌", "快充", "持久", "轻薄", "大容量", "天然",
               "有机", "无添加", "敏感肌", "不伤手", "一擦即净",
               "升级", "不留", "残胶", "粘", "撕", "强力", "劲", "轻松", "适合",
               "强", "加宽", "防尘", "壁挂", "斜撕", "易揭", "不浪费", "收纳",
               "方便", "高效", "省时", "省力", "不伤", "无痕"]

    def _extract_selling_points(text: str) -> list:
        # 压缩字间空格："粘 力 强 劲" → "粘力强劲"
        compact = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        # 去掉数字前后的空格："强劲一 1不留" → "强劲一1不留"
        compact = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", compact)
        compact = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", compact)
        # 按标点+数字+2+空格 拆分
        parts = re.split(r"[,，。.！!？?、；：;:\n|]+|\d+\s*|\s{3,}", compact)
        results = []
        for p in parts:
            p = p.strip(" -=~·|,./\\\n\r\t")
            p = re.sub(r"^[^一-龥\w%]+|[^一-龥\w%]+$", "", p)
            # 去掉末尾单个汉字（如 OCR 把"——"识别成"一"）
            p = re.sub(r"(?<=[\u4e00-\u9fff])[\u4e00-\u9fff](?=$)", "", p) if len(p) > 4 and len(re.findall(r"[\u4e00-\u9fff]", p)) >= 2 else p
            p = p.strip()
            if len(p) < 4 or len(p) > 40:
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", p)) < 2:
                continue
            if re.search(r"[=~`!@#$^&*()+\[\]{}|;:'\"<>,/]", p):
                continue
            # 长片段可能含多个卖点，按 marker 边界二次拆分
            if len(p) >= 10:
                sub_frags = re.split(r"(?<=[\u4e00-\u9fff])(?=粘|不|撕|易|加|壁|斜|防|清|去|快|持|轻|大|天|有)", p)
                for sf in sub_frags:
                    sf = sf.strip()
                    if len(sf) >= 4 and len(sf) <= 30 and len(re.findall(r"[\u4e00-\u9fff]", sf)) >= 2:
                        for marker in markers:
                            if marker in sf:
                                if sf not in results:
                                    results.append(sf)
                                break
            else:
                for marker in markers:
                    if marker in p:
                        if p not in results:
                            results.append(p)
                        break
        return results

    selling_point_candidates = []
    if combined_text:
        for line in all_texts:
            selling_point_candidates.extend(_extract_selling_points(line))
    # 过滤：移除与产品名相同的项
    if not selling_point_candidates:
        result["selling_points"] = ""
        return result
    product_name = result.get("product_name", "")
    if product_name:
        selling_point_candidates = [s for s in selling_point_candidates if product_name not in s and s not in product_name]
    # 质量评分 & 过滤
    def _score_candidate(s: str) -> float:
        cn = re.findall(r"[\u4e00-\u9fff]", s)
        non_cn = re.findall(r"[^\u4e00-\u9fff\s]", s)
        cn_ratio = len(cn) / max(len(s), 1)
        # 扣分项：非中文字符数、含明显噪声标记
        penalty = len(non_cn) * 2
        if re.search(r"[′'′′>\)\)\*\+\-]", s):
            penalty += 5
        if re.search(r"[A-Z]{2,}", s):
            penalty += 3
        if re.match(r"^[^一-龥]", s) or re.search(r"[^一-龥]$", s):
            penalty += 1
        # 加分项：卖点关键词
        bonus = 0
        for kw in ["强劲", "强力", "升级", "不留", "不伤", "持久", "清洁", "高效", "撕", "粘", "易"]:
            if kw in s:
                bonus += 2
        return cn_ratio * 5 + bonus - penalty

    scored = [(s, _score_candidate(s)) for s in selling_point_candidates if _score_candidate(s) > -2]
    scored.sort(key=lambda x: -x[1])

    # 子片段的二次拆分（如"不留残胶升级热熔"→"不留残胶"+"升级热熔"）
    split_frags = []
    for s, sc in scored:
        if sc > -2 and len(re.findall(r"[\u4e00-\u9fff]", s)) >= 5:
            alt_markers = ["不留", "升级", "不伤", "易揭", "可撕", "轻松", "强力", "高效", "加宽", "防尘", "壁挂"]
            positions = []
            for m in alt_markers:
                idx = s.find(m)
                if idx >= 0:
                    positions.append((idx, m, len(m)))
            if len(set(i for i, _, _ in positions)) >= 1 and len(s) >= 8:
                # 取中间位置的 marker 作为切分点（排除开头0）
                mids = [(i, m, l) for i, m, l in positions if 2 <= i <= len(s) - 4]
                if mids:
                    best = min(mids, key=lambda x: x[0])
                    split_at = best[0]
                    left = s[:split_at].strip()
                    right = s[split_at:].strip()
                    if 3 <= len(left) <= 20 and 3 <= len(right) <= 20:
                        split_frags.extend([left, right])
                        continue
        split_frags.append(s)
    scored2 = []
    for s in split_frags:
        cn = len(re.findall(r"[\u4e00-\u9fff]", s))
        if cn >= 2:
            scored2.append((s, _score_candidate(s)))
    scored2.sort(key=lambda x: -x[1])

    # 模糊去重：子串只保留高分者
    unique = []
    seen_texts = set()
    for s, sc in scored2:
        if sc < -1:
            continue
        if any(s in e or e in s for e in seen_texts):
            continue
        # 字符级别去重（"粘力强劲" vs "强劲粘力"）
        chars = frozenset(re.findall(r"[\u4e00-\u9fff]", s))
        if any(chars == frozenset(re.findall(r"[\u4e00-\u9fff]", e)) for e in seen_texts):
            continue
        unique.append(s)
        seen_texts.add(s)

    result["selling_points"] = "；".join(unique[:5]) if unique else ""

    # 9. 白底图OpenCV核验摘要
    bg_notes = []
    if result["white_bg_path"] and cv2 is not None:
        bgr, alpha = _img_to_cv2(result["white_bg_path"])
        if bgr is not None:
            domain = _classify_color_domain(bgr, alpha)
            edge_den = _edge_density(bgr)
            bg_notes.append(f"背景:{domain}")
            if edge_den < 0.05:
                bg_notes.append("干净白底")
            else:
                bg_notes.append(f"边缘密度:{edge_den:.2%}")

    # 10. 摘要
    parts = []
    if result["product_name"]:
        parts.append(f"产品: {result['product_name']}")
    if result["category"]:
        parts.append(f"类别: {result['category']}")
    if result["white_bg_path"]:
        parts.append(f"白底图: {Path(result['white_bg_path']).name}")
        parts.extend(bg_notes)
    if detail_images:
        parts.append(f"详情图: {len(detail_images)}张")
        if cv2 is not None:
            has_text_detail = any(m.get("text_region_count", 0) > 3 for m in all_metadata)
            avg_edge = round(sum(m.get("edge_density", 0) for m in all_metadata) / max(len(all_metadata), 1), 4)
            parts.append(f"平均边缘密度:{avg_edge:.2%}")
            if has_text_detail:
                parts.append("含文字区域")
    if not product_name_from_ocr:
        parts.append("未识别到文字，可手动补充产品信息")

    result["analysis_summary"] = " | ".join(parts)

    # 可信度
    confidence = 0.4
    if product_name_from_ocr:
        confidence += 0.3
    if result["category"]:
        confidence += 0.15
    if result["selling_points"]:
        confidence += 0.15
    if result["white_bg_path"] and bg_notes:
        confidence += 0.1
    result["confidence"] = min(confidence, 1.0)

    return result


def auto_detect_white_bg(folder: str) -> str:
    """简化的白底图检测（兼容旧版调用）"""
    analysis = analyze_images(folder)
    return analysis.get("white_bg_path", "")


def save_analysis_report(analysis: dict, output_dir: str):
    """保存图片分析报告到JSON"""
    report_path = os.path.join(output_dir, "image_analysis.json")
    report = {
        "product_name": analysis.get("product_name", ""),
        "category": analysis.get("category", ""),
        "selling_points": analysis.get("selling_points", ""),
        "white_bg_image": analysis.get("white_bg_path", ""),
        "detail_images": analysis.get("detail_images", []),
        "ocr_texts": analysis.get("ocr_texts", []),
        "analysis_summary": analysis.get("analysis_summary", ""),
        "confidence": analysis.get("confidence", 0),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  图片分析报告: {report_path}")
    return report_path


# ============================================================
# CLI入口
# ============================================================
# 适用人群/场景推断（OCR启发式）
# ============================================================

_AUDIENCE_RULES = [
    (r"宠物|猫|狗|毛孩|萌宠|除毛", "20-35岁养宠的年轻独居人群、25-45岁有宠物的家庭用户"),
    (r"宝宝|婴儿|孕|哺乳|儿童|小孩", "25-40岁注重安全的母婴人群、家有萌宝的年轻父母"),
    (r"敏感肌|痘痘|痘肌|油皮|干皮|混油|混干|护肤|精华", "18-35岁关注日常护肤的年轻女性、25-45岁有抗衰需求的熟龄肌人群"),
    (r"主妇|妈妈|家务|煮妇|厨房", "30-50岁注重家居清洁的家庭主妇/煮夫"),
    (r"户外|旅行|出差|露营|便携", "20-40岁经常出差旅行的户外/差旅人群"),
    (r"学生|宿舍|租房", "18-25岁在校学生、刚入职场的年轻租房群体"),
    (r"衣物|衣服|毛衣|大衣|西装|羊绒|粘毛", "25-45岁注重衣物护理的通勤人群"),
    (r"办公|桌|工位|键盘|办公室", "22-40岁追求工位整洁的办公人群"),
    (r"收纳|整理|挂|壁挂", "25-45岁追求家居收纳秩序的家庭用户"),
]

_SCENARIO_RULES = [
    (r"居家|家居|客厅|卧室|沙发|床", "日常居家客厅沙发周边清洁、卧室衣柜收纳整理场景"),
    (r"厨房|油烟|灶台|锅|餐具", "每日厨房灶台台面周边清洁、饭后厨具清洗整理场景"),
    (r"衣物|衣服|毛衣|大衣|粘毛|滚粘", "每日出门前衣物表面清洁、衣柜旁毛呢大衣护理场景"),
    (r"浴室|卫生间|洗手间|马桶", "浴室洗漱台周边日常清洁、卫生间马桶清洁除菌场景"),
    (r"办公|桌|工位|键盘|办公室", "日常通勤办公室工位桌面清洁、午休时间键盘清理场景"),
    (r"宠物|猫|狗|毛|除毛", "客厅沙发区宠物掉毛清洁、宠物窝垫周边除毛场景"),
    (r"户外|旅行|出差|露营|便携", "出差酒店行李箱旁开箱场景、户外露营野餐桌收纳场景"),
    (r"车载|车内|汽车", "车内中控台和座椅日常清洁、自驾出游车载杂物收纳场景"),
    (r"收纳|整理|挂|壁挂", "卧室衣柜壁挂收纳整理、玄关挂架壁挂收纳场景"),
]


def infer_audience_and_scenario(ocr_texts: list, category: str = "") -> tuple:
    all_text = " ".join(ocr_texts).lower()
    audience = ""
    scenario = ""
    for pattern, label in _AUDIENCE_RULES:
        if re.search(pattern, all_text):
            audience = label
            break
    for pattern, label in _SCENARIO_RULES:
        if re.search(pattern, all_text):
            scenario = label
            break
    if not audience:
        if category == "家居日用":
            audience = "25-45岁注重家居品质的家庭用户、20-35岁养宠的年轻独居人群"
        elif category == "美妆护肤":
            audience = "18-35岁关注日常护肤的年轻女性、25-45岁有抗衰需求的熟龄肌人群"
        elif category == "食品零食":
            audience = "18-30岁喜爱尝鲜的零食爱好者、25-40岁注重健康饮食的办公室人群"
        elif category == "服饰鞋包":
            audience = "25-40岁追求松弛感穿搭的通勤女性、18-30岁喜爱街头潮流的学生/年轻群体"
        elif category == "数码3C":
            audience = "20-35岁热衷新科技的数码玩家、25-45岁追求效率的职场办公人群"
    if not scenario:
        if category == "家居日用":
            scenario = "日常居家客厅沙发周边清洁、卧室衣柜收纳整理场景"
        elif category == "美妆护肤":
            scenario = "每日早晚家中梳妆台护肤流程、出差旅行酒店洗漱台随身场景"
        elif category == "食品零食":
            scenario = "家庭聚餐餐桌分享场景、办公室工位旁下午茶休闲区"
        elif category == "服饰鞋包":
            scenario = "日常通勤办公室工位周边穿搭、周末探店沿街休闲步道街拍场景"
        elif category == "数码3C":
            scenario = "日常办公书桌工作区周边、周末宅家客厅沙发娱乐游戏场景"
    return audience, scenario


# ============================================================
# AI 卖点自动生成
# ============================================================

# 品类卖点示例（仅作格式参考，AI生成时不使用）
_PRODUCT_SPECIFIC_HINTS = {
    "家居日用": ("根据「{product_name}」的具体功能、材质、收纳方式等生成卖点",
                 ["加厚加固，承重力强", "易收纳不占空间", "一物多用，居家必备"]),
    "美妆护肤": ("根据「{product_name}」的成分配方、肤感质地、使用效果等生成卖点",
                 ["温和不刺激，敏感肌适用", "质地清爽好吸收", "长效保湿，告别干燥"]),
    "食品零食": ("根据「{product_name}」的口感风味、原料产地、包装设计等生成卖点",
                 ["选料精良，口感纯正", "独立小包装，方便随身", "0添加更健康"]),
    "服饰鞋包": ("根据「{product_name}」的面料工艺、版型裁剪、穿搭场景等生成卖点",
                 ["优质面料，亲肤透气", "版型立体，修饰身形", "做工考究，细节见品质"]),
    "数码3C": ("根据「{product_name}」的核心功能、使用体验、便携性等生成卖点",
               ["轻巧便携，随身无负担", "操作简单，即开即用", "功能实用，满足日常需求"]),
}


def _load_sudocode_key() -> str:
    """Load Sudocode API key from env var or .env"""
    key = os.environ.get("SUDOCODE_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SUDOCODE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _auto_gen_full_info(product_name: str, category: str, detail_texts: list = None, usage_instructions: str = ""):
    """AI生成完整信息（卖点+适用人群+适用场景+功能属性+痛点），失败返回(None, None, None, None, None)"""
    key = _load_sudocode_key()
    if key and requests is not None:
        result = _try_ai_full_info(product_name, category, detail_texts, usage_instructions)
        if result:
            return result
    result = _fallback_full_info(product_name, category)
    if result and result[0]:
        sp, aud, scn = result
        pf = _infer_product_function(product_name, category)
        upp = _infer_user_pain_point(product_name, sp)
        return sp, aud, scn, pf, upp
    return None


def _build_name_hints(product_name: str) -> str:
    """从产品名推断特征词提示"""
    hints = []
    if "烘干" in product_name or "吹风" in product_name or "干发" in product_name:
        hints.append("该产品是烘干/吹干类电器，卖点应围绕快速烘干、恒温保护、低噪音、多档调节")
    if "宠物" in product_name or "猫" in product_name or "狗" in product_name:
        hints.append("该产品是宠物用品，卖点应围绕宠物适用、安全静音、防应激、易操作")
    if "收纳" in product_name or "整理" in product_name or "置物" in product_name or "架" in product_name:
        hints.append("该产品是收纳类，卖点应围绕节省空间、承重强、免安装、分类整理")
    if "护肤" in product_name or "精华" in product_name or "面霜" in product_name or "面膜" in product_name:
        hints.append("该产品是护肤品，卖点应围绕成分配方、肤感质地、使用效果")
    if "食品" in product_name or "零食" in product_name or "茶" in product_name or "饮料" in product_name:
        hints.append("该产品是食品饮料，卖点应围绕口感风味、原料品质、包装设计")
    if "手机壳" in product_name or "支架" in product_name or "保护壳" in product_name:
        hints.append("该产品是手机配件，卖点应围绕外观设计、防护功能、使用便利性")
    if "粘毛" in product_name or "滚筒" in product_name or "除毛" in product_name:
        hints.append("该产品是清洁工具，卖点应围绕清洁效果、易用性、收纳设计")
    if "睡衣" in product_name or "居家服" in product_name or "家居服" in product_name:
        hints.append("该产品是居家服饰，卖点应围绕材质舒适度、版型设计、穿脱便利性")
    if "茶" in product_name or "饮料" in product_name or "礼盒" in product_name:
        hints.append("该产品是饮品/礼盒，卖点应围绕口味组合、无糖健康、送礼体面")
    return "；".join(hints) if hints else "根据产品名称推断其功能和特性"


def _try_ai_full_info(product_name: str, category: str, detail_texts: list = None, usage_instructions: str = ""):
    """调用 Sudocode API 生成完整信息，失败返回 None"""
    key = _load_sudocode_key()
    if not key or requests is None:
        return None

    detail_info = ""
    if detail_texts:
        texts = [t for t in detail_texts if len(t) > 5][:3]
        if texts:
            detail_info = "\n详情图文字信息：\n" + "\n".join(f"  - {t[:100]}" for t in texts)

    usage_info = f"\n产品补充说明：{usage_instructions}" if usage_instructions else ""
    name_hint_str = _build_name_hints(product_name)

    prompt = f"""你是一个电商卖点策划专家。根据「{product_name}」这个产品生成结构化商品信息。

商品名称：{product_name}
所属品类：{category}（辅助参考）
产品特性提示：{name_hint_str}{detail_info}{usage_info}

请按以下格式输出（纯JSON，不要markdown）：

{{
  "sell_points": [
    "卖点1（4-12字，短小精悍）",
    "卖点2",
    "卖点3"
  ],
  "target_audience": "XX-XX岁XX人群（含年龄区间）",
  "usage_scenario": "XX场所、XX场所（含具体环境）",
  "product_function": "产品功能属性（从以下枚举中选最匹配的一个，且仅选一个：便捷收纳, 清洁效果, 舒适体验, 耐用品质, 效果对比, 遮盖修饰, 色彩表达, 质地体验, 工具辅助, 持久定妆, 即食口感, 冲泡过程, 开箱仪式, 健康功能, 日常百搭, 场景氛围, 功能实测, 品质工艺, 性能参数, 便捷易用, 外观设计）",
  "user_pain_point": "用户核心痛点（4-8字，从卖点中提炼，如'烘干时间长'或'毛发乱飞'，如无明显痛点可为空）"
}}

要求：
- sell_points 3-5条，每条4-12字，简洁有力，如"快速烘干，省时省力" "低噪静音，宠物不抗拒"
- target_audience 要含年龄区间，如"20-35岁养宠家庭"
- usage_scenario 要含具体环境，如"居家客厅宠物垫旁、宠物店烘干操作台"
- product_function 从枚举中选最匹配的，不限于枚举值但建议贴近
- user_pain_point 从卖点/产品名称中提炼，如"怕毛发乱飞""费时费力"
- 必须紧扣产品本身，不能写成通用描述"""

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 2048, "temperature": 0.5}

    for timeout in (30, 120):
        try:
            resp = requests.post("https://sudocode.run/v1/chat/completions",
                                 json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                first_brace = content.find("{")
                last_brace = content.rfind("}")
                if first_brace >= 0 and last_brace > first_brace:
                    data = json.loads(content[first_brace:last_brace + 1])
                    sp = data.get("sell_points", [])
                    aud = data.get("target_audience", "")
                    scn = data.get("usage_scenario", "")
                    pf = data.get("product_function", "")
                    upp = data.get("user_pain_point", "")
                    if sp and len(sp) >= 2 and aud and scn:
                        return sp, aud, scn, pf, upp
        except requests.exceptions.Timeout:
            continue
        except Exception:
            return None
    return None


def _infer_product_function(product_name: str, category: str) -> str:
    """从产品名+品类推断功能属性（未匹配到品类时按名称关键词匹配）"""
    # 无品类时按产品名称关键词猜测
    if not category or category not in ("美妆护肤", "食品零食", "家居日用", "服饰鞋包", "数码3C"):
        if any(k in product_name for k in ["宠物", "猫", "狗", "烘干", "吹风", "粘毛", "除毛"]):
            category = "家居日用"
        elif any(k in product_name for k in ["口红", "唇", "精华", "面霜", "面膜", "粉底"]):
            category = "美妆护肤"
        elif any(k in product_name for k in ["零食", "食品", "茶", "咖啡", "饮料", "巧克力"]):
            category = "食品零食"
        elif any(k in product_name for k in ["鞋", "包", "服装", "衣", "裤", "裙"]):
            category = "服饰鞋包"
        elif any(k in product_name for k in ["手机", "充电", "耳机", "蓝牙", "数码"]):
            category = "数码3C"
    if category == "美妆护肤":
        if any(k in product_name for k in ["遮瑕", "粉底", "遮", "眉笔", "修容"]):
            return "遮盖修饰"
        if any(k in product_name for k in ["口红", "唇釉", "唇膏", "眼影", "腮红", "眼线"]):
            return "色彩表达"
        if any(k in product_name for k in ["精华", "面霜", "乳液", "面膜", "爽肤"]):
            return "质地体验"
        if any(k in product_name for k in ["蛋", "刷", "工具", "粉扑", "美妆"]):
            return "工具辅助"
        if any(k in product_name for k in ["定妆", "散粉", "喷雾"]):
            return "持久定妆"
        return "质地体验"
    if category == "食品零食":
        if any(k in product_name for k in ["薯片", "饼干", "巧克力", "糖果", "坚果", "即食"]):
            return "即食口感"
        if any(k in product_name for k in ["咖啡", "奶茶", "麦片", "冲泡", "茶饮", "冲"]):
            return "冲泡过程"
        if any(k in product_name for k in ["礼盒", "套装", "零食大"]):
            return "开箱仪式"
        if any(k in product_name for k in ["代餐", "蛋白", "保健", "维生素"]):
            return "健康功能"
        return "即食口感"
    if category == "家居日用":
        if any(k in product_name for k in ["收纳", "置物", "架", "盒", "挂"]):
            return "便捷收纳"
        if any(k in product_name for k in ["清洁", "除", "洗", "拖把", "扫", "刷"]):
            return "清洁效果"
        if any(k in product_name for k in ["枕头", "床垫", "拖鞋", "垫", "靠"]):
            return "舒适体验"
        if any(k in product_name for k in ["刀具", "锅", "五金", "工具"]):
            return "耐用品质"
        if any(k in product_name for k in ["粘毛", "滚筒", "除毛"]):
            return "清洁效果"
        if any(k in product_name for k in ["宠物", "猫", "狗", "烘干", "吹风"]):
            return "效果对比"
        return "舒适体验"
    if category == "服饰鞋包":
        if any(k in product_name for k in ["T恤", "牛仔", "基础", "休闲"]):
            return "日常百搭"
        if any(k in product_name for k in ["礼服", "运动", "户外"]):
            return "场景氛围"
        if any(k in product_name for k in ["防水", "防晒", "登山", "功能"]):
            return "功能实测"
        if any(k in product_name for k in ["皮", "高端", "工艺"]):
            return "品质工艺"
        return "日常百搭"
    if category == "数码3C":
        if any(k in product_name for k in ["手机", "电脑", "显卡", "性能"]):
            return "性能参数"
        if any(k in product_name for k in ["智能", "穿戴", "蓝牙", "耳机"]):
            return "便捷易用"
        if any(k in product_name for k in ["壳", "保护", "配件", "饰"]):
            return "外观设计"
        if any(k in product_name for k in ["充电", "数据线", "电池"]):
            return "耐用品质"
        return "便捷易用"
    return "效果对比"


def _validate_product_function(pf: str, category: str, product_name: str) -> str:
    """验证AI生成的功能属性是否匹配品类，不匹配则用规则推断"""
    if not pf:
        return _infer_product_function(product_name, category)
    # 各品类的有效功能属性集合
    valid_by_cat = {
        "美妆护肤": {"遮盖修饰", "色彩表达", "质地体验", "工具辅助", "持久定妆"},
        "食品零食": {"即食口感", "冲泡过程", "开箱仪式", "健康功能"},
        "家居日用": {"便捷收纳", "清洁效果", "舒适体验", "耐用品质", "效果对比"},
        "服饰鞋包": {"日常百搭", "场景氛围", "功能实测", "品质工艺"},
        "数码3C": {"性能参数", "便捷易用", "外观设计", "耐用品质"},
    }
    if category and category in valid_by_cat:
        valid_set = valid_by_cat[category]
        parts = [p.strip() for p in pf.split(",")]
        for p in parts:
            if p in valid_set:
                return p
        return _infer_product_function(product_name, category)
    # 无品类时直接用规则推断（更准确）
    return _infer_product_function(product_name, category)


def _infer_user_pain_point(product_name: str, sell_points: list) -> str:
    """从产品名+卖点推断用户核心痛点"""
    # 先从卖点中查找暗示痛点的词
    pain_kw = {
        "收纳乱": ["收纳", "乱", "整理", "杂乱", "井井有条"],
        "清洁难": ["清洁", "去污", "除菌", "不留", "脏", "污"],
        "效果不明显": ["效果", "明显", "持久", "长效"],
        "不会用": ["操作", "简单", "方便", "轻松"],
        "操作复杂": ["一键", "省力", "省时", "快捷"],
        "不耐用": ["耐用", "持久", "加固", "承重"],
        "占空间": ["收纳", "节省", "不占", "折叠"],
        "太干卡纹": ["保湿", "补水", "水润", "滋润", "干"],
        "遮不住": ["遮瑕", "遮盖", "遮"],
        "容易掉色": ["持久", "定妆", "不脱", "不掉"],
        "不好吃": ["口感", "好吃", "美味", "入口"],
        "噪音大": ["静音", "低噪", "安静"],
    }
    all_sp = "".join(sell_points)
    for pain, kws in pain_kw.items():
        if any(k in all_sp or k in product_name for k in kws):
            return pain
    # 从产品名关键词反推
    if "粘毛" in product_name or "除毛" in product_name:
        return "清理麻烦"
    if "烘干" in product_name or "吹风" in product_name:
        return "烘干耗时"
    if "宠物" in product_name:
        return "宠物抗拒"
    if "睡衣" in product_name or "家居" in product_name:
        return "穿着不适"
    if "壳" in product_name:
        return "容易摔坏"
    return ""


def _fallback_full_info(product_name: str, category: str):
    """规则回退生成卖点+人群+场景。返回 (sp, aud, scn) 三元组"""
    if not product_name or product_name in ("产品名称", ""):
        return ["设计独特，实用性强", "品质保障，放心使用"], "", ""

    cn_chars = re.findall(r"[\u4e00-\u9fff]", product_name)
    features = list(dict.fromkeys(cn_chars))
    part3 = "".join(features[:3])
    part4 = "".join(features[:4])

    # 卖点（短小精悍，4-12字）
    sp = []
    if "烘干" in product_name or "吹风" in product_name:
        sp.append("快速烘干，省时省力")
        sp.append("恒温呵护，避免烫伤")
        sp.append("低噪静音，宠物不抗拒")
    if "宠物" in product_name or "猫" in product_name or "狗" in product_name:
        if not sp:
            sp.append("宠物专属，安心使用")
        if len(sp) < 4:
            sp.append("一机搞定，轻松洗护")
    if "收纳" in product_name or "置物" in product_name or "架" in product_name:
        sp.append("整齐收纳，节省空间")
    if "护肤" in product_name or "精华" in product_name or "面膜" in product_name:
        sp.append("温和配方，呵护肌肤")
    if "睡衣" in product_name or "家居服" in product_name:
        sp.append("柔软亲肤，舒适好穿")
    if not sp:
        sp.append(f"{part3}，实用好物")
    sp = sp[:5]
    if len(sp) < 2:
        sp.append(f"{part4}，品质之选")

    # 适用人群
    aud = ""
    if "宠物" in product_name or "猫" in product_name or "狗" in product_name:
        aud = "20-40岁养宠家庭、宠物店从业者"
    elif "护肤" in product_name or "精华" in product_name or "面膜" in product_name:
        aud = "18-45岁关注皮肤护理的女性"
    elif "食品" in product_name or "零食" in product_name or "茶" in product_name:
        aud = "18-40岁喜爱零食/茶饮的消费者"
    elif "服饰" in product_name or "鞋" in product_name or "包" in product_name or "睡衣" in product_name:
        aud = "20-35岁注重居家舒适感的年轻群体"
    elif "数码" in product_name or "手机" in product_name or "壳" in product_name:
        aud = "18-35岁喜爱个性数码配件的年轻人"
    elif category == "家居日用":
        aud = "25-45岁注重生活品质的家庭用户"
    elif category == "美妆护肤":
        aud = "18-40岁关注皮肤护理人群"
    elif category == "食品零食":
        aud = "18-35岁喜爱尝鲜的年轻消费者"
    elif category == "服饰鞋包":
        aud = "20-35岁注重穿搭的时尚人群"
    elif category == "数码3C":
        aud = "18-35岁热衷潮玩数码的年轻人"

    # 适用场景
    scn = ""
    if "宠物" in product_name or "猫" in product_name or "狗" in product_name:
        scn = "居家客厅宠物活动区、宠物店美容护理台"
    elif "护肤" in product_name or "精华" in product_name or "面膜" in product_name:
        scn = "家中浴室洗漱台、出差旅行酒店"
    elif "食品" in product_name or "零食" in product_name or "茶" in product_name:
        scn = "家庭聚餐餐桌、办公室茶水间休闲区"
    elif "睡衣" in product_name or "家居服" in product_name:
        scn = "家中卧室休闲区、好友轰趴聚会现场"
    elif "手机" in product_name or "壳" in product_name:
        scn = "日常外出随身使用、居家沙发床头追剧"
    elif "粘毛" in product_name or "滚筒" in product_name:
        scn = "居家客厅沙发旁清洁、出门前衣物快速整理"
    elif category == "家居日用":
        scn = "居家客厅、卧室收纳整理场景"
    elif category == "服饰鞋包":
        scn = "日常通勤穿搭、周末休闲出行"
    elif category == "数码3C":
        scn = "日常办公书桌、居家娱乐场景"

    return sp, aud, scn


# ============================================================

def format_output_as_markdown(output_dir: str, product_name: str = "", country: str = "", video_type: str = "") -> str:
    """将分析结果格式化为 Markdown，供 ArkClaw 聊天框展示"""
    sp_path = Path(output_dir) / "selling_points.json"
    base_path = Path(output_dir) / "base_layers.json"

    sp_data = {}
    if sp_path.exists():
        with open(sp_path, "r", encoding="utf-8") as f:
            sp_data = json.load(f)

    base_data = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

    name = sp_data.get("product_name") or sp_data.get("商品名称") or product_name or "未命名产品"
    category = sp_data.get("category", "未识别")
    core = sp_data.get("核心卖点", {})
    main_sp = core.get("主卖点", "")
    secondary_sps = core.get("次卖点", [])
    audience = sp_data.get("适用人群", "")
    scenario = sp_data.get("适用场景", "")
    bg_desc = base_data.get("background_layer", {}).get("description", "")
    market = base_data.get("target_market", country or "未知")
    vt = base_data.get("video_type", video_type or "未指定")

    lines = [f"📦 **产品分析完成** — 【{name}】{market}市场", "---", ""]
    lines.append("**【一、图片分析摘要】**")
    lines.append(f"- 识别产品: {name}")
    lines.append(f"- 推测品类: {category}")
    lines.append(f"- 白底图: {'已识别' if base_data.get('product_layer', {}).get('ref_image') else '使用原图'}")
    lines.append("")
    lines.append("**【二、凝练卖点】**")
    lines.append(f"1. **商品名称**：{name}")
    lines.append(f"2. **核心卖点**：")
    lines.append(f"   - **主卖点**：{main_sp}")
    for s in secondary_sps[:3]:
        lines.append(f"   - {s}")
    lines.append(f"3. **适用人群**：{audience}")
    lines.append(f"4. **适用场景**：{scenario}")
    lines.append("")
    lines.append("**【三、基图设计方案】**")
    lines.append(f"- 视频类型：{vt}")
    lines.append(f"- 目标市场：{market}")
    if bg_desc:
        lines.append(f"- 背景描述：{bg_desc}")
    lines.append("")
    lines.append("**【四、输出文件】**")
    lines.append(f"📁 输出目录: `{output_dir}`")
    product_png = Path(output_dir) / "product_layer.png"
    lines.append(f"   - `product_layer.png` — {'✓ 已完成' if product_png.is_file() else '✗ 未生成'}")
    lines.append(f"   - `base_layers.json` — 基图分层数据")
    lines.append(f"   - `selling_points.json` — 结构化卖点")
    lines.append("")
    lines.append("---")
    lines.append("✅ **基图已生成**，可继续执行下游管线")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="aigc.hkttok.com 电商基图生成器（三层合成版）")

    parser.add_argument("--folder", "-f", help="产品图片文件夹路径（自动识别白底图）")
    parser.add_argument("--product", "-p", default="", help="层①：产品白底图路径（手动指定，与--folder二选一）")
    parser.add_argument("--detail-images", default="", help="详情图路径列表，逗号分隔（多图上传时使用）")

    parser.add_argument("--output-format", default="json", choices=["json", "markdown", "chat"], help="输出格式: json(文件)|markdown(聊天展示)|chat(精简聊天)")

    parser.add_argument("--name", default="", help="产品名称（不填则自动从文件夹/图片名推测）")
    parser.add_argument("--selling-points", default="", help="原始卖点（分号分隔，不填则自动从图片信息提取）")
    parser.add_argument("--country", default="", help="目标国家，如'泰国'")
    parser.add_argument("--video-type", default="", help="视频类型（8选1）")
    parser.add_argument("--hook-points", default="", help="吸睛点（非必填）")
    parser.add_argument("--duration", type=int, default=0, help="视频时长秒数")
    parser.add_argument("--price", default="", help="福利价格（非必填）")
    parser.add_argument("--background", "-bg", help="层②：背景参考图路径（可选）")
    parser.add_argument("--people", "-ppl", help="层③：人物参考图路径（可选）")
    parser.add_argument("--output", "-o", default=_default_output_dir(), help="输出目录（默认桌面AI视频脚本文件夹）")

    parser.add_argument("--market", "-m", default="china", help="目标市场")
    parser.add_argument("--aesthetic", default="", help="审美风格（层②背景）")
    parser.add_argument("--color", default="", help="色调偏好（层②背景）")
    parser.add_argument("--scene", default="", help="场景描述（层②背景）")
    parser.add_argument("--lighting", default="", help="光影风格（层②背景）")
    parser.add_argument("--no-api", action="store_true", help="不使用 API，降级为本地 OpenCV 抠图")

    parser.add_argument("--people-features", default="", help="人物特征描述（层③）")
    parser.add_argument("--people-action", default="", help="人物动作（层③）")
    parser.add_argument("--people-expression", default="", help="人物表情（层③）")
    parser.add_argument("--target-audience", default="", help="适用人群（不填则自动从OCR推断）")
    parser.add_argument("--usage-scenario", default="", help="适用场景（不填则自动从OCR推断）")
    parser.add_argument("--usage-instructions", default="", help="产品使用说明（若有则提炼后纳入卖点并单独输出）")

    args = parser.parse_args()

    # 统一输入 → 详细参数映射
    COUNTRY_MAP = {
        "中国": "china", "中国大陆": "china", "CN": "china",
        "美国": "north-america", "USA": "north-america", "United States": "north-america",
        "日本": "japan", "Japan": "japan", "JP": "japan",
        "韩国": "korea", "Korea": "korea", "KR": "korea",
        "泰国": "southeast-asia", "Thailand": "southeast-asia",
        "英国": "europe", "德国": "europe", "法国": "europe",
        "巴西": "brazil", "Brazil": "brazil", "BR": "brazil",
    }

    if args.country and args.market == "china":
        args.market = COUNTRY_MAP.get(args.country, args.country)

    product_path = args.product

    # 如果传入了文件夹，进行完整图片分析
    image_analysis = None
    if args.folder:
        print(f"\n{'='*60}")
        print(f"图片分析 — {args.folder}")
        print(f"{'='*60}")
        image_analysis = analyze_images(args.folder)
        print(f"  {image_analysis['analysis_summary']}")

        if not product_path and image_analysis["white_bg_path"]:
            product_path = image_analysis["white_bg_path"]
            print(f"  白底图: {Path(product_path).name}")

        if not args.name and image_analysis["product_name"]:
            args.name = image_analysis["product_name"]
            print(f"  产品名称: {args.name}")

        if not args.selling_points and image_analysis["selling_points"]:
            args.selling_points = image_analysis["selling_points"]
            print(f"  卖点: {args.selling_points[:80]}...")

        if not args.target_audience:
            aud, scn = infer_audience_and_scenario(image_analysis.get("ocr_texts", []), image_analysis.get("category", ""))
            args.target_audience = aud
            args.usage_scenario = scn
            if aud:
                print(f"  适用人群: {aud}")
            if scn:
                print(f"  适用场景: {scn}")

        if not image_analysis["product_name"] and image_analysis["confidence"] < 0.5:
            print(f"\n  ⚠ 图片识别可信度较低 ({image_analysis['confidence']:.0%})")
            print(f"  ⚠ 请检查 --name / --selling-points 是否需要手动指定")

    if not product_path and args.folder and not image_analysis:
        product_path = auto_detect_white_bg(args.folder)

    # 处理 --detail-images（ArkClaw 多图上传时使用）
    detail_paths = []
    if args.detail_images:
        detail_paths = [p.strip() for p in args.detail_images.split(",") if p.strip()]
        valid = [p for p in detail_paths if Path(p).is_file()]
        if valid:
            print(f"  [多图] 详情图: {len(valid)} 张")
            if not args.folder and not image_analysis:
                # 简单分析：用第一张详情图辅助识别
                from PIL import Image as PIL_Image
                try:
                    img = PIL_Image.open(valid[0])
                    print(f"  [多图] 首张详情图: {Path(valid[0]).name} ({img.size[0]}x{img.size[1]})")
                except Exception:
                    pass
        else:
            print(f"  [多图] 详情图路径无效，跳过")

    if not product_path:
        print("错误: 请指定 --product（产品白底图路径）或 --folder（产品图片文件夹）")
        sys.exit(1)

    # 自动推测产品名称（降级）
    if not args.name:
        args.name = Path(product_path).stem
        if args.name and not args.name.isdigit():
            print(f"  产品名称（来自文件名）: {args.name}")

    os.makedirs(args.output, exist_ok=True)

    # ============================================================
    # 产品白底图生成 — 优先选文件名含"白底"的图直接复制，否则 API
    # ============================================================
    print(f"\n{'='*60}")
    print(f"产品白底图生成")
    print(f"{'='*60}")

    product_output = os.path.join(args.output, "product_layer.png")

    # 扫描整个文件夹，找文件名含"白底"的图片
    white_bg_file = None
    if args.folder:
        for f in Path(args.folder).iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                if "白底" in f.stem:
                    white_bg_file = str(f)
                    break

    if white_bg_file:
        print(f"  [模式] 找到白底图: {Path(white_bg_file).name} → 直接复制（颜色保真）")
        shutil.copy2(white_bg_file, product_output)
        print(f"  白底图已复制: {Path(product_output).name}")
    else:
        print(f"  [模式] Seedream 5.0 Lite API 白底图（图生图）")
        api_ok = generate_product_with_seedream(
            product_path, product_output,
            product_name=args.name,
            product_category=image_analysis.get("category", "") if image_analysis else "",
            ocr_texts=image_analysis.get("ocr_texts", []) if image_analysis else [],
        )
        if not api_ok:
            print(f"  → API 失败，降级为本地 OpenCV 抠图")
            extract_product(product_path, product_output)
        else:
            print(f"  → API 生成成功")

    # ============================================================
    # 背景环境描述（文本，不生成图片）
    # ============================================================
    print(f"\n{'='*60}")
    print(f"背景环境描述（文本）")
    print(f"{'='*60}")

    background_description = build_background_description(
        market=args.market,
        aesthetic=args.aesthetic,
        color_palette=args.color,
        scene=args.scene,
        lighting=args.lighting,
    )
    print(f"  {background_description}")

    # ============================================================
    # 输出管线对接文件
    # ============================================================
    pipeline_meta = {
        "product_name": args.name or Path(product_path).stem,
        "video_type": args.video_type,
        "hook_points": args.hook_points,
        "target_duration": args.duration,
        "price": args.price,
        "target_market": args.market,
    }

    if image_analysis:
        save_analysis_report(image_analysis, args.output)

    category = image_analysis.get("category", "") if image_analysis else ""
    base_layers = {
        "product_layer": {
            "ref_image": product_output if Path(product_output).is_file() else product_path,
        },
        "background_layer": {
            "description": background_description,
            "market": args.market,
        },
    }
    base_layers.update(pipeline_meta)
    base_layers["category"] = category
    with open(os.path.join(args.output, "base_layers.json"), "w", encoding="utf-8") as f:
        json.dump(base_layers, f, ensure_ascii=False, indent=2)
    print(f"  输出: base_layers.json")

    selling_points = args.selling_points or (image_analysis.get("selling_points", "") if image_analysis else "")
    sp_list = [s.strip() for s in selling_points.replace("；", ";").split(";") if s.strip()]
    # 若卖点/人群/场景不足，自动生成
    need_full_gen = len(sp_list) < 2 or not args.target_audience or not args.usage_scenario
    if need_full_gen:
        detail_texts = image_analysis.get("ocr_texts", []) if image_analysis else []
        print(f"  自动生成完整商品信息...")
        auto_result = _auto_gen_full_info(args.name or category, category, detail_texts, args.usage_instructions)
        auto_pf = ""
        auto_upp = ""
        if auto_result:
            if len(auto_result) == 5:
                auto_sp, auto_aud, auto_scn, auto_pf, auto_upp = auto_result
            else:
                auto_sp, auto_aud, auto_scn = auto_result
            # 补充卖点
            existing = set(sp_list)
            for s in auto_sp:
                if s not in existing:
                    sp_list.append(s)
                    existing.add(s)
            # 补充人群/场景
            if not args.target_audience and auto_aud:
                args.target_audience = auto_aud
            if not args.usage_scenario and auto_scn:
                args.usage_scenario = auto_scn
            if auto_pf and not args.target_audience:
                pass  # product_function 直接后续写入
            print(f"  → 卖点: {sp_list[:5]}")
            if auto_aud:
                print(f"  → 适用人群: {auto_aud}")
            if auto_scn:
                print(f"  → 适用场景: {auto_scn}")
            if auto_pf:
                print(f"  → 功能属性: {auto_pf}")
            if auto_upp:
                print(f"  → 用户痛点: {auto_upp}")
    # 若有使用说明，提炼后纳入卖点
    usage_refined = ""
    if args.usage_instructions:
        raw = args.usage_instructions.strip().rstrip(".;；。")
        # 按中文句号/分号/换行切分（避免切到数字序号如"1."中的英文句点）
        sentences = [s.strip() for s in re.split(r'[。；\n]', raw) if s.strip()]
        # 去除非中文开头的枚举序号行（如"1.拉开拉链" -> "拉开拉链"）
        cleaned = []
        for s in sentences:
            s = re.sub(r'^[\d\s]+[.、．\)）]\s*', '', s).strip()
            if s and s not in cleaned:
                cleaned.append(s)
        usage_refined = "；".join(cleaned[:3]) if len(cleaned) > 1 else raw
        # 将使用说明提炼为一条卖点添加入 sp_list（若还不存在）
        first = cleaned[0] if cleaned else raw
        derived = f"操作简单：{first}" if len(first) < 30 else f"操作简单，{first[:20]}…"
        if derived not in sp_list:
            sp_list.append(derived)
    # 额外说明：取使用说明 + 吸睛点合并
    extra_notes = ""
    notes = []
    if args.hook_points:
        notes.append(f"吸睛点：{args.hook_points}")
    if usage_refined:
        notes.append(f"使用说明：{usage_refined}")
    if notes:
        extra_notes = "；".join(notes)

    selling_points_data = {
        "产品图": product_path,
        "国家": args.country or args.market,
        "视频类型": args.video_type or "",
        "额外说明": extra_notes,
        "商品卖点": {
            "1、商品名称": args.name or "产品名称",
            "2、核心卖点": {
                "主卖点": sp_list[0] if sp_list else "",
                "次卖点": sp_list[1:] if len(sp_list) > 1 else [],
            },
            "3、适用人群": args.target_audience or "",
            "4、适用场景": args.usage_scenario or "",
        },
        # 兼容 Skill2 字段
        "商品名称": args.name or "产品名称",
        "product_name": args.name or "产品名称",
        "核心卖点": {
            "主卖点": sp_list[0] if sp_list else "",
            "次卖点": sp_list[1:] if len(sp_list) > 1 else [],
        },
        "适用人群": args.target_audience or "",
        "适用场景": args.usage_scenario or "",
        "category": category,
        "product_function": _validate_product_function(auto_pf, category, args.name or ""),
        "user_pain_point": auto_upp if auto_upp else _infer_user_pain_point(args.name or "", sp_list),
    }
    if usage_refined:
        selling_points_data["使用说明"] = usage_refined
    selling_points_data.update(pipeline_meta)
    with open(os.path.join(args.output, "selling_points.json"), "w", encoding="utf-8") as f:
        json.dump(selling_points_data, f, ensure_ascii=False, indent=2)
    print(f"  输出: selling_points.json")
    print(f"  产品图: {Path(product_path).name}")
    print(f"  国家: {args.country or args.market}")
    if extra_notes:
        print(f"  额外说明: {extra_notes[:80]}...")

    print(f"\n{'='*60}")
    print(f"生成完成!")
    print(f"输出目录: {os.path.abspath(args.output)}")
    print(f"  product_layer.png: {'完成' if Path(product_output).is_file() else '失败'}")
    print(f"  base_layers.json: 已生成")
    print(f"  selling_points.json: 已生成")
    print(f"{'='*60}")

    if args.output_format in ("markdown", "chat"):
        md = format_output_as_markdown(
            args.output, args.name or "",
            args.country or args.market, args.video_type
        )
        print("\n" + "=" * 60)
        print("ARKCLAW_CHAT_OUTPUT")
        print("=" * 60)
        print(md)


if __name__ == "__main__":
    main()
