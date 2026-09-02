import cv2
import numpy as np
import os
import re
from PIL import Image as PILImage

# ============================================================
# ラインレーザー靴底溝検出
#
# 1. 横線排除
# 2. 縦の基準線を作成
# 3. 基準線からのずれ / 赤色の弱さから溝候補を検出
# 4. 全画像の溝候補を1枚にマッピング
#
# 対象: image フォルダ内の Txxx_P130.png
# ============================================================

BASE_DIR = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole"
IMAGE_DIR = os.path.join(BASE_DIR, "image")
RESULT_DIR = os.path.join(BASE_DIR, "result")

STAGE1_DIR = os.path.join(RESULT_DIR, "01_horizontal_removed")
STAGE2_DIR = os.path.join(RESULT_DIR, "02_baseline")
STAGE3_DIR = os.path.join(RESULT_DIR, "03_groove")
STAGE4_DIR = os.path.join(RESULT_DIR, "04_mapping")

for d in [STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

# 赤色抽出
H1_LOW = 0
H1_HIGH = 12
H2_LOW = 168
H2_HIGH = 180
S_MIN = 30
V_MIN = 180

# 横線判定
# -30～30度を横線として除去
HORIZONTAL_ANGLE_MIN = -30
HORIZONTAL_ANGLE_MAX = 30
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 10
LINE_REMOVE_THICKNESS = 7

# 基準線
# 1行につきこの範囲内の赤色領域を候補として探す
SEARCH_MARGIN = 35

# 基準線からこの画素以上ずれたら溝候補
DEVIATION_THRESHOLD = 8

# 赤色がこの値より弱ければ溝候補
# 0～255。大きくすると厳しくなる
RED_WEAK_THRESHOLD = 100

# 基準線の多項式次数
BASELINE_DEGREE = 2

# マッピング点の大きさ
MAP_POINT_RADIUS = 2


# ============================================================
# 画像読み込み
# ============================================================

def load_image(path):
    """
    日本語を含むWindowsパスでも読み込めるようにPILを使用
    """
    try:
        rgb = np.array(PILImage.open(path).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"画像読み込み失敗: {path}")
        print(f"  {e}")
        return None


# ============================================================
# 赤色マスク
# ============================================================

def make_red_mask(img):
    """
    レーザーの赤色だけをHSVで抽出
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([H1_LOW, S_MIN, V_MIN], dtype=np.uint8),
        np.array([H1_HIGH, 255, 255], dtype=np.uint8)
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([H2_LOW, S_MIN, V_MIN], dtype=np.uint8),
        np.array([H2_HIGH, 255, 255], dtype=np.uint8)
    )

    mask = cv2.bitwise_or(mask1, mask2)

    # 小さなノイズだけ除去
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# ============================================================
# 1. 横線排除
# ============================================================

def remove_horizontal_lines(mask):
    """
    HoughLinesPで横方向の線を検出し、マスクから除去する。
    縦線は残す。
    """
    result = mask.copy()

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    removed = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # -30～30度を横線とみなす
            if HORIZONTAL_ANGLE_MIN <= angle <= HORIZONTAL_ANGLE_MAX:
                cv2.line(
                    result,
                    (x1, y1),
                    (x2, y2),
                    0,
                    LINE_REMOVE_THICKNESS
                )
                removed += 1

    return result, removed


# ============================================================
# マスクから各行のレーザー位置を取得
# ============================================================

def get_row_centers(mask, img, ref_x=None):
    """
    各yについて、赤色領域の中心xを取得する。
    同じ行に複数の赤領域がある場合は、赤チャンネルの
    合計が最大のクラスタを使用する。
    """
    red = img[:, :, 2].astype(np.float32)

    centers = {}
    strengths = {}

    h, w = mask.shape

    for y in range(h):
        xs = np.where(mask[y] > 0)[0]

        if len(xs) == 0:
            continue

        # 近いx同士をクラスタ化
        gaps = np.where(np.diff(xs) > 4)[0]
        clusters = np.split(xs, gaps + 1)
        clusters = [c for c in clusters if len(c) > 0]
        if len(clusters) == 0:
            continue
 
        if ref_x is not None and y in ref_x:
            target = ref_x[y]
            best_cluster = min(clusters, key=lambda c: abs(np.mean(c) - target))
        else:
            best_cluster = max(clusters, key=lambda c: np.sum(red[y, c]))

        values = red[y, best_cluster]

        # 赤の強さで重み付けした中心
        denominator = np.sum(values)

        if denominator <= 0:
            x = float(np.mean(best_cluster))
        else:
            x = float(np.sum(best_cluster * values) / denominator)

        centers[y] = x
        strengths[y] = float(np.max(values))

    return centers, strengths


# ============================================================
# 2. 縦の基準線を作成
# ============================================================

def make_baseline(centers):
    """
    検出された縦線に対して2次多項式を当てはめ、
    大きく外れている点を除外して再度フィッティングする。

    これにより、溝部分による大きなずれを
    基準線に取り込みにくくする。
    """
    if len(centers) < 20:
        return None

    ys = np.array(sorted(centers.keys()), dtype=np.float64)
    xs = np.array([centers[int(y)] for y in ys], dtype=np.float64)

    degree = min(BASELINE_DEGREE, len(xs) - 1)

    # 初回フィット
    coeff = np.polyfit(ys, xs, degree)
    predicted = np.polyval(coeff, ys)

    residual = np.abs(xs - predicted)

    # 大きなずれを除いて再フィット
    inlier_limit = DEVIATION_THRESHOLD
    inliers = residual <= inlier_limit

    # 点が少なくなりすぎた場合は、上位80%程度を使用
    if np.sum(inliers) < max(10, len(xs) * 0.3):
        limit = np.percentile(residual, 80)
        inliers = residual <= limit

    if np.sum(inliers) >= degree + 1:
        coeff = np.polyfit(ys[inliers], xs[inliers], degree)

    return coeff


def baseline_x(coeff, y):
    return float(np.polyval(coeff, y))


# ============================================================
# 3. 溝候補検出
# ============================================================

def detect_grooves(img, mask, centers, strengths, coeff):
    """
    溝候補を検出する。

    条件1:
        実際のレーザー中心が基準線から
        DEVIATION_THRESHOLD px以上ずれている

    条件2:
        基準線付近の赤色が弱い
        （レーザーが途切れる / 弱くなる）

    戻り値:
        groove_points = [(x, y), ...]
    """
    h, w = mask.shape
    red = img[:, :, 2]

    groove_points = []

    # 靴底が実際に写っている行の範囲だけを対象にする。
    if len(centers) == 0:
        return groove_points
    y_min = min(centers.keys())
    y_max = max(centers.keys())

    for y in range(y_min, y_max + 1):
        bx = baseline_x(coeff, y)

        # 画像外なら無視
        if bx < 0 or bx >= w:
            continue

        actual_x = centers.get(y, None)

        # 基準線付近の赤色強度を確認
        x0 = max(0, int(round(bx)) - 3)
        x1 = min(w, int(round(bx)) + 4)

        local_red = red[y, x0:x1]

        if len(local_red) == 0:
            red_strength = 0
        else:
            red_strength = int(np.max(local_red))

        weak_red = red_strength < RED_WEAK_THRESHOLD

        # 実際に赤線が検出できた場合
        if actual_x is not None:

            # 基準線からのずれ
            deviation = abs(actual_x - bx)

            shifted = deviation >= DEVIATION_THRESHOLD

            if shifted or weak_red:
                groove_points.append(
                    (int(round(actual_x)), int(y))
                )

        # 赤線そのものが検出できない場合
        # → 基準線上を溝候補とする
        elif weak_red:
            groove_points.append(
                (int(round(bx)), int(y))
            )

    return groove_points


# ============================================================
# 画像保存
# ============================================================

def save_image(path, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PILImage.fromarray(rgb).save(path)


# ============================================================
# メイン処理
# ============================================================

files = []

for fname in os.listdir(IMAGE_DIR):

    m = re.fullmatch(r"T(\d+)_P(\d+)\.png", fname, re.IGNORECASE)

    if not m:
        continue

    tilt = int(m.group(1))
    pan = int(m.group(2))

    # 現在はTiltスキャンのみ
    if pan != 130:
        continue

    files.append((tilt, pan, fname))

files.sort()

if len(files) == 0:
    print("対象画像がありません。")
    print("Txxx_P130.png が image フォルダにあるか確認してください。")
    exit()

print(f"対象画像: {len(files)}枚")


# ============================================================
# 4. マッピング用キャンバス
# ============================================================

first_img = load_image(os.path.join(IMAGE_DIR, files[0][2]))

if first_img is None:
    exit()

h, w = first_img.shape[:2]

# 黒背景
mapping = np.zeros((h, w, 3), dtype=np.uint8)

# 背景画像を使用したい場合は以下を有効化
background_path = os.path.join(IMAGE_DIR, "background.png")
if os.path.exists(background_path):
    bg = load_image(background_path)

    if bg is not None and bg.shape[:2] == (h, w):
        mapping = bg.copy()
        print("背景画像をマッピング背景として使用")
else:
    print("background.pngなし → 黒背景")


# ============================================================
# 全画像処理
# ============================================================

total_groove = 0

for index, (tilt, pan, fname) in enumerate(files, start=1):

    print()
    print("=" * 60)
    print(f"[{index}/{len(files)}] {fname}")

    path = os.path.join(IMAGE_DIR, fname)
    img = load_image(path)

    if img is None:
        continue

    # --------------------------------------------------------
    # 赤色マスク
    # --------------------------------------------------------
    img = cv2.convertScaleAbs(img, alpha=2.0, beta=30)
    red_mask = make_red_mask(img)

    # --------------------------------------------------------
    # 1. 横線排除
    # --------------------------------------------------------
    vertical_mask, removed_count = remove_horizontal_lines(red_mask)

    stage1 = np.zeros_like(img)

    # 残ったレーザーを緑色で表示
    stage1[vertical_mask > 0] = (0, 255, 0)

    save_image(
        os.path.join(STAGE1_DIR, fname),
        stage1
    )

    print(f"1. 横線排除: Hough線 {removed_count}本を除去, mask px={int((vertical_mask>0).sum())}")

    # --------------------------------------------------------
    # 各行の縦レーザー位置
    # --------------------------------------------------------
    centers, strengths = get_row_centers(vertical_mask, img)

    # --------------------------------------------------------
    # 2. 基準線
    # --------------------------------------------------------
    coeff = make_baseline(centers)

    if coeff is None:
        print("2. 基準線作成失敗: 検出点不足")

        stage2 = stage1.copy()

        save_image(
            os.path.join(STAGE2_DIR, fname),
            stage2
        )

        # 3, 4は処理できない
        continue

    stage2 = stage1.copy()
    y_min, y_max = min(centers.keys()), max(centers.keys())

    # 基準線を青色で描画
    for y in range(y_min, y_max + 1):
        x = int(round(baseline_x(coeff, y)))

        if 0 <= x < w:
            cv2.circle(
                stage2,
                (x, y),
                1,
                (255, 0, 0),  # BGR: 青
                -1
            )

    save_image(
        os.path.join(STAGE2_DIR, fname),
        stage2
    )

    print(f"2. 基準線作成: {len(centers)}行")

    # --------------------------------------------------------
    # 3. 溝候補検出
    # --------------------------------------------------------
    groove_points = detect_grooves(
        img,
        vertical_mask,
        centers,
        strengths,
        coeff
    )

    stage3 = stage2.copy()

    # 溝候補を赤色で表示
    for x, y in groove_points:

        cv2.circle(
            stage3,
            (x, y),
            2,
            (0, 0, 255),  # BGR: 赤
            -1
        )

    save_image(
        os.path.join(STAGE3_DIR, fname),
        stage3
    )

    print(f"3. 溝候補: {len(groove_points)}点")

    # --------------------------------------------------------
    # 4. マッピング
    # --------------------------------------------------------
    for x, y in groove_points:

        if 0 <= x < w and 0 <= y < h:

            cv2.circle(
                mapping,
                (x, y),
                MAP_POINT_RADIUS,
                (0, 255, 0),  # 緑
                -1
            )

    total_groove += len(groove_points)


# ============================================================
# 4. マップ保存
# ============================================================

mapping_path = os.path.join(
    STAGE4_DIR,
    "groove_mapping.png"
)

save_image(mapping_path, mapping)

print()
print("=" * 60)
print("処理完了")
print(f"処理画像数: {len(files)}枚")
print(f"溝候補総数: {total_groove}点")
print(f"マッピング画像: {mapping_path}")
print("=" * 60)
