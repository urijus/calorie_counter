"""
Prepare MIDV-500 for YOLO-v8 fine-tuning
and optionally create extra synthetic samples by pasting the cards
onto random background textures.

The script can be used as follows:
1) python prep_midv500_yolov8.py RAW_DIR OUT_DIR            # no synth aug
2) python prep_midv500_yolov8.py RAW_DIR OUT_DIR --synth BG # with synth aug

BG must be a directory containing background *.jpg / *.png images.
"""
import argparse, json, random, shutil, yaml, cv2, pathlib, subprocess, sys, midv500
from tqdm import tqdm
import numpy as np

CLS_NAME      = "credit_card"
TRAIN_SPLIT   = 0.9
CARD_SCALE    = (0.45, 0.8)          # min/max relative scale when compositing
AUG_PER_IMG   = 2                    # synthetic samples per real frame
random.seed(42)


def download_midv(raw_root: pathlib.Path):
    if raw_root.is_dir():
        print("MIDV-500 already present.")
        return

    print("MIDV-500 not found — downloading.")
    dataset_name = "midv500"
    midv500.download_dataset(raw_root, dataset_name)
    print("MIDV-500 downloaded to", raw_root)

def midv_to_yolo(src: pathlib.Path, dst_img: pathlib.Path, dst_lbl: pathlib.Path):
    frames = list(src.glob("*/frames/*.png"))
    for im in tqdm(frames, desc="MIDV frames"):
        with open(im.with_suffix(".json")) as f:
            pts = json.load(f)["quad"]      # list[[x,y],…]
        img = cv2.imread(str(im)); h, w = img.shape[:2]

        norm = [(x / w, y / h) for x, y in pts]
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        cv2.imwrite(str(dst_img / im.name), img)
        (dst_lbl / f"{im.stem}.txt").write_text(f"0 {flat}\n")


def paste_on_bg(card_img: np.ndarray, alpha: np.ndarray,
                bg: np.ndarray) -> tuple[np.ndarray, list[tuple[float,float]]]:
    """Return composite and polygon (normalised coordinates)."""
    H, W = bg.shape[:2]

    # random scale & rotation
    scale = random.uniform(*CARD_SCALE)
    angle = random.uniform(-20, 20)
    h, w  = card_img.shape[:2]
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    card  = cv2.warpAffine(card_img, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderValue=(0, 0, 0, 0))
    a_ch  = cv2.warpAffine(alpha,    M, (w, h), flags=cv2.INTER_LINEAR)

    ph, pw = np.where(a_ch > 0)
    if ph.size == 0:                     # fully transparent → skip
        raise ValueError("alpha vanished")
    ys, xs = ph.min(), pw.min()
    ye, xe = ph.max(), pw.max()
    card   = card[ys:ye+1, xs:xe+1]
    a_ch   = a_ch[ys:ye+1, xs:xe+1]

    ch, cw = card.shape[:2]
    y0 = random.randint(0, H - ch)
    x0 = random.randint(0, W - cw)

    roi = bg[y0:y0+ch, x0:x0+cw].astype(float)
    alpha_f = (a_ch / 255.0)[..., None]
    comp = (alpha_f * card[..., :3] + (1-alpha_f) * roi).astype(np.uint8)
    bg_out = bg.copy(); bg_out[y0:y0+ch, x0:x0+cw] = comp

    # polygon (card corners) in original card before crop →
    # they became axis-aligned bounding box in a_ch > 0 region:
    corners = [(x0          )/W, (y0          )/H,
               (x0+cw - 1   )/W, (y0          )/H,
               (x0+cw - 1   )/W, (y0+ch - 1   )/H,
               (x0          )/W, (y0+ch - 1   )/H]
    return bg_out, corners


def synth_augment(train_dir: pathlib.Path, bg_dir: pathlib.Path):
    print("Generating synthetic composites …")
    cards = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
    bgs   = list(pathlib.Path(bg_dir).glob("*.*"))
    out_i = 0
    for c in tqdm(cards):
        img = cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue                          # need alpha channel
        alpha = img[..., 3]
        card_rgb = img[..., :3]

        for _ in range(AUG_PER_IMG):
            bg   = cv2.imread(str(random.choice(bgs)))
            if bg is None:
                continue
            # resize bg to 640×640 for speed
            bg = cv2.resize(bg, (640, 640), interpolation=cv2.INTER_AREA)
            comp, poly = paste_on_bg(card_rgb, alpha, bg)

            name = f"synth_{out_i:05d}"
            cv2.imwrite(str(train_dir / f"{name}.jpg"), comp)
            lbl  = " ".join(map(lambda x: f"{x:.6f}", poly))
            (train_dir.parent.parent / "labels/train" /
             f"{name}.txt").write_text(f"0 {lbl}\n")
            out_i += 1


def make_yaml(root: pathlib.Path):
    yaml.safe_dump(
        {
            "path": str(root),
            "train": "images/train",
            "val":   "images/val",
            "names": [CLS_NAME],
        },
        (root / "data.yaml").open("w")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", help="folder that will hold /midv (auto-downloaded)")
    ap.add_argument("out_dir", help="output YOLO dataset directory")
    ap.add_argument("--synth", metavar="BG_DIR",
                    help="folder with background textures for synthetic aug")
    args = ap.parse_args()

    raw = pathlib.Path(args.raw_dir).expanduser()
    out = pathlib.Path(args.out_dir).expanduser()

    # Download if needed
    download_midv(raw)

    # Directory layout
    for p in (out / "images/train", out / "images/val",
              out / "labels/train", out / "labels/val"):
        p.mkdir(parents=True, exist_ok=True)

    # Convert MIDV
    tmp_i, tmp_l = out / "_tmp_i", out / "_tmp_l"
    tmp_i.mkdir(exist_ok=True); tmp_l.mkdir(exist_ok=True)
    midv_to_yolo(raw / "midv", tmp_i, tmp_l)

    # Train/val split
    imgs = list(tmp_i.glob("*.png"))
    random.shuffle(imgs); split = int(TRAIN_SPLIT * len(imgs))
    for lst, sub in ((imgs[:split], "train"), (imgs[split:], "val")):
        for im in lst:
            shutil.move(im, out / f"images/{sub}" / im.name)
            shutil.move(tmp_l / f"{im.stem}.txt", out / f"labels/{sub}" / f"{im.stem}.txt")
    shutil.rmtree(tmp_i); shutil.rmtree(tmp_l)

    # Synthetic composites
    if args.synth:
        synth_augment(out / "images/train", out / "labels/train",
                      pathlib.Path(args.synth))

    # Write data.yaml
    make_yaml(out)
    print("Dataset ready at", out)
