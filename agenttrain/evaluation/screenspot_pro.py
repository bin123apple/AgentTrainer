import os
import json
from pathlib import Path
from typing import List, Dict, Optional

from huggingface_hub import snapshot_download  # pip install -U huggingface_hub
from PIL import Image                         # pip install pillow
from datasets import Dataset                  # pip install datasets

__all__ = [
    "load_screenspot_dataset",
    "convert_to_qa_format",
]

###############################################
# Internal helper
###############################################

def _collect_entries(ann_dir: Path, img_dir: Path) -> List[Dict]:
    """Parse annotation JSONs and attach absolute image path."""
    entries: List[Dict] = []
    for ann_file in ann_dir.glob("*.json"):
        try:
            ann_list = json.loads(ann_file.read_text("utf-8"))
        except json.JSONDecodeError as e:
            print(f"✖️ {ann_file} parse failed → {e}")
            continue
        for e in ann_list:
            full_img = img_dir / e.get("img_filename", "")
            if not full_img.exists():
                print(f"⚠️ Missing image: {full_img}")
            e["full_img_path"] = str(full_img.resolve())
            entries.append(e)
    return entries

###############################################
# Public API
###############################################

def load_screenspot_dataset(
    *,
    hf_repo: str,
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Download ScreenSpot‑Pro dataset from HF Hub and return annotation list."""
    root = Path(
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            revision=hf_revision,
            token=hf_token or os.getenv("HUGGINGFACE_TOKEN"),
            cache_dir=cache_dir,
            allow_patterns=["annotations/*", "images/*"],
        )
    )
    ann_dir, img_dir = root / "annotations", root / "images"
    if not (ann_dir.is_dir() and img_dir.is_dir()):
        raise RuntimeError("Invalid dataset structure: missing annotations/ or images/ dir")
    return _collect_entries(ann_dir, img_dir)


def convert_to_qa_format(
    *,
    hf_repo: str,
    save_path: str,
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    question_key: str = "instruction",
    answer_from_bbox: bool = True,
) -> Dataset:
    """Convert ScreenSpot‑Pro annotations to QA Arrow dataset.

    **image 字段直接存原始字节流 (bytes)**，方便下游用 `Image.open(io.BytesIO(x))` 读取。
    """
    entries = load_screenspot_dataset(
        hf_repo=hf_repo,
        hf_token=hf_token,
        hf_revision=hf_revision,
        cache_dir=cache_dir,
    )

    qa = {k: [] for k in ("image", "width", "height", "question", "answer")}

    for e in entries:
        img_path = Path(e["full_img_path"])
        qa["image"].append(img_path.read_bytes())        # ★ raw bytes

        if len(e.get("img_size", [])) == 2:
            w, h = e["img_size"]
        else:
            with Image.open(img_path) as im:
                w, h = im.size
        qa["width"].append(int(w))
        qa["height"].append(int(h))

        qa["question"].append(e.get(question_key, ""))
        if answer_from_bbox:
            qa["answer"].append(",".join(map(str, e.get("bbox", []))))
        else:
            qa["answer"].append(e.get("instruction_cn", ""))

    ds = Dataset.from_dict(qa)
    out_dir = Path(save_path).expanduser()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"✔️ Saved {len(ds)} samples to {out_dir}")
    return ds

###############################################
# CLI demo
###############################################
if __name__ == "__main__":
    convert_to_qa_format(
        hf_repo="likaixin/ScreenSpot-Pro",
        save_path="/home/uconn/datasets/screenspot_arrow",  # Arrow directory
    )
