import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional

from huggingface_hub import snapshot_download  # pip install -U huggingface_hub
from PIL import Image  # pip install pillow

__all__ = [
    "load_screenspot_dataset",
    "convert_to_qa_format",
]

###############################################
# Internal helper
###############################################

def _collect_entries(ann_dir: Path, img_dir: Path) -> List[Dict]:
    """Parse all annotation jsons under ``ann_dir`` and attach full image path.

    Parameters
    ----------
    ann_dir : Path
        Directory containing ``*.json`` files.
    img_dir : Path
        Directory containing raw images.

    Returns
    -------
    List[Dict]
        Annotation list with an extra key ``full_img_path``.
    """
    all_entries: List[Dict] = []
    for ann_file in ann_dir.glob("*.json"):
        try:
            entries = json.loads(ann_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"✖️ JSON parse failed: {ann_file} → {e}")
            continue

        for entry in entries:
            img_rel = entry.get("img_filename", "")
            full_img = img_dir / img_rel
            if not full_img.exists():
                print(f"⚠️ Missing image: {full_img}")
            entry["full_img_path"] = str(full_img.resolve())
            all_entries.append(entry)

    return all_entries

###############################################
# Public APIs
###############################################

def load_screenspot_dataset(
    hf_repo: Optional[str] = None,
    hf_revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    repo_type: str = "dataset",  # "dataset" or "model"
) -> List[Dict]:
    """Load ScreenSpot‑Pro either from local disk **or** directly from HuggingFace Hub.

    Parameters
    ----------
    hf_repo : str | None
        Repository id on the Hub, e.g. ``"Org/ScreenSpot-Pro"``. Required when ``base_dir`` is None.
    hf_revision : str | None
        Branch / tag / commit hash. Default hub HEAD if None.
    token : str | None
        Access token for private repos; ignored for public.
    cache_dir : str | None
        Custom cache dir for ``snapshot_download``.

    Returns
    -------
    List[Dict]
        Parsed annotations; each dict includes ``full_img_path`` absolute path.
    """
    # 1. Determine root directory
    token = os.getenv("HUGGINGFACE_TOKEN") if token is None else token
    if not hf_repo:
        raise ValueError("hf_repo must be specified.")
    print(f"📥 Downloading ScreenSpot-Pro from HF Hub: {hf_repo} …")
    root = Path(
        snapshot_download(
            repo_id=hf_repo,
            repo_type=repo_type,
            revision=hf_revision,
            token=token,
            cache_dir=cache_dir,
            allow_patterns=["annotations/*", "images/*"],
        )
    )

    # 2. Validate structure
    ann_dir, img_dir = root / "annotations", root / "images"
    if not ann_dir.is_dir() or not img_dir.is_dir():
        raise RuntimeError(f"Dataset dir {root} lacks annotations/ or images/ subfolder.")

    print(f"🔍 Parsing annotations under {ann_dir} …")
    return _collect_entries(ann_dir, img_dir)


def convert_to_qa_format(
    src_root: str,
    save_path: str,
    *,
    question_key: str = "instruction",
    answer_from_bbox: bool = True,
    image_to_base64: bool = True,
) -> None:
    """Convert ScreenSpot‑Pro annotations into the required QA JSON structure.

    The output JSON looks like::

        {
            "image":   ["<base64>", ...],
            "width":   [3840, ...],
            "height":  [2160, ...],
            "question":["...",   ...],
            "answer":  ["1774,1586,2113,1618", ...]
        }

    Parameters
    ----------
    src_root : str
        Local ScreenSpot‑Pro root (already downloaded) *or* path returned by
        :func:`load_screenspot_dataset` when used with HF mode.
    save_path : str
        Where to write the converted JSON file.
    question_key : str, default "instruction"
        Which original field becomes *question* (e.g. "instruction_cn").
    answer_from_bbox : bool, default True
        If True, bbox → answer. If False, use ``instruction_cn`` as answer.
    image_to_base64 : bool, default True
        Whether to embed images as base64 strings. If False, keeps absolute file paths.
    """
    entries = load_screenspot_dataset(src_root)

    qa_dict = {
        "image": [],
        "width": [],
        "height": [],
        "question": [],
        "answer": [],
    }

    for e in entries:
        img_path = Path(e["full_img_path"])

        # 1. Handle image field
        if image_to_base64:
            with img_path.open("rb") as fp:
                qa_dict["image"].append(base64.b64encode(fp.read()).decode("ascii"))
        else:
            qa_dict["image"].append(str(img_path))

        # 2. Width / height
        if "img_size" in e and len(e["img_size"]) == 2:
            w, h = e["img_size"]
        else:
            with Image.open(img_path) as im:
                w, h = im.size
        qa_dict["width"].append(int(w))
        qa_dict["height"].append(int(h))

        # 3. Question
        qa_dict["question"].append(e.get(question_key, ""))

        # 4. Answer
        if answer_from_bbox:
            bbox = e.get("bbox", [])
            qa_dict["answer"].append(",".join(map(str, bbox)))
        else:
            qa_dict["answer"].append(e.get("instruction_cn", ""))

    # 5. Write JSON
    save_path = Path(save_path).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fp:
        json.dump(qa_dict, fp, ensure_ascii=False, indent=2)

    print(f"✔️ Converted {len(entries)} samples → {save_path}")

###############################################
# Example CLI usage
###############################################
if __name__ == "__main__":
    # 1) Download from HF if local dir not provided
    # dataset_root = load_screenspot_dataset(
    #     base_dir=None,
    #     hf_repo="Org/ScreenSpot-Pro",
    #     cache_dir="~/.cache/screenspot",
    # )

    # 2) Or just use an existing local copy
    dataset_root = "likaixin/ScreenSpot-Pro"  # adjust this path

    convert_to_qa_format(
        src_root=dataset_root,
        save_path="/mnt/data1/home/lei00126/datasets/screenspot_qa.json",
        question_key="instruction",
        answer_from_bbox=True,
        image_to_base64=True,
    )
