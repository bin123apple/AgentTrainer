import glob
import os
from multiprocessing import Pool
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def load_shard(path: str) -> Dataset:
    return Dataset.from_parquet(path, cache_dir = '/home/uconn/.cache/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box')

def parallel_load_dataset(cache_dir: str, num_proc: int = 8) -> Dataset:
    # 1. 递归搜所有 parquet
    pattern = os.path.join(cache_dir, "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet shards found under {pattern}")

    # 2. 并行加载并显示进度条
    with Pool(processes=num_proc) as pool:
        # 使用 imap 保持顺序，imap_unordered 则可以更快一点但输出顺序会打乱
        shards_iter = pool.imap(load_shard, files)
        shards = []
        for shard in tqdm(shards_iter, total=len(files), desc="Loading shards", unit="shard"):
            shards.append(shard)

    # 3. 合并
    return concatenate_datasets(shards)

if __name__ == "__main__":
    CACHE_DIR = "/home/uconn/.cache/huggingface/datasets/datasets--osunlp--UGround-V1-Data-Box"
    ds = parallel_load_dataset(CACHE_DIR, num_proc=16)
    print(ds)
    first = ds[0]
    print("First record:\n", first)