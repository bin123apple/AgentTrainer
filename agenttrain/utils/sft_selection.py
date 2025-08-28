import random
from collections import defaultdict
from typing import List, Tuple, Dict


def uniform_sample_correct_indices(
    correct_indices: List[int],
    all_tool_used: List[List[str]],
    max_rounds: int = 5,
    end_token: str = "END",
    seed: int = 0,
) -> List[int]:
    """
    输入:
      - correct_indices: 已判定为正确回答的样本下标
      - all_tool_used: 每条样本对应的工具调用顺序(list)，为空表示没用工具
    输出:
      - selected_indices: 各“形式桶”按最小桶大小等量抽取后的样本下标（升序）
    """
    rng = random.Random(seed)

    def canonical_form(seq: List[str]) -> Tuple[str, ...]:
        """把工具调用序列转为形式：截断 + 末尾补 END"""
        seq = [s.strip().lower() for s in seq][:max_rounds]
        if len(seq) < max_rounds:
            return tuple(seq + [end_token])
        else:
            seq[-1] = end_token
            return tuple(seq)

    # 分桶：只对正确样本分桶
    buckets: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx in correct_indices:
        form = canonical_form(all_tool_used[idx])
        buckets[form].append(idx)

    if not buckets:
        return []

    # 目标每桶数量 = 最小桶大小
    per_bucket = min(len(idxs) for idxs in buckets.values())
    print(f"Buckets formed: {len(buckets)}, sampling {per_bucket} from each.")

    # 各桶无放回等量抽样
    selected_indices: List[int] = []
    for idxs in buckets.values():
        if len(idxs) <= per_bucket:
            chosen = list(idxs)  # 刚好或更小：全取（更小不会发生，因为 per_bucket 是最小值）
        else:
            chosen = rng.sample(idxs, per_bucket)
        selected_indices.extend(chosen)

    selected_indices.sort()
    return selected_indices

import torch
from collections import Counter
from typing import List

def compute_weight_vector(
    selected_indices: List[int],
    correct_indices: List[int],
    all_tool_used: List[List[str]],
    device: torch.device,
    mode: str = "set",  # "tier" = NONE/SINGLE/MULTI, 也可以改成更细的 "set"
) -> torch.Tensor:
    """
    返回 shape=(batch_size,) 的权重向量。
    - selected_indices: 你想用于 SFT 的子集
    - correct_indices : >=1 判定正确的样本下标
    - all_tool_used   : 工具调用序列
    - mode            : 分桶方式
    """

    def bucket_key(tools: List[str]) -> str:
        s = [str(t).strip().lower() for t in tools if str(t).strip()]
        if mode == "tier":
            if len(s) == 0:
                return "NONE"
            return "SINGLE" if len(set(s)) == 1 else "MULTI"
        else:  # "set"
            return ",".join(sorted(set(s))) if s else "NONE"

    # 1. 在 correct_indices 里统计各类 case 数量
    keys_correct = [bucket_key(all_tool_used[i]) for i in correct_indices]
    cnt = Counter(keys_correct)
    correct_total = len(correct_indices)

    # 2. 先算每个类别的倒数权重（1/p），再归一化
    raw_w = {k: 1.0 / (cnt[k] / correct_total) for k in cnt}
    Z = sum(raw_w.values())
    norm_w = {k: v / Z for k, v in raw_w.items()}

    # 3. 构造 batch 权重向量
    batch_size = len(all_tool_used)
    weights = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for i in selected_indices:
        if i in correct_indices:
            k = bucket_key(all_tool_used[i])
            weights[i] = norm_w[k]
        else:
            weights[i] = 0.0  # 非正确不计权重

    return weights