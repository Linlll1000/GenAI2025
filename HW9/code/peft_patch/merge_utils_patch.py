# =============================================================================
# 此檔案的程式碼要插入到：
#   peft/src/peft/utils/merge_utils.py
#
# 插入位置：在現有的 ties() 函數之後，檔案結尾之前。
# =============================================================================

# 同時需要確認 merge_utils.py 的 import 區塊已有以下內容（通常原本就有）：
#   from typing import List, Literal, Optional
#   import torch

# -----------------------------------------------------------------------------
# 以下三個函數貼入 merge_utils.py
# -----------------------------------------------------------------------------

def sce_mask(
    task_tensors: "torch.Tensor",   # shape: (T, ...)，T = task 數量
    density: float,
    mask_dtype: "Optional[torch.dtype]" = None,
) -> "torch.Tensor":
    """
    SCE 的 S（Select）步驟。

    計算所有 task tensor 在 task 維度（dim=0）上的 variance，
    保留 variance 最大的 top-k 元素（k = nonzero_count * density），
    回傳一個與 task_tensors 形狀相同（但去掉 task 維度，即 [...]）的二值 mask。

    差異於 TIES 的 magnitude_prune（各自對每個 task vector 剪枝），
    SCE 是「跨所有 task vector」一起看 variance 來決定保留哪些位置。
    """
    import torch

    if density <= 0:
        # 全部遮蔽
        return torch.zeros_like(task_tensors[0], dtype=mask_dtype)
    if density >= 1:
        # 全部保留
        return torch.ones_like(task_tensors[0], dtype=mask_dtype)

    # 計算每個參數位置在 T 個 task 上的 variance（unbiased=False 和論文一致）
    var = torch.var(task_tensors, dim=0, unbiased=False)  # shape: (...)

    # 只考慮 variance 非零的位置來計算 k，避免 density 被零元素稀釋
    nonzero = torch.count_nonzero(var)
    k = int(nonzero.item() * density)
    if k == 0:
        return torch.zeros_like(var, dtype=mask_dtype)

    # 選出 variance 絕對值最大的 top-k 個位置
    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)

    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask


def sce_weight(task_tensors: "torch.Tensor") -> "torch.Tensor":
    """
    SCE 的 C（Calculate coefficient）步驟。

    對每個 task vector，計算其所有元素的平方均值（energy），
    再對所有 task 做 normalization，得到每個 task 的合併係數。

    公式：η_{j,m} = Σ δ²_{j,m} / (Σ_j Σ δ²_{j,m})

    Args:
        task_tensors: shape (T, ...) 的 tensor，T 為 task 數量

    Returns:
        shape (T,) 的 weight tensor，和為 1
    """
    import torch

    # 對每個 task，計算除 task 維度以外所有維度的平方均值
    weights = torch.mean(
        task_tensors ** 2,
        dim=list(range(1, task_tensors.dim()))
    )  # shape: (T,)

    weight_sum = torch.sum(weights).item()

    # Edge case：所有 task vector 都是零向量時，退化為均勻權重
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]

    return weights / weight_sum


def sce(
    task_tensors: "List[torch.Tensor]",
    density: float,
    majority_sign_method: "Literal['total', 'frequency']" = "total",
) -> "torch.Tensor":
    """
    SCE（Select-Calculate-Erase）合併演算法。
    來自論文：FuseChat: Knowledge fusion of chat models (arXiv:2408.07990)

    步驟：
      S（Select）  ── 跨所有 task vector 選 top-k variance 的位置
      C（Calculate）── 計算每個 task 的平方能量係數（η）
      E（Erase）   ── 消除少數方向的元素（sign consensus mask）
      Merge        ── 加權求和

    Args:
        task_tensors: 各 task 的 task vector（list of Tensor，每個 shape 相同）
        density:      保留的比例（0~1），用於 S 步驟
        majority_sign_method: Elect Sign 方法，"total" 或 "frequency"

    Returns:
        合併後的單一 Tensor，shape 與輸入 task_tensors 元素相同
    """
    import torch

    # 將 list 疊成 (T, ...) 的 tensor
    tv = torch.stack(task_tensors, dim=0)

    # S：根據跨 task 的 variance 選出重要位置
    if density < 1.0:
        mask = sce_mask(tv, density, mask_dtype=tv.dtype)
        # mask shape: (...)，需 broadcast 到 (T, ...)
        tv = tv * mask.unsqueeze(0)

    # E：消除少數符號方向的元素
    # sign_consensus_mask 已存在於 merge_utils.py（TIES 也使用它）
    erase_mask = sign_consensus_mask(tv, method=majority_sign_method, mask_dtype=tv.dtype)
    tv = tv * erase_mask

    # C：計算各 task 的能量係數
    tv_weights = sce_weight(tv)  # shape: (T,)

    # Merge：加權求和
    # tv_weights reshape 成 (T, 1, 1, ...) 以對齊 (T, ...) 的 tv
    weight_shape = [-1] + [1] * (tv.dim() - 1)
    merged = torch.sum(tv * tv_weights.view(weight_shape), dim=0)

    return merged
