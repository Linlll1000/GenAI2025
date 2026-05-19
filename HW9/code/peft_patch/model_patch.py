# =============================================================================
# 此檔案說明要對 peft/src/peft/tuners/lora/model.py 做的修改。
#
# 共需修改 3 處，以下用 [修改 1/2/3] 標示。
# 搜尋各 FIND 字串，在對應位置做替換。
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 【修改 1】import 區塊
# ─────────────────────────────────────────────────────────────────────────────
#
# FIND（原本的 import，通常在檔案頂端）：
#   from peft.utils.merge_utils import magnitude_prune, ties
#
# REPLACE WITH（加入 sce）：
#   from peft.utils.merge_utils import magnitude_prune, ties, sce


# ─────────────────────────────────────────────────────────────────────────────
# 【修改 2】add_weighted_adapter() 的 combination_type 合法清單
# ─────────────────────────────────────────────────────────────────────────────
#
# FIND（原本的 elif，判斷 combination_type 是否合法）：
#   elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
#
# REPLACE WITH（加入 "sce"）：
#   elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune", "sce"]:


# ─────────────────────────────────────────────────────────────────────────────
# 【修改 3】_generalized_task_arithmetic_weighted_adapter() 的 dispatch 邏輯
# ─────────────────────────────────────────────────────────────────────────────
#
# FIND（在 for 迴圈內，呼叫 ties 的那個 elif，類似這樣）：
#   elif combination_type == "ties":
#       lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
#
# REPLACE WITH（在它後面緊接著加入 sce 的 elif）：
#   elif combination_type == "ties":
#       lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
#   elif combination_type == "sce":
#       lora_deltas[i] = sce(task_tensors, density, majority_sign_method)
#
# 注意：SCE 的 C 步驟會自動計算各 task 的係數（η），
#       不使用外部傳入的 valid_weights，所以不帶 valid_weights 參數。


# =============================================================================
# 完整的 sce 函數呼叫簽名（供確認用）
#   sce(task_tensors, density, majority_sign_method) -> torch.Tensor
# =============================================================================
