#%%
from pathlib import Path
from src.helper.evaluation import Evaluator

exclusion_list = [3, 4, 7, 8, 9, 10, 11, 14, 18, 31, 33, 35, 39, 42, 45, 48, 56, 59, 60, 63, 66, 67, 68]
data_dir=Path("logs/2023-01-25/evaluation_no_blend/")
baseline_dir=Path("logs/2023-01-26/evaluation_baseline/")
#%%
eval = Evaluator(data_dir, baseline_dir, exclude_ids=exclusion_list, idx_max=69)
eval.fill_dfs() # This will take a few mins.
#%%
eval.get_avg_cosine_sim_combined()
# %%
eval.get_avg_cosine_sim_part_prompts()
# %%
eval.get_comparison_metrics()
# %%
eval.get_r_precision(R=10)
# %%
eval.get_r_precision_part_prompt(R=5)
# %%
