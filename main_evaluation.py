#%%
from pathlib import Path
from src.helper.evaluation import Evaluator

exclusion_list = []#3, 4, 7, 8, 9, 10, 11, 14, 18, 31, 33, 35, 39, 42, 45, 48, 56, 59, 60, 63, 66, 67, 68]
exclusion_list = [0,1, 5, 8, 11, 20, 25, 27, 28, 30, 33, 34, 41, 44, 54, 55, 57, 61, 62, 64, 65, 75, 84, 85, 87, 91, 94, 95, 104, 105, 107, 110, 111, 113, 115, 121, 123, 125]
data_dir=Path("logs/2023-01-27/evaluation_no_blend_new")
baseline_dir=Path("logs/2023-01-27/evaluation_baseline_new")
combined_prompts_path = Path("data/new_combined_sentences.txt")
uncombined_prompts_path = Path("data/new_uncombined_sentences.txt")
#%%
print(len(exclusion_list))
#%%
eval = Evaluator(data_dir, baseline_dir, device="cuda:0", exclude_ids=exclusion_list, idx_min=0, idx_max=150, combined_prompts_path=combined_prompts_path, uncombined_prompts_path=uncombined_prompts_path)
eval.fill_dfs() # This will take a few mins.
#%%
print("### combined prompt (top=new, bottom=baseline) ###")
eval.get_avg_cosine_sim(eval.cosine_sims, eval.id_to_correct_combined_prompt)
eval.get_avg_cosine_sim(eval.cosine_sims_baseline, eval.id_to_correct_combined_prompt)

eval.get_r_precision(eval.cosine_sims, eval.id_to_correct_combined_prompt, combined_prompts=True, R=1)
eval.get_r_precision(eval.cosine_sims_baseline, eval.id_to_correct_combined_prompt, combined_prompts=True, R=1)

eval.get_comparison_metrics(eval.cosine_sims, eval.cosine_sims_baseline, eval.id_to_correct_combined_prompt, combined=True)

print("\n### part prompts, w/o segmented part imgs ###")
eval.get_avg_cosine_sim(eval.cosine_sims, eval.id_to_correct_part_prompts,combined_prompts=False)
eval.get_avg_cosine_sim(eval.cosine_sims_baseline, eval.id_to_correct_part_prompts, combined_prompts=False)

eval.get_r_precision(eval.cosine_sims, eval.id_to_correct_part_prompts, combined_prompts=False, R=1)
eval.get_r_precision(eval.cosine_sims_baseline, eval.id_to_correct_part_prompts, combined_prompts=False, R=1)

eval.get_comparison_metrics(eval.cosine_sims, eval.cosine_sims_baseline, eval.id_to_correct_part_prompts, combined=False)

print("\n### part prompts, with segmented part imgs ###")
eval.get_avg_cosine_sim(eval.cosine_sims_part_imgs, eval.id_to_short_part_prompts, combined_prompts=False)
eval.get_avg_cosine_sim(eval.cosine_sims_part_imgs_baseline, eval.id_to_short_part_prompts, combined_prompts=False)

eval.get_r_precision(eval.cosine_sims_part_imgs, eval.id_to_short_part_prompts, combined_prompts=False, R=1)
eval.get_r_precision(eval.cosine_sims_part_imgs_baseline, eval.id_to_short_part_prompts, combined_prompts=False, R=1)

eval.get_comparison_metrics(eval.cosine_sims_part_imgs, eval.cosine_sims_part_imgs_baseline, eval.id_to_short_part_prompts, combined=False)
#%%
