"""
Main file for evaluation. This is separated from training, meaning this script
only reads log data from previously conducted experiments.
"""
from pathlib import Path
from src.helper.evaluation import Evaluator

combined_prompts_path = Path("data/new_combined_sentences.txt")
uncombined_prompts_path = Path("data/new_uncombined_sentences.txt")

data_dir=Path("logs/evaluation_dir/evaluation_f") # select folder with log files 
baseline_dir=Path("logs/evaluation_dir/evaluation_a") # select baseline folder with log files

eval = Evaluator(data_dir, baseline_dir, device="cuda:0", idx_min=0, idx_max=249, combined_prompts_path=combined_prompts_path, uncombined_prompts_path=uncombined_prompts_path)

# Filling the dataframes, this can take several minutes
eval.fill_dfs()

# Printing results
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