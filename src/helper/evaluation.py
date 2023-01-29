import numpy as np
import trimesh
from pathlib import Path
import torch
import kaolin as kal
import clip
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from src.submodels.render import Renderer
from src.utils.render import get_render_resolution
from src.data.mesh import Mesh


class Evaluator:
    def __init__(
        self,
        data_dir,
        baseline_dir,
        exclude_ids=[],
        clipmodel="ViT-L/14@336px",
        device="cpu",
        idx_min=0,
        idx_max=41,
    ) -> None:
        """
        implemented metrics:
        1) cosine similarity, combined prompt
        2) cosine similarity, part prompts
        2a) avg of part prompts
        2b) min of part prompts
        3) retrieval precision (R=5)

        4) direct baseline comparison
        4a) using combined prompt
        4b) using avg part prompts
        4c) using min part prompts
        """
        res = get_render_resolution(clipmodel)
        self.renderer = Renderer(dim=(res, res))

        self.clip_model, _ = clip.load(clipmodel, device=device)

        clip_normalizer = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        self.clip_transform = transforms.Compose(
            [transforms.Resize((res, res)), clip_normalizer]
        )

        self.all_combined_prompts = (
            Path("data/combined_sentences.txt").read_text().splitlines()
        )
        self.all_uncombined_prompts = (
            Path("data/uncombined_sentences.txt").read_text().splitlines()
        )

        self.data_dir = data_dir
        self.baseline_dir = baseline_dir

        self.cosine_sims = None  # pandas.Dataframe, rows=ids, cols=prompts
        self.cosine_sims_baseline = None

        self.id_to_correct_combined_prompt = {}
        self.correct_combined_prompt_to_id = {}
        self.id_to_correct_part_prompts = {}
        # self.correct_part_prompts_to_id = {}

        self.exclude_ids = exclude_ids

        self.idx_min = idx_min
        self.idx_max = idx_max

        self.device = device

    def fill_dfs(self):
        self.cosine_sims = self.get_df(self.data_dir)
        self.cosine_sims_baseline = self.get_df(self.baseline_dir)

    def get_df(self, data_dir):
        columns = [(True, prompt) for prompt in self.all_combined_prompts]
        columns.extend(
            [
                (False, "a chair with a " + prompt)
                for prompt in np.unique(self.all_uncombined_prompts)
            ]
        )
        data = []
        indices = [
            i
            for i in range(self.idx_min, self.idx_max + 1)
            if i not in self.exclude_ids
        ]
        for idx in tqdm(indices):
            cosine_similarities = []

            base_path = data_dir.joinpath(f"version_{idx}")
            real_combined_prompt, real_part_prompts = Evaluator.get_real_prompts(
                base_path
            )
            self.add_to_dicts(idx, real_combined_prompt, real_part_prompts)

            encoded_img = self.get_encoded_img(base_path)

            for _, prompt in columns:
                prompt_token = clip.tokenize([prompt]).to(self.device)
                encoded_prompt = self.clip_model.encode_text(prompt_token)
                cosine_sim = torch.cosine_similarity(encoded_img, encoded_prompt).item()
                cosine_similarities.append(cosine_sim)
            data.append(cosine_similarities)

        df = pd.DataFrame(
            data,
            index=indices,
            columns=pd.MultiIndex.from_tuples(columns, names=["combined", "prompts"]),
        )

        return df

    def add_to_dicts(self, idx, real_combined_prompt, real_part_prompts):
        self.id_to_correct_combined_prompt[idx] = real_combined_prompt
        self.correct_combined_prompt_to_id[real_combined_prompt] = idx
        self.id_to_correct_part_prompts[idx] = real_part_prompts
        # self.correct_part_prompts_to_id[real_part_prompts] = idx

    @staticmethod
    def get_real_prompts(base_path):
        prompts = base_path.joinpath("prompts.txt").read_text().splitlines()
        if len(prompts) == 4:
            combined_prompt = "a chair with a "
            for i, prompt in enumerate(prompts):
                combined_prompt += prompt + ", a " if i < 3 else prompt
            part_prompts = prompts
        else:
            combined_prompt = prompts[0]
            part_prompts = [
                prompt.removeprefix("a chair with a ")
                if "a chair with a " in prompt
                else prompt.removeprefix("a ")
                for prompt in combined_prompt.split(", ")
            ]

        # add "a chair with a " to all part prompts
        part_prompts = ["a chair with a " + prompt for prompt in part_prompts]

        return combined_prompt, part_prompts

    def get_encoded_img(self, base_path):

        tri_mesh = trimesh.load(base_path.joinpath("final_mesh.obj"))
        mesh = Mesh(str(base_path.joinpath("final_mesh.obj")), use_trimesh=True)
        rgb = (
            torch.tensor(tri_mesh.visual.vertex_colors[:, :3] / 255.0)
            .float()
            .to(mesh.vertices.device)
        )
        mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            rgb.unsqueeze(0), mesh.faces
        )
        mesh.vertex_colors = rgb

        img = self.renderer.render_single_view(
            mesh, elev=np.pi / 6, azim=-np.pi / 4, show=False
        )
        clip_image = self.clip_transform(img)
        encoded_img = self.clip_model.encode_image(clip_image.to(self.device))

        return encoded_img

    def get_avg_cosine_sim_combined(self, verbose=True):
        cosine_sims_new = []
        cosine_sims_baseline = []
        for i in self.cosine_sims.index:
            cosine_sims_new.append(
                self.cosine_sims[True].at[i, self.id_to_correct_combined_prompt[i]]
            )
            cosine_sims_baseline.append(
                self.cosine_sims_baseline[True].at[
                    i, self.id_to_correct_combined_prompt[i]
                ]
            )

        if verbose:
            print("\n ### NEW ###")
            print(
                f"cosine sim: {np.nanmean(cosine_sims_new):.3f} +- {np.nanstd(cosine_sims_new):.3f}"
            )
            print("\n### BASELINE ###")
            print(
                f"cosine sim: {np.nanmean(cosine_sims_baseline):.3f} +- {np.nanstd(cosine_sims_baseline):.3f}"
            )
            return

        return cosine_sims_new, cosine_sims_baseline

    def get_avg_cosine_sim_part_prompts(self, verbose=True):
        cosine_sims_new_avg = []
        cosine_sims_new_min = []
        cosine_sims_baseline_avg = []
        cosine_sims_baseline_min = []
        for i in self.cosine_sims.index:
            vals = [
                self.cosine_sims[False].at[i, part_prompt]
                for part_prompt in self.id_to_correct_part_prompts[i]
            ]
            cosine_sims_new_avg.append(np.mean(vals))
            cosine_sims_new_min.append(np.min(vals))

            vals = [
                self.cosine_sims_baseline[False].at[i, part_prompt]
                for part_prompt in self.id_to_correct_part_prompts[i]
            ]
            cosine_sims_baseline_avg.append(np.mean(vals))
            cosine_sims_baseline_min.append(np.min(vals))

        if verbose:
            print("\n ### NEW ###")
            print(
                f"cosine sim (avg): {np.nanmean(cosine_sims_new_avg):.3f} +- {np.nanstd(cosine_sims_new_avg):.3f}"
            )
            print(
                f"cosine sim (min): {np.nanmean(cosine_sims_new_min):.3f} +- {np.nanstd(cosine_sims_new_min):.3f}"
            )

            print("\n### BASELINE ###")
            print(
                f"cosine sim (avg): {np.nanmean(cosine_sims_baseline_avg):.3f} +- {np.nanstd(cosine_sims_baseline_avg):.3f}"
            )
            print(
                f"cosine sim (min): {np.nanmean(cosine_sims_baseline_min):.3f} +- {np.nanstd(cosine_sims_baseline_min):.3f}"
            )
            return

        return (
            cosine_sims_new_avg,
            cosine_sims_new_min,
            cosine_sims_baseline_avg,
            cosine_sims_baseline_min,
        )

    def get_r_precision(self, R=5):
        # TODO: part prompts
        r_prec_new = []
        r_prec_baseline = []
        for i in self.cosine_sims.index:
            real_prompt = self.id_to_correct_combined_prompt[i]
            if (
                real_prompt
                in self.cosine_sims[True].columns[
                    self.cosine_sims[True].loc[i].argsort()[-R:]
                ]
            ):
                r_prec_new.append(1)
            else:
                r_prec_new.append(0)

            if (
                real_prompt
                in self.cosine_sims_baseline[True].columns[
                    self.cosine_sims_baseline[True].loc[i].argsort()[-R:]
                ]
            ):
                r_prec_baseline.append(1)
            else:
                r_prec_baseline.append(0)

        print("\n ### NEW ###")
        print(f"R-precision (R={R}): {np.sum(r_prec_new)/len(r_prec_new)}")

        print("\n ### BASELINE ###")
        print(f"R-precision (R={R}): {np.sum(r_prec_baseline)/len(r_prec_baseline)}")

    def get_r_precision_part_prompt(self, R=5):
        # TODO: this is a non-usual implementation of the R-precision
        r_prec_new = []
        r_prec_baseline = []
        for i in self.cosine_sims.index:
            real_prompts = self.id_to_correct_part_prompts[i]
            for real_prompt in real_prompts:
                is_inside = False
                if (
                    real_prompt
                    in self.cosine_sims[False].columns[
                        self.cosine_sims[False].loc[i].argsort()[-R:]
                    ]
                ):
                    is_inside = True
                    break
            if is_inside:
                r_prec_new.append(1)
            else:
                r_prec_new.append(0)

            for real_prompt in real_prompts:
                is_inside = False
                if (
                    real_prompt
                    in self.cosine_sims_baseline[False].columns[
                        self.cosine_sims_baseline[False].loc[i].argsort()[-R:]
                    ]
                ):
                    is_inside = True
                    break
            if is_inside:
                r_prec_baseline.append(1)
            else:
                r_prec_baseline.append(0)

        print("\n ### NEW ###")
        print(f"R-precision (R={R}): {np.sum(r_prec_new)/len(r_prec_new)}")

        print("\n ### BASELINE ###")
        print(f"R-precision (R={R}): {np.sum(r_prec_baseline)/len(r_prec_baseline)}")

    def get_comparison_metrics(self):
        # combined prompts
        new, baseline = self.get_avg_cosine_sim_combined(verbose=False)
        mask = np.array(new) >= np.array(baseline)
        val = np.sum(mask) / len(mask)

        print("### COMBINED ###")
        print(f"avg: {val}")

        # per part prompts
        (
            new_avg,
            new_min,
            baseline_avg,
            baseline_min,
        ) = self.get_avg_cosine_sim_part_prompts(verbose=False)

        mask = np.array(new_avg) >= np.array(baseline_avg)
        val_avg = np.sum(mask) / len(mask)

        mask = np.array(new_min) >= np.array(baseline_min)
        val_min = np.sum(mask) / len(mask)

        print("### PER PART ###")
        print(f"avg: {val_avg}")
        print(f"min: {val_min}")