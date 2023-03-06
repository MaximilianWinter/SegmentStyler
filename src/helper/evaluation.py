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
from src.data.partglot_data import PartGlotData

class EvalVersionConverter:
    
    def __init__(self):
        self.mapping = {}
        samples = Path("data/samples.txt").read_text().splitlines()
        combined_prompts = Path("data/combined_sentences.txt").read_text().splitlines()
        uncombined_prompts = Path("data/uncombined_sentences.txt").read_text().splitlines()
        
        version_id = 0
        for i, sample_id in enumerate(samples):
            for j, combined_prompt in enumerate(combined_prompts):
                uncombined_list = uncombined_prompts[4*j:4*(j+1)]
                self.mapping[version_id] = (int(sample_id), uncombined_list, combined_prompt)
                version_id += 1

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
        combined_prompts_path = Path("data/combined_sentences.txt"),
        uncombined_prompts_path = Path("data/uncombined_sentences.txt")

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
            combined_prompts_path.read_text().splitlines()
        )
        self.all_uncombined_prompts = (
            uncombined_prompts_path.read_text().splitlines()
        )

        self.data_dir = data_dir
        self.baseline_dir = baseline_dir

        self.cosine_sims = None  # pandas.Dataframe, rows=ids, cols=prompts
        self.cosine_sims_baseline = None
        self.cosine_sims_part_imgs = None
        self.cosine_sims_part_imgs_baseline = None

        self.id_to_correct_combined_prompt = {}
        self.correct_combined_prompt_to_id = {}
        self.id_to_correct_part_prompts = {}
        self.id_to_short_part_prompts = {}
        # self.correct_part_prompts_to_id = {}

        self.exclude_ids = exclude_ids

        self.idx_min = idx_min
        self.idx_max = idx_max

        self.device = device

        self.pg_data = PartGlotData(None, return_keys=["mesh", "gt_labels"])

    def fill_dfs(self):
        self.cosine_sims = self.get_df(self.data_dir)
        self.cosine_sims_baseline = self.get_df(self.baseline_dir)
        self.cosine_sims_part_imgs = self.get_df_with_part_imgs(self.data_dir)
        self.cosine_sims_part_imgs_baseline = self.get_df_with_part_imgs(self.baseline_dir)
        self._fill_dicts(self.data_dir)

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

    def get_df_with_part_imgs(self, data_dir):
        # now we use the original uncombined prompts
        # without adding "a chair with a " as prefix
        columns = [(False, prompt) for prompt in np.unique(self.all_uncombined_prompts)]
        data = []
        indices = [
            i
            for i in range(self.idx_min, self.idx_max + 1)
            if i not in self.exclude_ids
        ]
        for idx in tqdm(indices):
            cosine_similarities = []
            base_path = data_dir.joinpath(f"version_{idx}")

            encoded_imgs = self.get_encoded_part_imgs(base_path)

            for _, prompt in columns:
                parts_in_prompt = [part for part in encoded_imgs.keys() if part in prompt]
                if len(parts_in_prompt) != 1:
                    raise ValueError("There are multiple parts in the prompt.")
                else:
                    part = parts_in_prompt[0]
                encoded_img = encoded_imgs[part]
                if encoded_img is not None:
                    prompt_token = clip.tokenize([prompt]).to(self.device)
                    encoded_prompt = self.clip_model.encode_text(prompt_token)
                    cosine_sim = torch.cosine_similarity(encoded_img, encoded_prompt).item()
                else:
                    cosine_sim = torch.nan
                cosine_similarities.append(cosine_sim)
            data.append(cosine_similarities)
        
        df = pd.DataFrame(
            data,
            index=indices,
            columns=pd.MultiIndex.from_tuples(columns, names=["combined", "prompts"]),
        )
        return df

    def _fill_dicts(self, data_dir):
        indices = [
            i
            for i in range(self.idx_min, self.idx_max + 1)
            if i not in self.exclude_ids
        ]
        for idx in indices:
            base_path = data_dir.joinpath(f"version_{idx}")
            real_combined_prompt, real_part_prompts, short_part_prompts = Evaluator.get_real_prompts(base_path)
            self.id_to_correct_combined_prompt[idx] = real_combined_prompt
            self.correct_combined_prompt_to_id[real_combined_prompt] = idx
            self.id_to_correct_part_prompts[idx] = real_part_prompts
            self.id_to_short_part_prompts[idx] = short_part_prompts

    def get_encoded_img(self, base_path, show=False):

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
            mesh, elev=np.pi / 6, azim=-np.pi / 4, show=show, lighting=False,
        )
        clip_image = self.clip_transform(img)
        encoded_img = self.clip_model.encode_image(clip_image.to(self.device))

        return encoded_img

    def get_encoded_part_imgs(self, base_path, show=False):
        tri_mesh = trimesh.load(base_path.joinpath("final_mesh.obj"))
        sample_id = int(base_path.joinpath("sample_id.txt").read_text())
        gt_labels = self.pg_data[sample_id]["gt_labels"]
        encoded_imgs = {}
        for part, part_id in self.pg_data.label_mapping.items():
            mask = gt_labels == part_id
            if np.sum(mask) > 100: # for small numbers, kal raises an Error when rendering, TODO: make more elegant
                masked_faces = tri_mesh.vertex_faces[mask]
                unique_faces = np.unique(masked_faces[masked_faces != -1])
                sub_tri_mesh = tri_mesh.submesh([unique_faces], append=True)
                submesh = Mesh(sub_tri_mesh)
                rgb = (
                torch.tensor(sub_tri_mesh.visual.vertex_colors[:, :3] / 255.0)
                .float()
                .to(submesh.vertices.device)
                )
                submesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
                    rgb.unsqueeze(0), submesh.faces
                )
                submesh.vertex_colors = rgb

                img = self.renderer.render_single_view(
                submesh, elev=np.pi / 6, azim=-np.pi / 4, show=show, lighting=False
                )
                clip_image = self.clip_transform(img)
                encoded_img = self.clip_model.encode_image(clip_image.to(self.device))
                encoded_imgs[part] = encoded_img
            else:
                encoded_imgs[part] = None

        return encoded_imgs

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
        short_part_prompts = part_prompts
        part_prompts = ["a chair with a " + prompt for prompt in part_prompts]

        return combined_prompt, part_prompts, short_part_prompts

    @staticmethod
    def get_avg_cosine_sim(df, id_to_prompt, combined_prompts=True, verbose=True, operation="mean"):
        cosine_sims = []
        for i in df.index:
            if combined_prompts:
                cosine_sims.append(df[combined_prompts].at[i, id_to_prompt[i]])
            else:
                vals = [df[combined_prompts].at[i, part_prompt] for part_prompt in id_to_prompt[i]]
                if operation == "mean":
                    cosine_sims.append(np.nanmean(vals))
                elif operation == "min":
                    cosine_sims.append(np.nanmin(vals))
                elif operation == "max":
                    cosine_sims.append(np.nanmax(vals))
                else:
                    raise ValueError("operation not known")

        if verbose:
            print(f"cosine sim: {np.nanmean(cosine_sims):.3f} +- {np.nanstd(cosine_sims):.3f}")
            return

        return cosine_sims

    @staticmethod
    def get_r_precision(df, id_to_prompt, combined_prompts=True, verbose=True, R=5):
        r_prec = []
        for i in df.index:
            top_R_prompts = df[combined_prompts].columns[df[combined_prompts].loc[i].argsort()[-R:]]
            if combined_prompts:
                real_prompt = id_to_prompt[i]
                criterion = real_prompt in top_R_prompts
            else:
                real_prompts = id_to_prompt[i]
                criterion = True in [real_prompt in top_R_prompts for real_prompt in real_prompts]
            if criterion:
                r_prec.append(1)
            else:
                r_prec.append(0)

        if verbose:
            print(f"R-precision (R={R}): {np.sum(r_prec)/len(r_prec)}")
            return
        
        return r_prec

    @staticmethod
    def get_comparison_metrics(df_1, df_2, id_to_prompt, combined, verbose=True):
        cosine_sim_1 = Evaluator.get_avg_cosine_sim(df_1, id_to_prompt, combined, verbose=False)
        cosine_sim_2 = Evaluator.get_avg_cosine_sim(df_2, id_to_prompt, combined, verbose=False)
        mask = np.array(cosine_sim_1) >= np.array(cosine_sim_2)
        val = np.sum(mask) / len(mask)

        if verbose:
            print(f"Percentage of 1 reaching higher score than 2: {val}")
            return
        
        return val
        