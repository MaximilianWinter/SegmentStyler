import os
import torch
import clip
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List

def get_probs(img_path:str, text_prompts:List[str], verbose=True) -> np.ndarray:
    """
    Returns similarity scores between list of texts and image.
    """
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        if verbose:
            print(f'File not found, treating given path as url. Path given: `{img_path}`')
        img = Image.open(requests.get(img_path, stream=True).raw)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize(text_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs


def wrap_dict(probas:np.ndarray, text_prompts:List[str]) -> dict:
    """Wraps similarity scores into dictionary of corresponding text prompts"""
    return {t: p for t,p  in zip(text_prompts, probas.squeeze().tolist())}


def get_clip_df(img_paths:List[str], text_prompts:List[str], verbose:bool=False) -> pd.DataFrame:
    """
    For a set of images and text prompts, it returns a pd.DataFrame with the similiarities scores.
    """
    data = {}
    for path_ in tqdm(img_paths):
        probas = get_probs(path_, text_prompts, verbose=verbose)
        data[path_] = wrap_dict(probas, text_prompts)

    df = pd.DataFrame(data).T
    
    return df
