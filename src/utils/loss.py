import torch

def default_loss(out_dict, encoded_text_prompts, args, clipavg=None):
    
    encoded_renders_dict = out_dict["encoded_renders"]
    losses_dict_per_prompt = {}
    
    for i, encoded_text_prompt in enumerate(encoded_text_prompts):
        losses_dict = {}
        for key, encoded_renders_list in encoded_renders_dict.items():
            loss = 0.0
            if key == "no_augs":
                for encoded_renders in encoded_renders_list:
                    # shouldn't there be a minus sign?
                    loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text_prompt))

            elif key in ["augs", "norm_augs", "geo"]:
                for encoded_renders in encoded_renders_list:
                    if (clipavg == "view") or (clipavg is None) or (key == "geo"):
                        if encoded_text_prompt.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                        torch.mean(encoded_text_prompt, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                        encoded_text_prompt)
                    else:
                        loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text_prompt))
            losses_dict[key] = loss

        losses_dict_per_prompt[args.prompts[i]] = losses_dict

    return losses_dict_per_prompt

def penalizing_loss(out_dict, encoded_text_prompts, args, clipavg=None):
    losses_dict_per_prompt = default_loss(out_dict, encoded_text_prompts, args, clipavg)
    for prompt, losses_dict in losses_dict_per_prompt.items():
        not_yet = True
        for key, loss in losses_dict.items():
            if isinstance(loss, torch.Tensor):
                if not_yet: # this flag makes sure that the penalizing term is added to the loss only once
                    not_yet = False
                    losses_dict[key] = loss + args.reg_lambda*out_dict["color_reg"][prompt]
        
        if not_yet:
            raise ValueError("CAUTION: Penalizing term not added.")

    return losses_dict_per_prompt