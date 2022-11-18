import torch

def default_loss(out_dict, encoded_text, args, clipavg=None):
    losses_dict = {}
    encoded_renders_dict = out_dict["encoded_renders"]

    for key, encoded_renders_list in encoded_renders_dict.items():
        loss = 0.0
        if key == "no_augs":
            for encoded_renders in encoded_renders_list:
                # shouldn't there be a minus sign?
                loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

        elif key in ["augs", "norm_augs", "geo"]:
            for encoded_renders in encoded_renders_list:
                if (clipavg == "view") or (clipavg is None) or (key == "geo"):
                    if encoded_text.shape[0] > 1:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                    torch.mean(encoded_text, dim=0), dim=0)
                    else:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                    encoded_text)
                else:
                    loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
        losses_dict[key] = loss

    return losses_dict

def penalizing_loss(out_dict, encoded_text, args, clipavg=None):
    losses_dict = default_loss(out_dict, encoded_text, args, clipavg)
    not_yet = True
    for key, loss in losses_dict.items():
        if isinstance(loss, torch.Tensor):
            if not_yet: # this flag makes sure that the penalizing term is added to the loss only once
                not_yet = False
                losses_dict[key] = loss + args.reg_lambda*out_dict["color_reg"]
    
    if not_yet:
        raise ValueError("CAUTION: Penalizing term not added.")

    return losses_dict