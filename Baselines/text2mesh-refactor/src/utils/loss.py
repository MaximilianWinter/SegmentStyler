import torch

def default_loss(encoded_renders_dict, encoded_text, clipavg=None):
    losses_dict = {}

    for key, encoded_renders_list in encoded_renders_dict.items():
        loss = 0.0
        if key == "no_augs":
            for encoded_renders in encoded_renders_list:
                # shouldn't there be a minus sign?
                loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

        elif key in ["augs", "norm_augs", "geo"]:
            for encoded_renders in encoded_renders_list["augs"]:
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