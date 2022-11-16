import clip
from torchvision import transforms

from src.utils.render import get_render_resolution
from src.utils.utils import device


class CLIPWithAugs():

    def __init__(self, args):
        self.args = args
        if args.cropforward:
            self.curcrop = args.normmincrop
        else:
            self.curcrop = args.normmaxcrop
        self.iteration_step = 0
        self.res = get_render_resolution(args.clipmodel)
        self.clip_model, preprocess = clip.load(
            args.clipmodel, device, jit=args.jit)

        self.clip_normalizer, self.clip_transform, \
            self.augment_transform, self.normaugment_transform, \
            self.cropupdate, self.displaugment_transform = self.init_transforms(
                args)

    def init_transforms(self, args):
        # CLIP transform
        clip_normalizer = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        clip_transform = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            clip_normalizer
        ])

        # Augment transform
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.res, scale=(1, 1)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

        # Normaugment transform
        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self.res, scale=(self.curcrop, self.curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        cropiter = 0
        cropupdate = 0

        if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
            cropiter = round(args.n_iter / (args.cropsteps + 1))
            cropupdate = (args.maxcrop - args.mincrop) / cropiter

            if not args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        displaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.res, scale=(
                args.normmincrop, args.normmincrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

        return clip_normalizer, clip_transform, augment_transform, normaugment_transform, cropupdate, displaugment_transform

    def update_normaugment_transform(self):
        if self.args.cropsteps != 0 and self.cropupdate != 0 and self.iteration_step != 0 and self.iteration_step % self.args.cropsteps == 0:
            self.curcrop += self.cropupdate
            # print(curcrop)
            self.normaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.res, scale=(self.curcrop, self.curcrop)),
                transforms.RandomPerspective(
                    fill=1, p=0.8, distortion_scale=0.5),
                self.clip_normalizer
            ])

    def get_encoded_renders(self, rendered_images, geo_rendered_images=None):
        self.iteration_step += 1
        self.update_normaugment_transform()

        encoded_renders_dict = {"no_augs": [],
                                "augs": [], "norm_augs": [], "geo": []}

        if self.args.n_augs == 0:
            clip_image = self.clip_transform(rendered_images)
            encoded_renders = self.clip_model.encode_image(clip_image)
            encoded_renders_dict["no_augs"].append(encoded_renders)
        elif self.args.n_augs > 0:
            for _ in range(self.args.n_augs):
                augmented_image = self.augment_transform(rendered_images)
                encoded_renders = self.clip_model.encode_image(augmented_image)
                encoded_renders_dict["augs"].append(encoded_renders)

        if self.args.n_normaugs > 0:
            for _ in range(self.args.n_normaugs):
                augmented_image = self.normaugment_transform(rendered_images)
                encoded_renders = self.clip_model.encode_image(augmented_image)
                encoded_renders_dict["norm_augs"].append(encoded_renders)

            if (geo_rendered_images is not None) and self.args.geoloss:
                for _ in range(self.args.n_normaugs):
                    augmented_image = self.displaugment_transform(geo_rendered_images)
                    encoded_renders = self.clip_model.encode_image(augmented_image)
                    encoded_renders_dict["geo"].append(encoded_renders)

        return encoded_renders_dict
