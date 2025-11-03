import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import math
import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torchvision.transforms import functional as F
from torch.distributions import Normal
import torch
import numpy as np
import torch
from basicsr.utils.registry import DATASET_REGISTRY


class RandomGammaCorrection(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image, self.gamma, gain=1)


def remove_background(image):
    # the input of the image is PIL.Image form with [H,W,C]
    image = np.float32(np.array(image))
    _EPS = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + _EPS)
    image = torch.from_numpy(image)
    return image


def glod_from_folder(folder_list, index_list):
    ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
    index_dict = {}
    for i, folder_name in enumerate(folder_list):
        data_list = []
        [data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
        data_list.sort()
        index_dict[index_list[i]] = data_list
    return index_dict
def get_hov(hov):
    if hov.isnumeric():
        return (hov / 180) * math.pi
    hov = random.uniform((20 / 180) * math.pi, (100 / 180) * math.pi)  # random
    return hov

class Paired_Flare_Image_Loader(data.Dataset):
    def __init__(self, image_path, image_pair_path, transform_base, transform_flare, mask_type=None):
        self.ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
        self.data_list = []
        [self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]

        self.paths_pair = glod_from_folder([image_pair_path['dataroot_lq'], image_pair_path['dataroot_gt']],
                                           ['lq', 'gt'])
        for i in range(len(self.paths_pair['lq'])):
            self.data_list.append([self.paths_pair['lq'][i], self.paths_pair['gt'][i]])

        self.flare_dict = {}
        self.flare_list = []
        self.flare_name_list = []

        self.reflective_flag = False
        self.reflective_dict = {}
        self.reflective_list = []
        self.reflective_name_list = []

        self.light_flag = False
        self.light_dict = {}
        self.light_list = []
        self.light_name_list = []

        self.mask_type = mask_type  # It is a str which may be None,"luminance" or "color"
        self.img_size = transform_base['img_size']

        self.transform_base = transforms.Compose(
            [transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()
             ])

        self.transform_flare = transforms.Compose([
            transforms.RandomAffine(degrees=(0, 0),
                                    scale=(transform_flare['scale_min'], transform_flare['scale_max']),
                                    translate=(
                                        transform_flare['translate'] / 1440, transform_flare['translate'] / 1440),
                                    shear=(-transform_flare['shear'], transform_flare['shear'])),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        self.transform_pair = transforms.Compose(
            [transforms.Resize([self.img_size, self.img_size]),
            # [transforms.RandomCrop((self.img_size, self.img_size), pad_if_needed=True, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()])
        
        
        self.to_tensor = transforms.ToTensor()

        self.data_ratio = []
        print("Base Image Loaded with examples:", len(self.data_list))

    def __getitem__(self, index):
        # load base image
        img_path = self.data_list[index]
        gamma = np.random.uniform(1.8, 2.2)
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
        if isinstance(img_path, list):
            gt_path = img_path[1]
            lq_path = img_path[0]
            img_lq = self.to_tensor(Image.open(lq_path).convert('RGB'))
            img_gt = self.to_tensor(Image.open(gt_path).convert('RGB'))
            img_merge = torch.cat((img_gt, img_lq), dim=0)
            img_merge = self.transform_pair(img_merge)
            img_gt, img_lq = torch.split(img_merge, 3, dim=0)
            flare_img =  adjust_gamma(img_lq) -  adjust_gamma(img_gt)
            flare_img = torch.clamp(flare_img,min=0,max=1)
            return {
                "gt": img_gt,
                "flare": adjust_gamma_reverse(flare_img),
                "lq": img_lq,
                "gamma": gamma,
            }
        else:
            base_img = Image.open(img_path).convert("RGB")

            ## open the depth map which is compute before training
            img_deep = Image.open(
                "dataset/24K_res/" + img_path.split("/")[-1].replace("jpg", "png")
            ).convert("RGB")

            gamma = np.random.uniform(1.8, 2.2)
            to_tensor = transforms.ToTensor()
            # img_deep = to_tensor(img_deep)
            adjust_gamma = RandomGammaCorrection(gamma)
            adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)
            color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
            if self.transform_base is not None:
                base_img = to_tensor(base_img)
                img_deep = to_tensor(img_deep)
                base_img = adjust_gamma(base_img)
                base_img = torch.cat((base_img, img_deep), dim=0)
                base_img = self.transform_base(base_img)
                base_img, img_deep = torch.split(base_img, 3, dim=0)
                # base_img=to_tensor(base_img)
                # base_img=adjust_gamma(base_img)
                # base_img=self.transform_base(base_img)
            else:
                base_img = to_tensor(base_img)
                base_img = adjust_gamma(base_img)
            sigma_chi = 0.01 * np.random.chisquare(df=1)
            base_img = Normal(base_img, sigma_chi).sample()
            gain = np.random.uniform(0.5, 1.2)
            flare_DC_offset = np.random.uniform(-0.02, 0.02)
            base_img = gain * base_img
            base_img = torch.clamp(base_img, min=0, max=1)

            choice_dataset = random.choices(
                [i for i in range(len(self.flare_list))], self.data_ratio
            )[0]
            choice_index = random.randint(0, len(self.flare_list[choice_dataset]) - 1)

            # load flare and light source image
            if self.light_flag:
                assert len(self.flare_list) == len(
                    self.light_list
                ), "Error, number of light source and flares dataset no match!"
                for i in range(len(self.flare_list)):
                    assert len(self.flare_list[i]) == len(
                        self.light_list[i]
                    ), f"Error, number of light source and flares no match in {i} dataset!"
                flare_path = self.flare_list[choice_dataset][choice_index]
                light_path = self.light_list[choice_dataset][choice_index]
                light_img = Image.open(light_path).convert("RGB")
                light_img = to_tensor(light_img)
                light_img = adjust_gamma(light_img)
            else:
                flare_path = self.flare_list[choice_dataset][choice_index]
            flare_img = Image.open(flare_path).convert("RGB")
            if self.reflective_flag:
                reflective_path_list = self.reflective_list[choice_dataset]
                if len(reflective_path_list) != 0:
                    reflective_path = random.choice(reflective_path_list)
                    reflective_img = Image.open(reflective_path).convert("RGB")
                else:
                    reflective_img = None

            flare_img = to_tensor(flare_img)
            flare_img = adjust_gamma(flare_img)

            if self.reflective_flag and reflective_img is not None:
                reflective_img = to_tensor(reflective_img)
                reflective_img = adjust_gamma(reflective_img)
                flare_img = torch.clamp(flare_img + reflective_img, min=0, max=1)

            flare_img = remove_background(flare_img)

            ## random 1~3 light(s), user can change the number to simulate specific datasets.
            num_light = random.randint(1, 3)
            flare_img_list = []
            light_img_list = []
            merge_img_list = []
            blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))

            for i in range(num_light):

                flare_merge = torch.cat((flare_img, light_img), dim=0)
                flare_merge = self.transform_flare(flare_merge)
                merge_img_list.append(flare_merge)

                flare_img_temp, light_img_temp = torch.split(flare_merge, 3, dim=0)

                flare_img_temp = blur_transform(flare_img_temp)
                light_img_temp = blur_transform(light_img_temp)
                # flare_img=flare_img+flare_DC_offset
                flare_img_temp = torch.clamp(flare_img_temp, min=0, max=1)
                light_img_temp = torch.clamp(light_img_temp, min=0, max=1)

                flare_img_list.append(flare_img_temp)
                light_img_list.append(light_img_temp)
            distance_list = []
            cos_theta_list = []
            dex_dey = []
            if num_light != 1:
                f_fang = 0
                average_dis = img_deep[0].mean(axis=0).mean()
                for i in range(num_light):
                    flare_img_temp = flare_img_list[i]
                    light_img_temp = light_img_list[i]

                    ## obtain the light area
                    non_zero_indices = (light_img_temp[0] > 0.97).nonzero()

                    delta_x = non_zero_indices[:, 0].sum() / len(
                        non_zero_indices[:, 0]
                    ) - int(light_img_temp[0].size()[0] / 2)
                    delta_y = non_zero_indices[:, 1].sum() / len(
                        non_zero_indices[:, 1]
                    ) - int(light_img_temp[0].size()[0] / 2)
                    delta_x = torch.where(torch.isnan(delta_x), torch.tensor(0.0), delta_x)
                    delta_y = torch.where(torch.isnan(delta_y), torch.tensor(0.0), delta_x)

                    dex_dey.append(
                        (delta_x * delta_x + delta_y * delta_y)
                        / (
                            int(light_img_temp[0].size()[0] / 2)
                            * int(light_img_temp[0].size()[0] / 2)
                        )
                    )

                    sum_values = img_deep[0][
                        non_zero_indices[:, 0], non_zero_indices[:, 1]
                    ].sum()

                    num = non_zero_indices.size(0)

                    if num == 0 or sum_values == 0:
                        distance_list.append(average_dis)
                        continue
                    distance = (sum_values / num).item()
                    ## get the depth of the light source
                    distance_list.append(distance)

                ### Using different field of view angles
                hov = get_hov("random")  ## Optional numbers or 'random'
                # hov = get_hov('20')  ## degrees

                ## get the incident angle of the light rays
                for i in range(num_light):
                    cos_theta = 1 / math.sqrt(
                        1 + dex_dey[i] * math.tan(hov) * math.tan(hov)
                    )
                    cos_theta_list.append(cos_theta)
                cos_theta_list = np.nan_to_num(cos_theta_list, nan=1)

            ## Brightness Adjustment
            scale_list = [
                distance_list[i] * distance_list[i] * cos_theta_list[i]
                for i in range(len(distance_list))
            ]
            # if len(scale_list) > 1:
            #     scale_list = normalize_to_range(scale_list)
            if num_light != 1:
                flare_img_list = [
                    flare_img_list[i] * scale_list[i] for i in range(num_light)
                ]
                light_img_list = [
                    light_img_list[i] * scale_list[i] for i in range(num_light)
                ]
            merge_img = base_img.clone()
            del cos_theta_list
            del dex_dey
            for i in range(len(flare_img_list)):

                # merge image
                merge_img += flare_img_list[i]

                base_img += light_img_list[i]

            flare_reduce_light_list = [
                torch.clamp((flare_img_list[i] - light_img_list[i]), min=0, max=1)
                for i in range(len(flare_img_list))
            ]

            flare_img = torch.clamp(sum(flare_reduce_light_list), min=0, max=1)

            merge_img = torch.clamp(merge_img, min=0, max=1)
            base_img = torch.clamp(base_img, min=0, max=1)
            return {
                "gt": adjust_gamma_reverse(base_img),
                "flare": adjust_gamma_reverse(flare_img),
                "lq": adjust_gamma_reverse(merge_img),
                "gamma": gamma,
            }

    def __len__(self):
        return len(self.data_list)

    def load_scattering_flare(self, flare_name, flare_path):
        flare_list = []
        [flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
        flare_list = sorted(flare_list)
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name] = flare_list
        self.flare_list.append(flare_list)
        len_flare_list = len(self.flare_dict[flare_name])
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:", flare_name, " is loaded successfully with examples", str(len_flare_list))
        print("Now we have", len(self.flare_list), 'scattering flare images')

    def load_light_source(self, light_name, light_path):
        # The number of the light source images should match the number of scattering flares
        light_list = []
        [light_list.extend(glob.glob(light_path + '/*.' + e)) for e in self.ext]
        light_list = sorted(light_list)
        self.flare_name_list.append(light_name)
        self.light_dict[light_name] = light_list
        self.light_list.append(light_list)
        len_light_list = len(self.light_dict[light_name])

        if len_light_list == 0:
            print("ERROR: Light Source images are not loaded properly")
        else:
            self.light_flag = True
            print("Light Source Image:", light_name, " is loaded successfully with examples", str(len_light_list))
        print("Now we have", len(self.light_list), 'light source images')

    def load_reflective_flare(self, reflective_name, reflective_path):
        if reflective_path is None:
            reflective_list = []
        else:
            reflective_list = []
            [reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
            reflective_list = sorted(reflective_list)
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name] = reflective_list
        self.reflective_list.append(reflective_list)
        len_reflective_list = len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            self.reflective_flag = True
            print("Reflective Flare Image:", reflective_name, " is loaded successfully with examples",
                  str(len_reflective_list))
        print("Now we have", len(self.reflective_list), 'refelctive flare images')


@DATASET_REGISTRY.register()
class FlareX_Pair_Loader(Paired_Flare_Image_Loader):
    def __init__(self, opt):
        Paired_Flare_Image_Loader.__init__(self, opt['image_path'], opt['image_pair_path'], opt['transform_base'],
                                           opt['transform_flare'], opt['mask_type'])
        scattering_dict = opt['scattering_dict']
        reflective_dict = opt['reflective_dict']
        light_dict = opt['light_dict']

        # defualt not use light mask if opt['use_light_mask'] is not declared
        if 'data_ratio' not in opt or len(opt['data_ratio']) == 0:
            self.data_ratio = [1] * len(scattering_dict)
        else:
            self.data_ratio = opt['data_ratio']

        if len(scattering_dict) != 0:
            for key in scattering_dict.keys():
                self.load_scattering_flare(key, scattering_dict[key])
        if len(reflective_dict) != 0:
            for key in reflective_dict.keys():
                self.load_reflective_flare(key, reflective_dict[key])
        if len(light_dict) != 0:
            for key in light_dict.keys():
                self.load_light_source(key, light_dict[key])
