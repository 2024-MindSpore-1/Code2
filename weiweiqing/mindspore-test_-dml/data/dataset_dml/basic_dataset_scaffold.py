from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision, transforms
import mindspore as ms
import numpy as np
from PIL import Image
from mindspore.dataset import PKSampler

"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset:
    # image_dict: { cls0: [0.jpg, 1.jpg, ...], cls1: [0.jpg, 1.jpg, ...], ...}
    def __init__(self, image_dict, opt, is_validation=False):
        self.is_validation = is_validation
        self.opt          = opt

        #####
        self.image_dict = image_dict

        #####
        self.init_setup()


        #####
        # if 'bninception' not in opt.arch:
        #     self.f_norm = normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # else:
        #     # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
        #     self.f_norm = normalize = vision.Normalize(mean=[0.502, 0.4588, 0.4078], std=[0.0039, 0.0039, 0.0039])

        # transf_list = []

        self.crop_size = crop_im_size = 224 if 'googlenet' not in opt.arch else 227
        if opt.augmentation == 'big':
            crop_im_size = 256

        #############
        # self.normal_transform = []
        # if not self.is_validation:
        #     if opt.augmentation == 'base' or opt.augmentation == 'big':
        #         self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        #     elif opt.augmentation == 'adv':
        #         self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
        #                                       transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        #     elif opt.augmentation == 'red':
        #         self.normal_transform.extend([transforms.Resize(size=256), transforms.RandomCrop(crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        # else:
        #     # if valid, just resize & center crop
        #     self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        # self.normal_transform.extend([transforms.ToTensor(), normalize])  # first convert to tensor, then normalize
        # self.normal_transform = transforms.Compose(self.normal_transform)


    def init_setup(self):
        self.n_files       = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])  # number of all images
        self.avail_classes = sorted(list(self.image_dict.keys()))  # list of class names


        counter = 0
        temp_image_dict = {}
        for i, key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1
        self.image_dict = temp_image_dict
        # add unique id(increased number) for all images
        # image_dict = {
        #   cls_0: [(0_0.jpg, 0), (0_1.jpg, 1)]
        #   cls_1: [(1_0.jpg, 2), (1_1.jpg, 3)]
        #   ...
        # }

        self.image_list = [[(x[0], key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        # [ [(0_0.jpg, cls_0), (0_1.jpg, cls_0)], [(1_0.jpg, cls1), (1_1.jpg, cls2)], ... ]

        self.image_list = [x for y in self.image_list for x in y]
        # [ (0_0.jpg, cls_0), (0_1.jpg, cls_0), (1_0.jpg, cls1), (1_1.jpg, cls2), ... ]

        self.image_paths = self.image_list

        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))
        im_a = input_image

        # # Basic preprocessing.
        # im_a = self.normal_transform(input_image)
        # if 'bninception' in self.pars.arch:
        #     im_a = im_a[range(3)[::-1], :]
        # return self.image_list[idx][-1], im_a, idx
        #       label, image_data, index

        # return: label, image_data, index
        return self.image_list[idx][-1], im_a, idx

    def __len__(self):
        return self.n_files

    def generate_dataset(self, data_sampler=None):

        assert self.opt.bs % self.opt.samples_per_class == 0, '#Samples per class must divide batch size!'

        dataset = GeneratorDataset(
            source=self,
            column_names=["label", "image", "index"],
            shuffle=False,
            num_parallel_workers=self.opt.kernels,
            sampler=None if self.is_validation else data_sampler,
        )

        label_tran = transforms.TypeCast(ms.int32)

        mean = [0.502, 0.4588, 0.4078]
        std = [0.0039, 0.0039, 0.0039]

        crop_im_size = 256 if self.opt.augmentation == "big" else 224

        img_tran = []
        if self.is_validation:
            img_tran = [
                vision.Resize(256),
                vision.CenterCrop(crop_im_size),
            ]
        else:
            if self.opt.augmentation in ["base", "big"]:
                img_tran = [
                    vision.RandomResizedCrop(crop_im_size),
                    vision.RandomHorizontalFlip(prob=0.5),
                ]
            elif self.opt.augmentation == "adv":
                img_tran = [
                    vision.RandomResizedCrop(crop_im_size),
                    vision.RandomGrayscale(prob=0.2),
                    # vision.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    vision.RandomHorizontalFlip(prob=0.5),
                ]
            elif self.opt.augmentation == "red":
                img_tran = [
                    vision.Resize(256),
                    vision.RandomCrop(crop_im_size),
                    vision.RandomHorizontalFlip(prob=0.5),
                ]

        img_tran.extend([
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ])

        dataset = dataset.map(operations=img_tran, input_columns=["image"])
        dataset = dataset.map(operations=label_tran, input_columns=["label"])
        dataset = dataset.batch(self.opt.bs)

        dataset.image_paths = self.image_paths

        return dataset
