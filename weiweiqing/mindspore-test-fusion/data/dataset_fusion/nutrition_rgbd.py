import numpy as np
import cv2
from PIL import Image

from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision, transforms
import mindspore as ms
import os


class Nutrition_RGBD:
    def __init__(self, args, is_train):
        image_path = os.path.join(args.data_root, 'imagery')
        if is_train:
            rgbd_txt_dir = os.path.join(args.data_root, 'imagery', 'rgbd_train_processed.txt')
            rgb_txt_dir = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_train_processed.txt')
        else:
            rgbd_txt_dir = os.path.join(args.data_root, 'imagery', 'rgbd_test_processed1.txt')  # depth_color.png
            rgb_txt_dir = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_test_processed1.txt')  # rbg.png

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)) # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB)) # cv2转PIL
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        return img_rgb, self.labels[index],\
            self.total_calories[index], self.total_mass[index],\
            self.total_fat[index], self.total_carb[index],\
            self.total_protein[index], img_rgbd  # 返回 2种image即可，然后再在train2.py中多一个判断，两个图片输入两次网络

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_dataset(
            args,
    ):
        train_trans = [
                vision.Resize((320, 448)),
                vision.RandomHorizontalFlip(),
                vision.Rescale(1.0 / 255.0, 0),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]

        train_trans_rgbd = [
            vision.Resize((320, 448)),
            vision.RandomHorizontalFlip(),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]

        val_trans = [
                vision.Resize((320, 448)),
                vision.Rescale(1.0 / 255.0, 0),
                vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                vision.HWC2CHW()
            ]

        val_trans_rgbd = [
            vision.Resize((320, 448)),
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]

        # label_tran = transforms.TypeCast(ms.int32)

        train_dataset = GeneratorDataset(
            source=Nutrition_RGBD(
                args=args,
                is_train=True,
            ),
            column_names=["image", "label", "total_calories", "total_mass", "total_fat", "total_carb", "total_protein", "image_rgbd"],
            shuffle=True,
            num_parallel_workers=4,
        )
        train_dataset = train_dataset.map(operations=train_trans, input_columns=["image"])
        train_dataset = train_dataset.map(operations=train_trans_rgbd, input_columns=["image_rgbd"])
        # train_dataset = train_dataset.map(operations=label_tran, input_columns=["label"])
        train_dataset = train_dataset.batch(args.b)

        val_dataset = GeneratorDataset(
            source=Nutrition_RGBD(
                args=args,
                is_train=False,
            ),
            column_names=["image", "label", "total_calories", "total_mass", "total_fat", "total_carb", "total_protein", "image_rgbd"],
            shuffle=False,
            num_parallel_workers=4,
        )
        val_dataset = val_dataset.map(operations=val_trans, input_columns=["image"])
        val_dataset = val_dataset.map(operations=val_trans_rgbd, input_columns=["image_rgbd"])
        # val_dataset = val_dataset.map(operations=label_tran, input_columns=["label"])
        val_dataset = val_dataset.batch(args.b)

        return train_dataset, val_dataset
