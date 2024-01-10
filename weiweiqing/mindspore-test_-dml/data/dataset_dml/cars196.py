from .basic_dataset_scaffold import BaseDataset
import os
import numpy as np
from . import register_dataset


@register_dataset
def Cars196(opt, data_path):
    image_source_path  = data_path + '/images'

    # ["image_class1_folder", "image_class2_folder", ...]
    image_classes     = sorted([x for x in os.listdir(image_source_path)])

    # { 0: "image_class1_folder", 1: "image_class2_folder", ...]
    total_conversion  = {i: x for i, x in enumerate(image_classes)}

    # region Generate image list
    # {
    #   0: ["datapath/images/class1_folder/0.jpg", "datapath/images/class1_folder/1.jpg"...],
    #   1: ["datapath/images/class2_folder/0.jpg", "datapath/images/class2_folder/1.jpg", ...],
    #   ...
    # }
    image_list    = {i: sorted([image_source_path+'/'+key+'/'+x for x in os.listdir(image_source_path+'/'+key)]) for i, key in enumerate(image_classes)}

    # [
    #  [(0: "datapath/images/class1_folder/0.jpg"), (0: "datapath/images/class1_folder/1.jpg"), ...],
    #  [(1: "datapath/images/class2_folder/0.jpg"), (1: "datapath/images/class2_folder/1.jpg"), ...],
    # ]
    image_list    = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]

    # [
    #  (0: "datapath/images/class1_folder/0.jpg"),
    #  (0: "datapath/images/class1_folder/1.jpg"),
    #  ...,
    #  (1: "datapath/images/class2_folder/0.jpg"),
    #  (1: "datapath/images/class2_folder/1.jpg"),
    #  ...
    # ]
    image_list    = [x for y in image_list for x in y]

    # endregion

    # region Generate image dict

    # Dictionary of structure class:list_of_samples_with_said_class
    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)
    #   -----------------------------------------------------------------------------------------
    # {
    #   0: ["datapath/images/class1_folder/0.jpg", "datapath/images/class1_folder/1.jpg"...],
    #   1: ["datapath/images/class2_folder/0.jpg", "datapath/images/class2_folder/1.jpg", ...],
    #   ...
    #  }
    #   ------------------------------------------------------------------------------------------

    # endregion

    # region Split classes to train classes and test classes

    # Use the first half of the sorted data as training and the second half as test set
    # 50% to 50% ?

    keys = sorted(list(image_dict.keys()))
    train, test      = keys[:len(keys)//2], keys[len(keys)//2:]

    # endregion

    # region Split train data to train set and validation set, by all classes or by images of per class, set val_image_dict, val_data_set, val_convertion

    # If required, split the training data into a train/val setup.
    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            # split classes to train & val
            train_val_split = int(len(train)*opt.tv_split_perc)
            train, val      = train[:train_val_split], train[train_val_split:]
            # any function?
            train_image_dict = {i: image_dict[key] for i, key in enumerate(train)}
            val_image_dict   = {i: image_dict[key] for i, key in enumerate(val)}
            # test_image_dict  = {i:image_dict[key] for i,key in enumerate(test)}
        else:
            # for each class, split images to train & val
            val = train
            train_image_dict, val_image_dict = {}, {}
            for key in train:
                # for images in one class, split to train images and val images randomly
                train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key]) * opt.tv_split_perc), replace=False)
                val_ixs   = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]   # train images in one class
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]  # val images in one class
        val_dataset   = BaseDataset(val_image_dict,   opt, is_validation=True)  # val dataset
        val_conversion = {i: total_conversion[key] for i, key in enumerate(val)}  # val conversion
        ###
        val_dataset.conversion   = val_conversion
    else:
        # no validate image dataset, all images as train images
        train_image_dict = {key: image_dict[key] for key in train}  # train image dict: { 0: [image1_path, image2_path,...], 1: []...}
        val_image_dict   = None
        val_dataset      = None

    # endregion, set h,, set valeses,

    # region for train/test/eval, set image_dict, conversion, dataset

    test_image_dict = {key: image_dict[key] for key in test}   # test image dict

    # conversion name is same to class folder name
    train_conversion = {i: total_conversion[key] for i, key in enumerate(train)}  # {0: "class1_folder", 1: "class2_folder", ...}
    test_conversion  = {i: total_conversion[key] for i, key in enumerate(test)}  # {1000: "class1_folder", 1001: "class2_folder", ...}

    #
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    # train & test dataset
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)

    # eval dataset, same to train
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)

    # train & test conversion
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion

    # set eval conversion(note: dataset is same), use train & test respectively
    eval_dataset.conversion        = test_conversion
    eval_train_dataset.conversion  = train_conversion

    # endregion

    return {
        'training': train_dataset,
        'testing': test_dataset,

        'validation': val_dataset,  # maybe None

        'evaluation': eval_dataset,  # maybe None
        'evaluation_train': eval_train_dataset  # maybe None
    }
