import numpy as np
import math
import cv2
import random
import os
from utils import load_config, extract_filename, load_base_xml, write_base_cml

#############################################
# Project Setting
#############################################
project = "nut/"


#############################################
# Load Config
#############################################
cfg = load_config("./")
if cfg["colab_run"]:
    project_path = "../drive/MyDrive/projects/" + project
else:
    project_path = "../projects/" + project
cfg["train_data_dir"] = project_path + "datasets/" + 'train'
cfg["val_data_dir"] = project_path + "datasets/" + 'val'
cfg["test_data_dir"] = project_path + "datasets/" + 'test'
cfg["prep_train_dir"] = project_path + 'preprocess/' + 'train'
cfg["prep_val_dir"] = project_path + 'preprocess/' + 'val'
cfg["prep_test_dir"] = project_path + 'preprocess/' + 'test'


#############################################
# Implement Functions
#############################################

def read_img(img_path, grayscale):
    """Read image"""
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


def is_image(filename):
    suffix_img_names = {"bmp", "jpg", "jpeg", "png"}
    return str.lower(filename.split(".")[-1]) in suffix_img_names


def fill_squre(img, color):
    """Fix image which not square"""
    height, width = img.shape[:2]
    max_val = max(height, width)
    if cfg["grayscale"]:
        size = (max_val, max_val)
        blank_image = np.zeros(size, np.uint8)
        blank_image[:, :] = color
    else:
        size = (max_val, max_val, 3)
        blank_image = np.zeros(size, np.uint8)
        blank_image[:, :] = (color, color, color)

    im = blank_image.copy()

    x_offset = y_offset = 0
    im[y_offset:y_offset + height, x_offset:x_offset + width] = img.copy()
    return im


"""
Edit rotation
"""


def rotate_image(img, angle):
    """Edit rotation image"""
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    return img_rotated


def random_rotate(img, angle_vari):
    """Random rotation degree"""
    angle = np.random.uniform(-angle_vari, angle_vari)
    return rotate_image(img, angle)


"""
Edit light & contrast
"""


def apply_brightness_contrast(img, brightness=0, contrast=0):
    """Edit light & contrast"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def random_light(img, light):
    """Random light brightness-contrast"""
    brightness = np.random.uniform(-light, light)
    contrast = np.random.uniform(-light, light)
    return apply_brightness_contrast(img, brightness, contrast)


"""
Add Noise
    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""


def noisy(img, noise_typ):
    """Add Noise"""
    # todo for gray image
    if noise_typ == "gauss" or noise_typ == 0:
        img_shape = img.shape
        if len(img_shape) == 2:
            row, col = img.shape
            ch = 1
        else:
            row, col, ch = img.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = img + gauss
        return noisy
    elif noise_typ == "s&p" or noise_typ == 1:
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson" or noise_typ == 2:
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle" or noise_typ == 3:
        img_shape = img.shape
        if len(img_shape) == 2:
            row, col = img.shape
            ch = 1
        else:
            row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        noisy = img + img * gauss
        return noisy


def random_noise(img, noise_type):
    """Random light brightness-contrast"""
    noise_type = np.random.uniform(0, noise_type)
    return noisy(img, noise_type)


def generate_image_list(conf, mode):
    """Generate all image data[path,] from target path"""
    if mode == "test":
        img_list = [(os.sep.join([conf["test_data_dir"], f]), 1) for f in os.listdir(conf["test_data_dir"]) if
                    is_image(f)]
    else:
        # todo random train-val
        if mode == "train":
            data_dir = conf["train_data_dir"]
            aug_num = int(conf["augment_num"] * (1 - conf["val_ratio"]))
        else:
            data_dir = conf["val_data_dir"]
            aug_num = int(conf["augment_num"] * conf["val_ratio"])
        filenames = [f for f in os.listdir(data_dir) if is_image(f)]
        num_imgs = len(filenames)
        num_ave_aug = int(math.floor(aug_num / num_imgs))
        rem = aug_num - num_ave_aug * num_imgs
        lucky_seq = [True] * rem + [False] * (num_imgs - rem)
        random.shuffle(lucky_seq)

        img_list = [(os.sep.join([data_dir, filename]), num_ave_aug + 1 if lucky else num_ave_aug)
                    for filename, lucky in zip(filenames, lucky_seq)]
    return img_list


def augment_images(file_list, conf, mode):
    """Generate augment images"""
    for file_path, n in file_list:
        img = read_img(file_path, conf["grayscale"])
        height, width = img.shape[:2]
        if (height, width) != (conf["img_resize"], conf["img_resize"]):
            if height != width and conf["fill_square"]:
                img = fill_squre(img, conf["fill_square_color"])
            img = cv2.resize(img, (conf["img_resize"], conf["img_resize"]))

        # IMAGE AUGMENTATION
        file_split = extract_filename(file_path)
        img_name = file_split["imgname"]
        ext = file_split["ext"]

        if mode == "train" or mode == "val":
            if mode == "train":
                prep_dir = conf["prep_train_dir"]
            else:
                prep_dir = conf["prep_val_dir"]
            for i in range(n):
                img_varied = img.copy()
                varied_imgname = '{}_{:0>3d}_'.format(img_name, i)

                if random.random() < conf["p_noise"]:
                    # not change pos fn
                    img_varied = random_noise(img_varied, conf["noise_vari"])
                    varied_imgname += 'n'

                if random.random() < conf["p_light"]:
                    # not change pos fn
                    img_varied = random_light(img_varied, conf["light_vari"])
                    varied_imgname += 'l'

                if random.random() < conf["p_rotate"]:
                    # todo change pos fn
                    img_varied_ = random_rotate(img_varied, conf["rotate_angle_vari"])
                    if img_varied_.shape[0] >= conf["img_resize"] and img_varied_.shape[1] >= conf["img_resize"]:
                        img_varied = img_varied_
                    varied_imgname += 'r'

                if random.random() < conf["p_horizonal_flip"]:
                    # todo change pos fn
                    img_varied = cv2.flip(img_varied, 1)
                    varied_imgname += 'h'

                if random.random() < conf["p_vertical_flip"]:
                    # todo change pos fn
                    img_varied = cv2.flip(img_varied, 0)
                    varied_imgname += 'v'

                output_file_path = os.path.join(prep_dir, '{}{}'.format(varied_imgname, ext))
                cv2.imwrite(output_file_path, img_varied)
        else:
            output_file_path = os.path.join(conf["prep_test_dir"], '{}{}'.format(img_name, ext))
            cv2.imwrite(output_file_path, img)


def segment_transform(file_list, conf, mode):
    """Copy data from base XML and change to new one"""
    not_change_pos_keys = {"", "n", "l", "nl"}
    img_resize = conf["img_resize"]

    if mode == "train":
        org_dir = conf["train_data_dir"]
        prep_dir = conf["prep_train_dir"]
    else:
        org_dir = conf["val_data_dir"]
        prep_dir = conf["prep_val_dir"]

    for file_path in file_list:
        file_split = extract_filename(file_path)
        filename = file_split["filename"]
        img_name = file_split["imgname"]
        transform_type = img_name.split("_")[-1]

        if transform_type in not_change_pos_keys:
            org_xml_file = "_".join(img_name.split("_")[:-2]) + ".xml"
            new_xml_file = img_name + ".xml"
            org_xml_path = os.path.join(org_dir, org_xml_file)
            mytree, myroot = load_base_xml(org_xml_path)

            # FILE DESCRIBE CONFIG
            for p11 in myroot.iter('filename'):
                p11.text = filename
            for p12 in myroot.iter('path'):
                p12.text = os.path.join(prep_dir, org_xml_file)

            # IMAGE DATA CONFIG
            ratio_changed = 1
            for p11 in myroot.iter('width'):
                ratio_changed = int(p11.text) / img_resize
                p11.text = str(img_resize)
            for p12 in myroot.iter('height'):
                p12.text = str(img_resize)
            for p1 in myroot.iter('xmin'):
                p1.text = str(int(p1.text) / ratio_changed)
            for p2 in myroot.iter('xmax'):
                p2.text = str(int(p2.text) / ratio_changed)
            for p3 in myroot.iter('ymin'):
                p3.text = str(int(p3.text) / ratio_changed)
            for p4 in myroot.iter('ymax'):
                p4.text = str(int(p4.text) / ratio_changed)
        else:
            # todo for position img changed case
            raise Exception("Error: Not implemented yet")
        new_xml_path = os.path.join(prep_dir, new_xml_file)
        write_base_cml(new_xml_path, mytree)


#############################################
# Run: Pre-process
#############################################

if __name__ == '__main__':
    # DATA AUGMENTATION DIRECTORY
    if not os.path.exists(project_path + 'preprocess/'):
        os.makedirs(project_path + 'preprocess/')

    # TRAIN
    if not os.path.exists(cfg["prep_train_dir"]):
        os.makedirs(cfg["prep_train_dir"])
        img_list = generate_image_list(cfg, mode="train")
        augment_images(img_list, cfg, mode="train")
    generated_files = os.listdir(cfg["prep_train_dir"])
    print("Images(train): ", len(generated_files))
    segment_transform(generated_files, cfg, mode="train")
    generated_files = os.listdir(cfg["prep_train_dir"])
    print("XML-Images(train): ", len(generated_files))

    # VALIDATE
    if not os.path.exists(cfg["prep_val_dir"]):
        os.makedirs(cfg["prep_val_dir"])
        img_list = generate_image_list(cfg, mode="val")
        augment_images(img_list, cfg, mode="val")
    generated_files = os.listdir(cfg["prep_val_dir"])
    print("Images(val): ", len(generated_files))
    segment_transform(generated_files, cfg, mode="val")
    generated_files = os.listdir(cfg["prep_val_dir"])
    print("XML-Images(val): ", len(generated_files))

    # test
    if not os.path.exists(cfg["prep_test_dir"]):
        os.makedirs(cfg["prep_test_dir"])
        # test
        img_list = generate_image_list(cfg, mode="test")
        augment_images(img_list, cfg, mode="test")
    generated_files = os.listdir(cfg["prep_test_dir"])
    print("Images(test): ", len(generated_files))
