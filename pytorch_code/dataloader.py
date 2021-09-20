import os
import torch
import random
import copy
import csv
from glob import glob
from PIL import Image
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage import measure
from skimage.transform import resize

from torch.utils.data import Dataset
import torchvision.transforms as transforms

NORMALIZATION_STATISTICS = {"luna16": [[0.2563873675129015, 0.2451283333368983]],
                            "self_learning_cubes_32": [[0.11303308354465243, 0.12595135887180803]],
                            "self_learning_cubes_64": [[0.11317437834743148, 0.12611378817031038]],
                            "lidc": [[0.23151727, 0.2168428080133056]],
                            "luna_fpr": [[0.18109835972793722, 0.1853707675313153]],
                            "lits_seg": [[0.46046468844492944, 0.17490586272419967]],
                            "pe": [[0.26125720740546626, 0.20363551346695796]],
                            "pe16": [[0.2887357771623902, 0.24429971299033243]],
                            # [[0.29407377554678416, 0.24441741466975556]], ->256x256x128
                            "brats": [[0.28239742604241436, 0.22023889204407615]],
                            "luna16_lung": [[0.1968134997129321, 0.20734707135528743]]}


# ---------------------------------------------2D Data augmentation---------------------------------------------
class Augmentation():
  def __init__(self, normalize):
    if normalize.lower() == "imagenet":
      self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      self.normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)

  def get_augmentation(self, augment_name, mode, *args):
    try:
      aug = getattr(Augmentation, augment_name)
      return aug(self, mode, *args)
    except:
      print("Augmentation [{}] does not exist!".format(augment_name))
      exit(-1)

  def basic(self, mode):
    transformList = []
    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def _basic_crop(self, transCrop, mode="train"):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomCrop(transCrop))
    else:
      transformList.append(transforms.CenterCrop(transCrop))
    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def basic_crop_224(self, mode):
    transCrop = 224
    return self._basic_crop(transCrop, mode)

  def _basic_resize(self, size, mode="train"):
    transformList = []
    transformList.append(transforms.Resize(size))
    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def basic_resize_224(self, mode):
    size = 224
    return self._basic_resize(size, mode)

  def _basic_crop_rot(self, transCrop, mode="train"):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomCrop(transCrop))
      transformList.append(transforms.RandomRotation(7))
    else:
      transformList.append(transforms.CenterCrop(transCrop))

    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def basic_crop_rot_224(self, mode):
    transCrop = 224
    return self._basic_crop_rot(transCrop, mode)

  def _basic_crop_flip(self, transCrop, transResize, mode="train"):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
    else:
      transformList.append(transforms.Resize(transResize))
      transformList.append(transforms.CenterCrop(transCrop))

    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def basic_crop_flip_224(self, mode):
    transCrop = 224
    transResize = 256
    return self._basic_crop_flip(transCrop, transResize, mode)

  def _basic_rdcrop_flip(self, transCrop, transResize, mode="train"):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomResizedCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
    else:
      transformList.append(transforms.Resize(transResize))
      transformList.append(transforms.CenterCrop(transCrop))

    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def basic_rdcrop_flip_224(self, mode):
    transCrop = 224
    transResize = 256
    return self._basic_rdcrop_flip(transCrop, transResize, mode)

  def _full(self, transCrop, transResize, mode="train", test_augment=True):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomResizedCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
      transformList.append(transforms.RandomRotation(7))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "valid":
      transformList.append(transforms.Resize(transResize))
      transformList.append(transforms.CenterCrop(transCrop))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "test":
      if test_augment:
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if self.normalize is not None:
          transformList.append(transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
      else:
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
          transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def full_224(self, mode, test_augment=True):
    transCrop = 224
    transResize = 256
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

  def full_448(self, mode):
    transCrop = 448
    transResize = 512
    return self._full(transCrop, transResize, mode)

  def _full_colorjitter(self, transCrop, transResize, mode="train"):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomResizedCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
      transformList.append(transforms.RandomRotation(7))
      transformList.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "valid":
      transformList.append(transforms.Resize(transResize))
      transformList.append(transforms.CenterCrop(transCrop))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "test":
      transformList.append(transforms.Resize(transResize))
      transformList.append(transforms.TenCrop(transCrop))
      transformList.append(
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
      if self.normalize is not None:
        transformList.append(transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def full_colorjitter_224(self, mode):
    transCrop = 224
    transResize = 256
    return self._full_colorjitter(transCrop, transResize, mode)


# ---------------------------------------------3D Data Normalization--------------------------------------------
def channel_wise_normalize_3d(data, mean_std):
  num_data = data.shape[0]
  num_channel = data.shape[1]

  if len(mean_std) == 1:
    mean_std = [mean_std[0]] * num_channel

  normalized_data = []
  for i in range(num_data):
    img = data[i, ...]
    normalized_img = []

    for j in range(num_channel):
      img_per_channel = img[j, ...]
      mean, std = mean_std[j][0], mean_std[j][1]
      _img = (img_per_channel - mean) / std
      normalized_img.append(_img)

    normalized_data.append(normalized_img)

  return np.array(normalized_data)


# ---------------------------------------------Downstream ChestX-ray14------------------------------------------
class ChestX_ray14(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14, anno_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpert(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, anno_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(pathDatasetFile, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(pathImageDirectory, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------------NPY DataSet------------------------------------------------
class NPYDataLoader(Dataset):
  def __init__(self, data):
    self.data_x, self.data_y = data

  def __len__(self):
    return self.data_x.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    return self.data_x[idx, ...], self.data_y[idx, ...]


# --------------------------------------------Downstream LUNA FPR 3D--------------------------------------------
def LUNA_FPR_3D(data_dir, fold, input_size, hu_range, crop=True, normalization=None, set="data", anno_percent=100,
                shuffle=True):
  input_rows, input_cols, input_deps = input_size[0], input_size[1], input_size[2]
  hu_min, hu_max = hu_range[0], hu_range[1]

  def load_image(data_dir, fold, input_rows, input_cols, hu_min, hu_max, crop=True):
    positives, negatives = [], []

    for subset in fold:
      LUNA16_PROCESSED_DIR_POS = os.path.join(data_dir, "subset" + str(subset), "positives")
      LUNA16_PROCESSED_DIR_NEG = os.path.join(data_dir, "subset" + str(subset), "negatives")

      positive_file_list = glob(os.path.join(LUNA16_PROCESSED_DIR_POS, "*.npy"))
      negative_file_list = glob(os.path.join(LUNA16_PROCESSED_DIR_NEG, "*.npy"))

      positive_index = [x for x in range(len(positive_file_list))]
      negative_index = [x for x in range(len(negative_file_list))]
      if shuffle:
        random.shuffle(positive_index)
        random.shuffle(negative_index)

      for i in range(min(len(positive_file_list), len(negative_file_list))):
        im_pos_ = np.load(positive_file_list[positive_index[i]])
        im_neg_ = np.load(negative_file_list[negative_index[i]])

        if crop:
          im_pos = np.zeros((input_rows, input_cols, im_pos_.shape[-1]), dtype="float")
          im_neg = np.zeros((input_rows, input_cols, im_pos_.shape[-1]), dtype="float")
          for z in range(im_pos_.shape[-1]):
            im_pos[:, :, z] = resize(im_pos_[:, :, z], (input_rows, input_cols), preserve_range=True)
            im_neg[:, :, z] = resize(im_neg_[:, :, z], (input_rows, input_cols), preserve_range=True)
        else:
          im_pos, im_neg = im_pos_, im_neg_

        im_pos[im_pos < hu_min] = hu_min
        im_pos[im_pos > hu_max] = hu_max
        im_neg[im_neg < hu_min] = hu_min
        im_neg[im_neg > hu_max] = hu_max

        im_pos = (im_pos - hu_min) / (hu_max - hu_min)
        im_neg = (im_neg - hu_min) / (hu_max - hu_min)

        positives.append(im_pos)
        negatives.append(im_neg)
    positives, negatives = np.array(positives), np.array(negatives)
    positives, negatives = np.expand_dims(positives, axis=-1), np.expand_dims(negatives, axis=-1)

    return positives, negatives

  x_pos, x_neg = load_image(data_dir, fold, input_rows, input_cols, hu_min, hu_max, crop=crop)
  x_data = np.concatenate((x_pos, x_neg), axis=0)
  y_data = np.concatenate((np.ones((x_pos.shape[0],)),
                           np.zeros((x_neg.shape[0],)),
                           ), axis=0)
  x_data = np.expand_dims(np.squeeze(x_data), axis=1)

  if normalization is not None and normalization.lower() != "none":
    mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
    x_data = channel_wise_normalize_3d(x_data, mean_std=mean_std)

  if anno_percent < 100:
    ind_list = [i for i in range(x_data.shape[0])]
    random.Random(99).shuffle(ind_list)

    num_data = int(x_data.shape[0] * anno_percent / 100.0)
    x_data = x_data[ind_list[:num_data], ...]
    y_data = y_data[ind_list[:num_data], ...]

  print("x_{}: {} | {:.2f} ~ {:.2f}".format(set, x_data.shape, np.min(x_data), np.max(x_data)))
  print("y_{}: {} | {:.2f} ~ {:.2f}".format(set, y_data.shape, np.min(y_data), np.max(y_data)))

  return x_data, y_data


# ----------------------------------------------Downstream LIDC 3D----------------------------------------------
def LIDC_3D(data_dir, set, normalization=None, anno_percent=100):
  x_data = np.squeeze(np.load(os.path.join(data_dir, 'x_' + set + '_64x64x32.npy')))
  y_data = np.squeeze(np.load(os.path.join(data_dir, 'm_' + set + '_64x64x32.npy')))
  x_data = np.expand_dims(x_data, axis=1)
  y_data = np.expand_dims(y_data, axis=1)

  if normalization is not None and normalization.lower() != "none":
    mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
    x_data = channel_wise_normalize_3d(x_data, mean_std=mean_std)

  if anno_percent < 100:
    ind_list = [i for i in range(x_data.shape[0])]
    random.Random(99).shuffle(ind_list)

    num_data = int(x_data.shape[0] * anno_percent / 100.0)
    x_data = x_data[ind_list[:num_data], ...]
    y_data = y_data[ind_list[:num_data], ...]

  print("x_{}: {} | {:.2f} ~ {:.2f}".format(set, x_data.shape, np.min(x_data), np.max(x_data)))
  print("y_{}: {} | {:.2f} ~ {:.2f}".format(set, y_data.shape, np.min(y_data), np.max(y_data)))

  return x_data, y_data


# ----------------------------------------------Downstream LiTS 3D----------------------------------------------
def LiTS_3D(data_path, id_list, obj="liver", normalization=None, anno_percent=100,
            input_size=(64, 64, 32), hu_range=(-1000.0, 1000.0), status=None):
  def load_data_npy(data_path, id_list, obj="liver", input_size=(64, 64, 32), hu_range=(-1000.0, 1000.0), status=None):

    x_data, y_data = [], []
    input_rows, input_cols, input_deps = input_size[0], input_size[1], input_size[2]
    hu_min, hu_max = hu_range[0], hu_range[1]

    for patient_id in id_list:

      Vol = np.load(os.path.join(data_path, "volume-" + str(patient_id) + ".npy"))
      Vol[Vol > hu_max] = hu_max
      Vol[Vol < hu_min] = hu_min
      Vol = (Vol - hu_min) / (hu_max - hu_min)
      Vol = np.expand_dims(Vol, axis=0)
      Mask = np.load(os.path.join(data_path, "segmentation-" + str(patient_id) + ".npy"))
      liver_mask, lesion_mask = copy.deepcopy(Mask), copy.deepcopy(Mask)
      liver_mask[Mask > 0.5] = 1
      liver_mask[Mask <= 0.5] = 0
      lesion_mask[Mask > 1] = 1
      lesion_mask[Mask <= 1] = 0
      Mask = np.concatenate((np.expand_dims(liver_mask, axis=0), np.expand_dims(lesion_mask, axis=0)), axis=0)

      if obj == "liver":
        for i in range(input_rows - 1, Vol.shape[1] - input_rows + 1, input_rows):
          for j in range(input_cols - 1, Vol.shape[2] - input_cols + 1, input_cols):
            for k in range(input_deps - 1, Vol.shape[3] - input_deps + 1, input_deps):
              if np.sum(Mask[0, i:i + input_rows, j:j + input_cols,
                        k:k + input_deps]) > 0 or random.random() < 0.01:
                x_data.append(Vol[:, i:i + input_rows, j:j + input_cols, k:k + input_deps])
                y_data.append(Mask[:, i:i + input_rows, j:j + input_cols, k:k + input_deps])

        if np.sum(Mask[0]) > 1000:
          cx, cy, cz = ndimage.measurements.center_of_mass(np.squeeze(Mask[0]))
          # print(cx, cy, cz)
          cx, cy, cz = int(cx), int(cy), int(cz)
          for delta_x in range(-10, 20, 20):
            for delta_y in range(-10, 20, 20):
              for delta_z in range(-5, 10, 10):
                if cx + delta_x - int(input_rows / 2) < 0 or cx + delta_x + int(input_rows / 2) > Vol.shape[1] - 1 or \
                    cy + delta_y - int(input_cols / 2) < 0 or cy + delta_y + int(input_cols / 2) > Vol.shape[2] - 1 or \
                    cz + delta_z - int(input_deps / 2) < 0 or cz + delta_z + int(input_deps / 2) > Vol.shape[3] - 1:
                  pass
                else:
                  x_data.append(Vol[:, cx + delta_x - int(input_rows / 2):cx + delta_x + int(input_rows / 2), \
                                cy + delta_y - int(input_cols / 2):cy + delta_y + int(input_cols / 2), \
                                cz + delta_z - int(input_deps / 2):cz + delta_z + int(input_deps / 2)])
                  y_data.append(Mask[:, cx + delta_x - int(input_rows / 2):cx + delta_x + int(input_rows / 2), \
                                cy + delta_y - int(input_cols / 2):cy + delta_y + int(input_cols / 2), \
                                cz + delta_z - int(input_deps / 2):cz + delta_z + int(input_deps / 2)])
      elif obj == "lesion":
        if np.sum(Mask[1]) > 0:
          labels = measure.label(Mask[1], neighbors=8, background=0)
          for label in np.unique(labels):
            if label == 0:
              continue
            labelMask = np.zeros(Mask[1].shape, dtype="int")
            labelMask[labels == label] = 1
            cx, cy, cz = ndimage.measurements.center_of_mass(np.squeeze(labelMask))
            cx, cy, cz = int(cx), int(cy), int(cz)
            if labelMask[cx, cy, cz] == 1:
              for delta_x in range(-5, 5, 5):
                for delta_y in range(-5, 5, 5):
                  for delta_z in range(-3, 3, 3):
                    if cx + delta_x - int(input_rows / 2) < 0 or cx + delta_x + int(input_rows / 2) > Vol.shape[1] - 1 \
                        or \
                        cy + delta_y - int(input_cols / 2) < 0 or cy + delta_y + int(input_cols / 2) > Vol.shape[2] - 1 \
                        or \
                        cz + delta_z - int(input_deps / 2) < 0 or cz + delta_z + int(input_deps / 2) > Vol.shape[3] - 1:
                      pass
                    else:
                      x_data.append(
                        Vol[:, cx + delta_x - int(input_rows / 2):cx + delta_x + int(input_rows / 2), \
                        cy + delta_y - int(input_cols / 2):cy + delta_y + int(input_cols / 2), \
                        cz + delta_z - int(input_deps / 2):cz + delta_z + int(input_deps / 2)])
                      y_data.append(
                        Mask[:, cx + delta_x - int(input_rows / 2):cx + delta_x + int(input_rows / 2), \
                        cy + delta_y - int(input_cols / 2):cy + delta_y + int(input_cols / 2), \
                        cz + delta_z - int(input_deps / 2):cz + delta_z + int(input_deps / 2)])
      else:
        print("Objetc [{}] does not exist!".format(obj))

    return np.array(x_data), np.array(y_data)

  x_data, y_data = load_data_npy(data_path, id_list, obj=obj, input_size=input_size, hu_range=hu_range, status=status)
  # print(x_data.shape, y_data.shape)

  if obj == "liver":
    y_data = y_data[:, 0:1, :, :, :]
  elif obj == "lesion":
    y_data = y_data[:, 1:2, :, :, :]

  if normalization is not None and normalization.lower() != "none":
    mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
    x_data = channel_wise_normalize_3d(x_data, mean_std=mean_std)

  if anno_percent < 100:
    ind_list = [i for i in range(x_data.shape[0])]
    random.Random(99).shuffle(ind_list)

    num_data = int(x_data.shape[0] * anno_percent / 100.0)
    x_data = x_data[ind_list[:num_data], ...]
    y_data = y_data[ind_list[:num_data], ...]

  print("x_{}: {} | {:.2f} ~ {:.2f}".format(status, x_data.shape, np.min(x_data), np.max(x_data)))
  print("y_{}: {} | {:.2f} ~ {:.2f}".format(status, y_data.shape, np.min(y_data), np.max(y_data)))

  return x_data, y_data


# ----------------------------------------------Downstream PE 3D----------------------------------------------
def PE_3D(data_dir, normalization=None, hu_range=(-1000.0, 1000.0), status="train", anno_percent=100, seed=None):
  hu_min, hu_max = hu_range[0], hu_range[1]

  if status == "train":
    x_data = np.load(os.path.join(data_dir, "pe-gt-voxels-features-tr-hu.npy"))
    y_data = np.load(os.path.join(data_dir, "pe-gt-voxels-labels-tr.npy"))

    validation_rate = 0.2
    idx_list = [i for i in range(x_data.shape[0])]
    random.Random(seed).shuffle(idx_list)

    x_train = x_data[idx_list[int(round(x_data.shape[0] * validation_rate)):]]
    y_train = y_data[idx_list[int(round(y_data.shape[0] * validation_rate)):]]
    x_train = np.expand_dims(x_train, axis=1)
    x_train[x_train > hu_max] = hu_max
    x_train[x_train < hu_min] = hu_min
    x_train = 1.0 * (x_train - hu_min) / (hu_max - hu_min)

    x_valid = x_data[idx_list[:int(round(x_data.shape[0] * validation_rate))]]
    y_valid = y_data[idx_list[:int(round(y_data.shape[0] * validation_rate))]]
    x_valid = np.expand_dims(x_valid, axis=1)
    x_valid[x_valid > hu_max] = hu_max
    x_valid[x_valid < hu_min] = hu_min
    x_valid = 1.0 * (x_valid - hu_min) / (hu_max - hu_min)

    # augmentation
    x, y = [], []
    for i in range(x_train.shape[0]):
      if y_train[i] == 1:
        for b in range(13, 19):
          degree = random.choice([0, 1, 2, 3])
          if degree == 0:
            x.append(x_train[i, :, :, :, b:b + 32])
          else:
            x.append(np.flip(x_train[i, :, :, :, b:b + 32], axis=degree))
          y.append(y_train[i])
      else:
        x.append(x_train[i, :, :, :, 16:48])
        y.append(y_train[i])
    x_train, y_train = copy.deepcopy(np.array(x)), copy.deepcopy(np.array(y))

    x, y = [], []
    for i in range(x_valid.shape[0]):
      if y_valid[i] == 1:
        for b in range(13, 19):
          degree = random.choice([0, 1, 2, 3])
          if degree == 0:
            x.append(x_valid[i, :, :, :, b:b + 32])
          else:
            x.append(np.flip(x_valid[i, :, :, :, b:b + 32], axis=degree))
          y.append(y_valid[i])
      else:
        x.append(x_valid[i, :, :, :, 16:48])
        y.append(y_valid[i])
    x_valid, y_valid = copy.deepcopy(np.array(x)), copy.deepcopy(np.array(y))

    if normalization is not None and normalization.lower() != "none":
      mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
      x_train = channel_wise_normalize_3d(x_train, mean_std=mean_std)
      x_valid = channel_wise_normalize_3d(x_valid, mean_std=mean_std)

    if anno_percent < 100:
      ind_list = [i for i in range(x_train.shape[0])]
      random.Random(99).shuffle(ind_list)

      num_data = int(x_train.shape[0] * anno_percent / 100.0)
      x_train = x_train[ind_list[:num_data], ...]
      y_train = y_train[ind_list[:num_data], ...]

    print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
    print("y_train: {} | {:.2f} ~ {:.2f}".format(y_train.shape, np.min(y_train), np.max(y_train)))
    print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
    print("y_valid: {} | {:.2f} ~ {:.2f}".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

    return x_train, y_train, x_valid, y_valid
  else:
    x_test = np.load(os.path.join(data_dir, "pe-gt-voxels-features-te-hu.npy"))
    y_test = np.load(os.path.join(data_dir, "pe-gt-voxels-labels-te.npy"))

    x_test = np.expand_dims(x_test, axis=1)
    x_test[x_test > hu_max] = hu_max
    x_test[x_test < hu_min] = hu_min
    x_test = 1.0 * (x_test - hu_min) / (hu_max - hu_min)
    x_test = x_test[:, :, :, :, 16:48]

    if normalization is not None and normalization.lower() != "none":
      mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
      x_test = channel_wise_normalize_3d(x_test, mean_std=mean_std)

    print("x_test:  {} | {:.2f} ~ {:.2f}".format(x_test.shape, np.min(x_test), np.max(x_test)))
    print("y_test:  {} | {:.2f} ~ {:.2f}".format(y_test.shape, np.min(y_test), np.max(y_test)))

    return x_test, y_test

# ----------------------------------------------Downstream BraTS 3D----------------------------------------------
class BraTS_Seg_3D(Dataset):
  def __init__(self, data_dir, file, mode="train", modality="flair", input_size=(64, 64, 32), normalization=None,
               positives=[1, 2, 4], crop_size=(100, 100, 50), delta=30, anno_percent=100, seed=0):

    self.patient_list = []
    with open(file, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()
          self.patient_list.append(lineItems[0])

    indexes = np.arange(len(self.patient_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]

      _patient_list = copy.deepcopy(self.patient_list)
      self.patient_list = []

      for i in indexes:
        self.patient_list.append(_patient_list[i])
    self.indexes = np.arange(len(self.patient_list))

    import BraTS
    self.brats = BraTS.DataSet(brats_root=data_dir, year=2018)
    self.modality = modality
    self.positives = positives

    self.input_size = input_size
    self.crop_size = crop_size
    self.delta = delta
    self.normalization = normalization
    self.mode = mode

    if seed is None:
      self.seed = random.randint(0, 10000)
    else:
      self.seed = seed
    self.batch_generator = random.Random()
    self.batch_generator.seed(self.seed)
    self.patch_generator = random.Random()
    self.patch_generator.seed(self.seed)

  def load_patient(self, patient_id):
    patient = self.brats.train.patient(patient_id)

    # load images
    if self.modality == "flair":
      img = patient.flair
      img = (img - np.min(img)) * 1.0 / (np.max(img) - np.min(img))
    elif self.modality == "t1":
      img = patient.t1
      img = (img - np.min(img)) * 1.0 / (np.max(img) - np.min(img))
    elif self.modality == "t1ce":
      img = patient.t1ce
      img = (img - np.min(img)) * 1.0 / (np.max(img) - np.min(img))
    elif self.modality == "t2":
      img = patient.t2
      img = (img - np.min(img)) * 1.0 / (np.max(img) - np.min(img))
    else:
      print("Modality [{}] is not available!".format(self.modality))
      exit(0)

    # load segmentations
    seg = patient.seg
    for l in self.positives:
      seg[seg == l] = 255
    seg[seg != 255] = 0
    seg[seg == 255] = 1

    return img, seg

  def preprocessing(self, org_img, org_seg):

    labels = measure.label(np.squeeze(org_seg), neighbors=8, background=0)
    if len(np.unique(labels)) == 1:
      cx, cy, cz = self.patch_generator.randint(100, 140), \
                   self.patch_generator.randint(100, 140), \
                   self.patch_generator.randint(50, 105)
    else:
      for label in np.unique(labels):
        if label == 1:
          labelMask = np.zeros(np.squeeze(org_seg).shape, dtype="int")
          labelMask[labels == label] = 1
          break
      cx, cy, cz = ndimage.measurements.center_of_mass(labelMask)
      cx, cy, cz = int(cx), int(cy), int(cz)

    if self.mode != 'test' and self.patch_generator.random() < 0.8:
      cx += random.randint(-self.delta, self.delta)
      cy += random.randint(-self.delta, self.delta)
      cz += random.randint(-self.delta, self.delta)

    sx = min(max(0, cx - self.crop_size[0] // 2), org_img.shape[0] - 1 - self.crop_size[0])
    sy = min(max(0, cy - self.crop_size[1] // 2), org_img.shape[1] - 1 - self.crop_size[1])
    sz = min(max(0, cz - self.crop_size[2] // 2), org_img.shape[2] - 1 - self.crop_size[2])

    crop_img = org_img[sx:sx + self.crop_size[0], sy:sy + self.crop_size[1], sz:sz + self.crop_size[2]]
    crop_msk = org_seg[sx:sx + self.crop_size[0], sy:sy + self.crop_size[1], sz:sz + self.crop_size[2]]

    resized_img = resize(crop_img, self.input_size, preserve_range=True)
    resized_msk = resize(crop_msk, self.input_size, preserve_range=True)

    if self.mode != "test":
      resized_img, resized_msk = self.data_augmentation(resized_img, resized_msk)

    img = np.expand_dims(resized_img, axis=0)
    msk = np.expand_dims(resized_msk, axis=0)
    msk[msk < 0.5] = 0
    msk[msk >= 0.5] = 1

    if self.normalization is not None and self.normalization.lower() != "none":
      mean_std = NORMALIZATION_STATISTICS[self.normalization.lower()]
      img = channel_wise_normalize_3d(img, mean_std=mean_std)

    return img, msk

  def data_augmentation(self, img, seg):
    # rotation
    def flip(img, seg, axis):
      flipped_img = np.flip(img, axis=axis)
      flipped_seg = np.flip(seg, axis=axis)
      return flipped_img, flipped_seg

    for _ in range(3):
      if self.patch_generator.random() < 0.7:
        img, seg = flip(img, seg, axis=self.patch_generator.choice([0, 1, 2]))

    # add noise
    def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
      variance = self.patch_generator.uniform(noise_variance[0], noise_variance[1])
      data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
      return data_sample

    if self.patch_generator.random() < 0.2:
      img = augment_rician_noise(img, noise_variance=(0, 0.1))

    def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
      if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
      else:
        variance = self.patch_generator.uniform(noise_variance[0], noise_variance[1])
      data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
      return data_sample

    if self.patch_generator.random() < 0.2:
      img = augment_gaussian_noise(img, noise_variance=(0, 0.1))

    img[img < 0] = 0
    img[img > 1] = 1
    return img, seg

  def __getitem__(self, index):
    _img, _seg = self.load_patient(self.patient_list[index])
    img, seg = self.preprocessing(_img, _seg)
    return torch.FloatTensor(np.flip(img,axis=0).copy()), torch.FloatTensor(np.flip(seg,axis=0).copy())

  def __len__(self):
    return len(self.patient_list)

def BraTS_Seg_3D_NPY(data_dir, data_file, mask_file, input_size=(64, 64, 32), normalization=None, anno_percent=100):
  x_data = []
  y_data = []

  with open(data_file, 'r') as f:
    image_list = f.read().split('\n')
  with open(mask_file, 'r') as f:
    mask_list = f.read().split('\n')

  for img_data in image_list:
    if img_data == '':
      continue
    key = img_data.split('Synthetic_Data/')[1].split('/VSD')[0]
    mk_data = [line for line in mask_list if key in line]

    img_itk = sitk.ReadImage(os.path.join(data_dir, img_data))
    img_ary = sitk.GetArrayFromImage(img_itk)
    img = np.einsum('ijk->kji', img_ary)
    img = resize(img, (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2), preserve_range=True)
    img = img / 4096.0

    msk_itl = sitk.ReadImage(os.path.join(data_dir, mk_data[0]))
    msk_ary = sitk.GetArrayFromImage(msk_itl)
    msk = np.einsum('ijk->kji', msk_ary)
    msk[msk > 0] = 1
    msk = resize(msk, (msk.shape[0] // 2, msk.shape[1] // 2, msk.shape[2] // 2), preserve_range=True)
    msk[msk > 0.5] = 1
    msk[msk <= 0.5] = 0

    for i in range(0, img.shape[0] - input_size[0] + 1, input_size[0]):
      for j in range(0, img.shape[1] - input_size[1] + 1, input_size[1]):
        for k in range(0, img.shape[2] - input_size[2] + 1, input_size[2]):
          x_data.append(img[i:i + input_size[0], j:j + input_size[1], k:k + input_size[2]])
          y_data.append(msk[i:i + input_size[0], j:j + input_size[1], k:k + input_size[2]])

    cx, cy, cz = ndimage.measurements.center_of_mass(msk)
    cx, cy, cz = int(cx), int(cy), int(cz)
    for delta_x in range(-5, 10, 5):
      for delta_y in range(-5, 10, 5):
        for delta_z in range(-3, 6, 3):
          if cx + delta_x - (input_size[0]//2) < 0 or cx + delta_x + (input_size[0]//2) > img.shape[
            0] - 1 or \
              cy + delta_y - (input_size[1]//2) < 0 or cy + delta_y + (input_size[1]//2) > img.shape[
            1] - 1 or \
              cz + delta_z - (input_size[2]//2) < 0 or cz + delta_z + (input_size[2]//2) > img.shape[
            2] - 1:
            pass
          else:
            x_data.append(img[cx + delta_x - (input_size[0]//2):cx + delta_x + (input_size[0]//2), \
                     cy + delta_y - (input_size[1]//2):cy + delta_y + (input_size[1]//2), \
                     cz + delta_z - (input_size[2]//2):cz + delta_z + (input_size[2]//2)])
            y_data.append(img[cx + delta_x - (input_size[0]//2):cx + delta_x + (input_size[0]//2), \
                     cy + delta_y - (input_size[1]//2):cy + delta_y + (input_size[1]//2), \
                     cz + delta_z - (input_size[2]//2):cz + delta_z + (input_size[2]//2)])

  x_data = np.expand_dims(np.array(x_data), axis=1)
  y_data = np.expand_dims(np.array(y_data), axis=1)

  if normalization is not None and normalization.lower() != "none":
    mean_std = NORMALIZATION_STATISTICS[normalization.lower()]
    x_data = channel_wise_normalize_3d(x_data, mean_std=mean_std)

  if anno_percent < 100:
    ind_list = [i for i in range(x_data.shape[0])]
    random.Random(99).shuffle(ind_list)

    num_data = int(x_data.shape[0] * anno_percent / 100.0)
    x_data = x_data[ind_list[:num_data], ...]
    y_data = y_data[ind_list[:num_data], ...]

  print("x_data: {} | {:.2f} ~ {:.2f}".format(x_data.shape, np.min(x_data), np.max(x_data)))
  print("y_data: {} | {:.2f} ~ {:.2f}".format(y_data.shape, np.min(y_data), np.max(y_data)))

  return x_data, y_data


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = "3"

  # data_dir = "/mnt/dataset/shared/zongwei/BraTS"
  # file_list = "dataset/BraTS_val.txt"
  #
  # bra = BraTS_Seg_3D(data_dir, file_list, modality="flair", positives=[1, 2, 4],
  #                    mode="train", input_size=(64, 64, 32), crop_size=(100, 100, 50),
  #                    delta=30, normalization="self_learning_cubes_32")
  #
  # print(len(bra))
  # X, Y = bra.__getitem__(0)
  # print(X.shape)
  # print(Y.shape)

  data_dir = "/mnt/dataset/shared/mahfuz/"
  mask_list = "dataset/BRATS2013_segmentation.txt"

  data_list = "dataset/BRATS2013_train.txt"
  X,Y = BraTS_Seg_3D_NPY(data_dir, data_list, mask_list, input_size=(64, 64, 32),
                         normalization=None, anno_percent=100)

  data_list = "dataset/BRATS2013_val.txt"
  X, Y = BraTS_Seg_3D_NPY(data_dir, data_list, mask_list, input_size=(64, 64, 32),
                          normalization=None, anno_percent=100)

  data_list = "dataset/BRATS2013_test.txt"
  X, Y = BraTS_Seg_3D_NPY(data_dir, data_list, mask_list, input_size=(64, 64, 32),
                          normalization=None, anno_percent=100)

  data_list = "dataset/BRATS2013_train.txt"
  X, Y = BraTS_Seg_3D_NPY(data_dir, data_list, mask_list, input_size=(64, 64, 32),
                          normalization=None, anno_percent=90)


  # data_dir = "/mnt/dataset/shared/ruibinf/ChestX-ray14/images"
  # file_list = "dataset/Xray14_train_official.txt"
  # batch_size = 16
  # augment = Augmentation(normalize="chestx-ray").get_augmentation("{}_{}".format("full", 224), "train")
  #
  # xray = ChestX_ray14(data_dir, file_list, augment=augment)
  #
  # print(len(xray))
  # X, Y = xray.__getitem__(0)
  # print(X.shape)
  # print(Y.shape)
