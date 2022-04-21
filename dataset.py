import os
from os.path import join

from typing import List

from cycle_with_distribution.data.base_dataset import BaseDataset
from cycle_with_distribution.data.image_folder import make_dataset
from PIL import Image
import random

from cycle_with_distribution.options.base_options import BaseOptions


class SampleClassPercentDataset(BaseDataset):
    """
    This dataset class can load the SAMPLE dataset.
    It requires two directories to host training images from domain synthetic '/SAMPLE_dataset_public/png_images/qpm/synth'
    and from domain measured '/SAMPLE_dataset_public/png_images/qpm/real' respectively.
    You can train the model with the dataset flag '--dataroot /SAMPLE_dataset_public/png_images/qpm'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'synth')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'real')  # create a path '/path/to/data/trainB'
        self.class_list = opt.class_list
        self.missing_class_num = opt.missing_class_num
        self.adjusted_class = BaseOptions.class_list
        self.percent_missing = opt.percent_missing

        print(self.dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        if opt.phase == 'train':
            self.A_paths = [
                join(root, image)
                for root, image in [os.path.split(path) for path in self.A_paths]
                if image.split('_')[4] != '017'
            ]
            ending_b_list = []
            ending_b_label_list = []
            for index, label in enumerate(self.class_list):
                prereduced_b_list: List[Image] = []
                for image in self.B_paths:
                    root, image = os.path.split(image)
                    elevation: str = image.split('_')[4]
                    class_name = image.split('_')[0]
                    class_number = int(self.class_list.index(class_name))
                    if elevation != '017' and class_number != self.missing_class_num and class_name == label:
                        prereduced_b_list.append(join(root, image))
                temp_b_list = self.find_reduced_list(prereduced_b_list)
                temp_label_list = [index] * len(temp_b_list)
                ending_b_list.extend(temp_b_list)
                ending_b_label_list.extend(temp_label_list)
                print(len(ending_b_list))
            print(ending_b_label_list)
            print(len(ending_b_label_list))

            self.B_paths = ending_b_list
            self.B_labels = ending_b_label_list
            self.A_labels = [
                self.class_list.index(image_name.split('_')[0])
                for root, image_name in [os.path.split(path) for path in self.A_paths]
                if image_name.split('_')[4] != '017'
            ]  # splitting file names to grab class labels

            self.A_labeled_dict = {}

            for index, classes in enumerate(self.class_list):
                classes_list = [
                    join(root, image)
                    for root, image in [os.path.split(path) for path in self.A_paths]
                    if image.split('_')[0] == classes
                ]
                self.A_labeled_dict.update({index: classes_list})

        else:
            self.A_paths = [image for image in self.A_paths if image.split('_')[4] == '017']
            self.B_paths = [image for image in self.B_paths if image.split('_')[4] == '017']
            self.A_labels = [
                self.class_list.index(image.split('_')[3].split('\\')[3])
                for image in self.A_paths if image.split('_')[4] == '017'
            ]
            self.B_labels = [self.class_list.index(image.split('_')[3].split('\\')[3]) for image in self.B_paths if
                             image.split('_')[7] == '017']

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = (self.opt.direction == 'BtoA')
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = opt.transform
        self.transform_B = opt.transform

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain (synthetic)
            B (tensor)       -- its corresponding image in the target domain (measured)
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range+
        index_B = index
        # todo FIX THIS ^_^
        # if self.opt.serial_batches:  # make sure index is within then range
        #     index_B = index % self.B_size
        # else:  # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        B_paired = self.transform_B(B_img)
        # label grabber
        A_lab = self.A_labels[index % self.A_size]
        B_lab = self.B_labels[index_B % self.B_size]

        possiable_A_pairs = self.A_labeled_dict[B_lab]
        A_paired_index = random.randint(0, len(possiable_A_pairs) - 1)
        A_paired_path = possiable_A_pairs[A_paired_index]
        A_paired_picture = Image.open(A_paired_path)
        A_paired = self.transform_A(A_paired_picture)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_lab,
                'B_label': B_lab, 'A_paired': A_paired, 'B_paired': B_paired,
                'paired_label': B_lab}  # added labels to ditionary

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def find_reduced_list(self, b_paths):
        new_b_paths = b_paths[:int(len(b_paths) * (1.0 - self.percent_missing))]

        return new_b_paths

