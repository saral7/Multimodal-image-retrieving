from torch.utils.data import Dataset 
import os
import json
import numpy as np
from PIL import Image


class FashionIQ (Dataset):
    def __init__ (self, images_path, annotations_path, dress_types, preprocess = None):
        self.images_path = images_path
        self.images = []
        self.annotations = []
        self.annotations_path = annotations_path
        self.dress_types = dress_types
        self.preprocess = preprocess


    def read(self, line):
        reference_image_name = line["candidate"] 
        caption = [line["captions"][0] + " and " + line["captions"][1]]
        flattened_captions: list = np.array(line["captions"]).T.flatten().tolist()
        caption = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)
        ]

        target_image_list = [line["target"]] 

        return reference_image_name, target_image_list[0], caption[0]
    
    def __len__ (self):
        return len(self.lines)


class FashionIQBasic (Dataset):
    def __init__(self, images_path, split_path, dress_types, preprocess, mode):
        self.dress_types = dress_types
        self.images_path = images_path
        self.split_path = split_path
        self.preprocess = preprocess
        self.lines = []
        self.mode = mode
        for type in self.dress_types:
            split_file = open(os.path.join(self.split_path, f"split.{type}.{self.mode}.json"), "r")
            self.lines.extend(json.load(split_file))
            split_file.close()

    def __getitem__(self, index):
        reference_image_name = self.lines[index]

        ref_img = self.preprocess(Image.open(os.path.join(self.images_path, reference_image_name + ".png")))
        
        return reference_image_name, ref_img
    
    def __len__ (self):
        return len(self.lines)
    

class FashionIQValDataset(FashionIQ):
    def __init__(self, images_path, annotations_path, dress_types, preprocess = None):
        super().__init__(images_path, annotations_path, dress_types, preprocess)
        self.lines = self.readlines()

    def readlines(self):
        # create lines from file
        for type in self.dress_types:
            annotations_file = open(os.path.join(self.annotations_path, f"cap.{type}.val.json"), "r")
            self.annotations.extend(json.load(annotations_file))
            annotations_file.close()

        lines = []

        for line in self.annotations:
            try:
                annotation_line = self.read(line)
            except:
                print(f"Line {line} cannot be parsed, skipping...")
                continue

            if annotation_line:
                lines.append(annotation_line)
        return lines
        
    def __getitem__(self, index):
        return self.lines[index]
    
class FashionIQTrainDataset(FashionIQ):
    def __init__(self, images_path, annotations_path, dress_types, preprocess = None):
        super().__init__(images_path, annotations_path, dress_types, preprocess)
        self.lines = self.readlines()
        
    def readlines(self):
        # create lines from file
        for type in self.dress_types:
            annotations_file = open(os.path.join(self.annotations_path, f"cap.{type}.train.json"), "r")
            self.annotations.extend(json.load(annotations_file))
            annotations_file.close()

        lines = []

        for line in self.annotations:
            try:
                annotation_line = self.read(line)
            except:
                print(f"Line {line} cannot be parsed, skipping...")
                continue

            if annotation_line:
                lines.append(annotation_line)
        return lines
    
    def __getitem__(self, index):
        reference_image_name, target_image_name, caption = self.lines[index]

        ref_img = self.preprocess(Image.open(os.path.join(self.images_path, reference_image_name + ".png")))
        target_img = self.preprocess(Image.open(os.path.join(self.images_path, target_image_name + ".png")))
        
        return ref_img, target_img, caption
