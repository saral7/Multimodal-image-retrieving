from data_utils import AnnotationsReader, AnnotationLine
import json
from torch.utils.data import Dataset
import os
from PIL import Image

class Basic (Dataset):
    def __init__(self, images_path, preprocess):
        self.images_path = images_path
        self.preprocess = preprocess     
        self.lines = os.listdir(images_path)   
        self.lines = [os.path.basename(x) for x in self.lines]

    def __getitem__(self, index):
        reference_image_name = self.lines[index][0 : self.lines[index].rindex(".")]
        ref_img = self.preprocess(Image.open(os.path.join(self.images_path, self.lines[index])))
        return reference_image_name, ref_img
    
    def __len__ (self):
        return len(self.lines)

class MyDataset(Dataset):
    def __init__(self, image_path, annotations_path):
        self.image_path = image_path
        self.annotations_path = annotations_path
        return

    def readlines(self):
        # create lines from file
        annotations_file = open(self.annotations_path, "r")
        annotations = json.load(annotations_file)
        annotations_file.close()

        lines = []

        for line in annotations["images"]:
            try:
                annotation_line = self.read(line)
            except:
                print(f"Line {line} cannot be parsed, skipping...")
                continue

            if annotation_line:
                lines.append(annotation_line)
        return lines

    def read(self, line):
        # read and parse lines, return AnnotationLine object
        reference_image_name = line["reference_img"][
            line["reference_img"].rindex("/") + 1 : len(line["reference_img"])-4
        ]
        caption = line["relative_caption"].removeprefix("* ")  # for composed
        # caption = line["description"]  # for text-to-image
        target_image_list = [
            line["target_img"][
                line["target_img"].rindex("/") + 1 : len(line["target_img"])-4
            ]
        ]
        return reference_image_name, target_image_list[0], caption
        
    
class MyDatasetVal (MyDataset):
    def __init__(self, image_path, annotations_path):
        super().__init__(image_path, annotations_path)
        self.lines = self.readlines()

    def __getitem__(self, index):
        return self.lines[index]

    def __len__ (self):
        return len(self.lines)