from data_utils import AnnotationsReader, AnnotationLine
import json


def padImageName(imageName):
    imageName = "0" * (12 - len(str(imageName))) + str(imageName) + ".jpg"
    return imageName


def stripImage(imageName):
    return int(imageName[:12])


class CircoAnnotations(AnnotationsReader):
    def __init__(self, annotations_path):
        super().__init__(annotations_path)
        self.lines = self.readlines()
        return

    def readlines(self):
        # create lines from file
        annotations_file = open(self.annotations_path, "r")
        annotations = json.load(annotations_file)
        annotations_file.close()
        lines = []

        for line in annotations:
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
        reference_image_name = padImageName(line["reference_img_id"])
        caption = line["relative_caption"]

        target_image_list = [
            padImageName(gt_img_id) for gt_img_id in line["gt_img_ids"]
        ]
        return AnnotationLine(reference_image_name, target_image_list, caption)
