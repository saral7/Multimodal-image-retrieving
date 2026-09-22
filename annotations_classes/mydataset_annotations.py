from data_utils import AnnotationsReader, AnnotationLine
import json


class MyDatasetAnnotations(AnnotationsReader):
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

        for line in annotations["images"]:
            try:
                annotation_line = self.read(line)
                print(annotation_line.caption)
            except:
                print(f"Line {line} cannot be parsed, skipping...")
                continue

            if annotation_line:
                lines.append(annotation_line)
        return lines

    def read(self, line):
        # read and parse lines, return AnnotationLine object
        reference_image_name = line["reference_img"][
            line["reference_img"].rindex("/") + 1 : len(line["reference_img"])
        ]
        #caption = line["relative_caption"].removeprefix("* ")  # za composed
        caption = line["description"]  # za text-to-image
        target_image_list = [
            line["target_img"][
                line["target_img"].rindex("/") + 1 : len(line["target_img"])
            ]
        ]
        return AnnotationLine(
            reference_image_name, [reference_image_name], caption
        )  # for text-to-image
        return AnnotationLine(
            reference_image_name, target_image_list, caption
        )  # for composed
