from data_utils import AnnotationsReader, AnnotationLine


class Flickr30kAnnotations(AnnotationsReader):
    def __init__(self, annotations_path, split_path):
        super().__init__(annotations_path, split_path)
        self.lines = self.readlines()
        return

    def readlines(self):
        # create lines from file
        annotations_file = open(self.annotations_path, "r")
        annotations_file.readline()  # skipping description
        annotations = annotations_file.readlines()
        annotations_file.close()
        lines = []
        if self.split_path:
            split_file = open(self.split_path)
            split_images = split_file.readlines()
            split_file.close()
            split_images = [x.strip() for x in split_images]
            self.split_images = split_images

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

        line = line.split("|")
        reference_image_name = line[0].strip()
        caption = [line[2].strip()]

        if self.split_path and not (
            reference_image_name.removesuffix(".jpg") in self.split_images
        ):
            return None

        target_image_list = [reference_image_name]
        return AnnotationLine(reference_image_name, target_image_list, caption)
