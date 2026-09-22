"""
Each line:
- reference image name, with extension
- target image list
- captions
- caption embedding (optional) - ??
- split path (optional)
"""


class AnnotationLine:
    def __init__(self, reference_image_name, target_image_list, captions):
        self.reference_image_name = reference_image_name
        self.target_image_list = target_image_list
        self.caption = captions


class AnnotationsReader:
    def __init__(self, annotations_path, split_path=None):
        self.annotations_path = annotations_path
        self.lines = []
        self.caption_embedding = None
        self.split_path = split_path
        return

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def append(self, annotation_line):
        self.lines.append(annotation_line)

    def set_caption_embedding(self, caption_embedding):
        self.caption_embedding = caption_embedding
