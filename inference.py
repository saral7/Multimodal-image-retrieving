import argparse
import open_clip
import torch
import os
import json
import numpy as np
import torch.nn.functional as F

from annotations_classes.flickr30k_annotations import Flickr30kAnnotations
from annotations_classes.circo_annotations import CircoAnnotations
from annotations_classes.fashioniq_annotations import FashionIQAnnotations
from annotations_classes.mydataset_annotations import MyDatasetAnnotations

from utils import (
    calculate_feature,
    retrieveAll,
)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        required=True,
        choices=["flickr30k", "fashioniq", "circo", "mydataset"],
    )
    argparser.add_argument("--features_path", required=True)
    argparser.add_argument("--annotations_path", required=True)
    argparser.add_argument("--split_file_path")
    argparser.add_argument(
        "--retrieval_type", choices=["composed", "text-to-image"], required=True
    )
    argparser.add_argument("--model_name", required=True)
    argparser.add_argument("--pretraining", required=True)
    argparser.add_argument(
        "--metric", choices=["recall_at_k"], default="recall_at_k"
    )

    args = argparser.parse_args()

    annotations_path = args.annotations_path

    annotations = None
    if args.dataset == "flickr30k":
        split_path = args.split_file_path
        annotations = Flickr30kAnnotations(annotations_path, split_path)
    if args.dataset == "circo":
        annotations = CircoAnnotations(annotations_path)
    if args.dataset == "fashioniq":
        split_path = args.split_file_path
        annotations = FashionIQAnnotations(annotations_path)
    if args.dataset == "mydataset":
        annotations = MyDatasetAnnotations(annotations_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelName = args.model_name
    pretraining = args.pretraining
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=modelName, pretrained=pretraining
    )
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(modelName)

    # cached features of the images in the database
    feature_dict = torch.load(
        os.path.join(args.features_path, f"{modelName}-{pretraining}_mydataset_image_features.pt"),
        map_location=device,
    )
    index_features_list, index_names = list(feature_dict.values()), list(
        feature_dict.keys()
    )
    featuredim = model.visual.output_dim
    index_features = torch.vstack(index_features_list)

    print("# of annotations: ", len(annotations))

    calculated_features = torch.empty((0, featuredim)).to(device, non_blocking=True)
    target_names = []
    for i in range(len(annotations)):
        reference_image_name = annotations[i].reference_image_name.removesuffix(".jpg")
        caption = annotations[i].caption
        target_image_list = annotations[i].target_image_list

        target_names.append(target_image_list[0].removesuffix(".jpg"))
        if args.retrieval_type == "text-to-image":
            curr_calculated_feature = calculate_feature(
                caption, model, tokenizer, device
            )

        if args.retrieval_type == "composed":
            reference_image_features = feature_dict[reference_image_name].removesuffix(".jpg")
            curr_calculated_feature = calculate_feature(
                caption, model, tokenizer, device, reference_image_features
            )

        calculated_features = torch.vstack(
            (calculated_features, F.normalize(curr_calculated_feature))
        )

    indices = retrieveAll(calculated_features, index_features)
    sorted_index_names = np.array(index_names)[indices]

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )

    K_list = [1, 2, 5, 10, 25, 50]
    sum_for_k = dict((k, 0) for k in K_list)

    for k in K_list:
        if args.metric == "recall_at_k":
            sum_for_k[k] += labels[:, :k].sum().item()

    print(
        f"Result: {args.retrieval_type} retrieval for {args.model_name} ({args.pretraining}): \n({args.metric})",
        sum_for_k / len(annotations),
    )


if __name__ == "__main__":
    main()
