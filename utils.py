import torch
import torch.nn.functional as F


def calculate_feature(reference_text, model, tokenizer, device, ref_img_features=None):
    with torch.no_grad(): 
        input_text = tokenizer(reference_text).to(device)
        text_features = model.encode_text(input_text)

        if ref_img_features != None:
            return F.normalize(text_features + ref_img_features)
        
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


def retrieveAll(calculated_features, index_features):
    distances = calculated_features @ F.normalize(index_features, dim=-1).float().T

    indices = torch.argsort(distances, dim=-1, descending=True)
    return indices


def retrieveTextToImage(num, reference_text, feature_dict, model, tokenizer, device):
    with torch.no_grad():  
        input_text = tokenizer(reference_text).to(device)

        text_features = model.encode_text(input_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # TODO: optimize distance calculation in the future 
        distance_list = []
        for key, image_features in feature_dict.items():  # iterate directly over items
            distance = (100.0 * image_features.float() @ text_features.T).item()
            distance_list.append((distance, key))

        distance_list.sort(reverse=True)

        del input_text
        del text_features

        ret_list = distance_list[:num]
        del distance_list  

        torch.cuda.empty_cache()  
        return ret_list


def retrieveComposed(
    num, ref_img_features, reference_text, feature_dict, model, tokenizer, device
):
    with torch.no_grad():  
        input_text = tokenizer(reference_text).to(device)

        text_features = model.encode_text(input_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        distance_list = []
        for key, image_features in feature_dict.items():  # iterate directly over items
            distance = (
                torch.nn.functional.normalize(image_features.float(), dim=-1)
                @ (
                    torch.nn.functional.normalize(
                        torch.add(ref_img_features, text_features), dim=-1
                    ).float()
                ).T
            ).item()
            distance_list.append((distance, key))

        distance_list.sort(reverse=True)

        del input_text
        del text_features

        ret_list = distance_list[:num]
        del distance_list  

        torch.cuda.empty_cache()  
        return ret_list


import json


def meanAveragePrecisionAtK(K, gt_ids, output_ids):
    sum = 0
    positives_by_k = 0
    for k in range(K):
        if output_ids[k] in gt_ids:  # check positivity at rank k
            positives_by_k = positives_by_k + 1
            sum += positives_by_k / (k + 1)
    return sum / min(K, len(gt_ids))


def recallAtK(K, gt_ids, output_ids):
    found = 0
    for k in range(K):
        if output_ids[k] in gt_ids:
            found += 1
    return found / len(gt_ids)
