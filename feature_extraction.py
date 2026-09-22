import torch
def extract_and_save_features (data_loader, model, save_path, device):
    image_features_list = dict()

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, (img_names_batch, img_batch) in enumerate(data_loader):
            img_batch = img_batch.to(device)
            image_features = model.encode_image(img_batch).to(device)
            
            batch_size = image_features.size(0)
            for i in range(batch_size):
                image_features_list[img_names_batch[i]] = image_features[i]


    torch.save(image_features_list, save_path)
    return image_features_list