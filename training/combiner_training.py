from torch.utils.data import DataLoader
import torchvision
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import os
sys.path.append(os.path.relpath(".."))
from dataset_classes.fashioniq import FashionIQValDataset, FashionIQTrainDataset, FashionIQBasic
from combiner import Combiner
from feature_extraction import extract_and_save_features
import datetime
from argparse import ArgumentParser
import logging
from torch.optim.lr_scheduler import StepLR


def train_step(train_loader : DataLoader, model, combiner, loss_criterion, optimizer, epoch_idx, device, tokenizer, log_freq):
    combiner.train()

    logging.info("Training step started...")
    for batch_idx, (ref_batch, target_batch, captions_batch) in enumerate(train_loader):
        # ref_batch consists of reference images (preprocessed), 
        # target batch of target images (preprocessed), 
        # and captions_batch of captions text
        
        ref_batch = ref_batch.to(device)
        target_batch = target_batch.to(device)
        captions_batch = tokenizer(captions_batch).to(device)

        optimizer.zero_grad()   

        with torch.amp.autocast("cuda"):
            # extract features from frozen CLIP
            ref_features =  torch.vstack([model.encode_image(ref_batch)]).to(device)
            target_features =  torch.vstack([model.encode_image(target_batch)]).to(device)
            text_features =  torch.vstack([model.encode_text(captions_batch)]).to(device)
            batch_size = ref_features.size(0)
            
            # calculated combined, meaning visual features of a reference image with the text features of the modification,
            # given combining function is applied (in case of combiner training, it is its combining function, but can also be e.g. torch.sum)
            combined_features = combiner(ref_features, text_features, target_features).to(device)   # batch_size x feature_dim

            targets = torch.Tensor([i for i in range(batch_size)])     
            targets = targets.to(torch.long).to(device)
            
            # calculate loss
            loss = loss_criterion(combined_features, targets) / batch_size
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        if batch_idx % log_freq == 0:
            logging.info("epoch: %s, batch_idx: %s (%.2f%%), loss: %f" % (epoch_idx, batch_idx+1, (batch_idx+1) * 100. / len(train_loader), loss))

        del ref_features, target_features, text_features, targets, combined_features
        torch.cuda.empty_cache()


def validation_step (val_loader : DataLoader, model, combining_function, device, feature_dict, tokenizer, epoch_idx=None, dress_type=None, loss_criterion=None):
    feature_dim = model.visual.output_dim 
    calculated_features = torch.empty((0, feature_dim)).to(device)

    logging.info(f"Validating started... {dress_type}" if dress_type != None  else "Validating started...")
    target_names = []
    with torch.no_grad():
        for batch_idx, (ref_batch, target_batch, captions_batch) in enumerate(val_loader):
            # ref_batch consists of reference images names, 
            # target batch of target images names, and captions_batch of captions text
            
            # extract features from frozen CLIP
            ref_features =  torch.vstack([feature_dict[img_name] for img_name in ref_batch]).to(device)
            text_features =  torch.vstack([model.encode_text(tokenizer(t).to(device)) for t in captions_batch]).to(device)

            with torch.amp.autocast("cuda"):
                # calculated combined, meaning visual features of a reference image with the text features of the modification,
                # given combining function is applied (in case of combiner training, it is its combining function, but can also be e.g. torch.sum)
                combined_features = combining_function(ref_features, text_features).to(device)   # batch_size x feature_dim

                # create a list of calculated features
                calculated_features = torch.vstack((calculated_features, combined_features)).to(device)
                target_names.extend(target_batch)

            del ref_features, text_features, combined_features
            torch.cuda.empty_cache()

    # validation database we will retrieve from: 
    # dictionary (key, value) = (image_name, clip visual features of that image)
    index_features_list, index_names = list(feature_dict.values()), list(
        feature_dict.keys()
    )
    
    index_features = torch.vstack(index_features_list)
    
    with torch.amp.autocast("cuda"):
        similarity = F.normalize(calculated_features, dim=-1) @ F.normalize(index_features, dim=-1).T   # len(validation annotation) x len(database)
        indices = torch.argsort(similarity, dim=-1, descending=True)                                    # sort descending (biggest similarity first)

    name_order = np.array(index_names)[indices.cpu()]         # len(validation) x len(database)

    labels = torch.tensor(
        name_order
        == np.repeat(np.array(target_names), len(index_names)).reshape(    
            len(target_names), -1
        )
    )

    K_list = [1, 2, 5, 10, 25, 50]
    sum_for_k = dict((k, 0) for k in K_list)


    # calculate recall at k
    for k in K_list:
        sum_for_k[k] += labels[:, :k].sum().item()
        sum_for_k[k] /= len(val_loader)

    del similarity, indices, name_order, labels, calculated_features
    torch.cuda.empty_cache()

    if epoch_idx != None:
      logging.info(f"Validating... epoch: {epoch_idx}, recall: {sum_for_k}")
    else:
      logging.info(f"Validating... recall: {sum_for_k}")



def train_network (train_loader, val_loader_list, model, combiner, loss_criterion, optimizer, device, epoch_num, feature_dict_list, tokenizer, log_freq, model_save_path, scheduler):
    params = {
        "model" : model,
        "device" : device,
        "tokenizer" : tokenizer
    }
    
    for i in range (len(val_loader_list)):
        val_loader = val_loader_list[i]
        combiner.eval()
        validation_step(**params, combining_function=combiner.combine_features, val_loader=val_loader, feature_dict=feature_dict_list[i], dress_type=val_dress_types[i])
        
    
    for epoch_idx in range(epoch_num):
        train_step(**params, combiner=combiner, train_loader=train_loader, epoch_idx=epoch_idx, loss_criterion=loss_criterion, optimizer=optimizer, log_freq=log_freq)
        for i in range(len(val_loader_list)):
            val_loader = val_loader_list[i]
            combiner.eval()
            validation_step(**params, combining_function=combiner.combine_features, val_loader=val_loader, epoch_idx=epoch_idx, feature_dict=feature_dict_list[i], dress_type=val_dress_types[i])
        torch.save(combiner.state_dict(), os.path.join(model_save_path, f"{date}.pt"))
            
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        logging.info(f'Learning Rate: {current_lr}')



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    argparser = ArgumentParser()
    argparser.add_argument("--images_path")
    argparser.add_argument("--annotations_path")
    argparser.add_argument("--features_path")
    argparser.add_argument("--split_path")
    argparser.add_argument("--model_save_path")

    args = argparser.parse_args()
    
    logging.shutdown()
    date = datetime.datetime.now()
    log_path = os.path.join("./logs", str(date) + ".log")

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    modelName = "ViT-L-14"
    pretraining = "openai"
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=modelName, pretrained=pretraining
    )
    tokenizer = open_clip.get_tokenizer(modelName)
    
    preprocess = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=512),
         torchvision.transforms.CenterCrop(size=(512,512)),
         torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(modelName)

    dress_types = ["toptee", "shirt", "dress"]          # subcategories whose images are used in training
    val_dress_types = ["shirt", "dress", "toptee"]      # subcategories whose images are used in validation
    
    # train dataset, consists of triplets (reference_image, target_image, caption)
    # passed through the CLIP each time
    train_dataset = FashionIQTrainDataset(args.images_path, args.annotations_path, dress_types, preprocess) 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # validation dataset, consists of triplet (reference_image_name, target_image_name, caption)
    # uses precalculated features, loaded in feature_dict variable 
    # (that is the "database" consisting of images from val annotations, as well as many negatives)
    val_loader_list = []
    for val_type in val_dress_types:
        val_dataset = FashionIQValDataset(args.images_path, args.annotations_path, [val_type], preprocess)   
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        val_loader_list.append(val_loader)

    # top layer we are trying to learn
    feature_dim = model.visual.output_dim
    combiner = Combiner(feature_dim, 128, 256)   
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total model", total_params)
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total model frozen", total_params)
    total_params = sum(p.numel() for p in combiner.parameters() if p.requires_grad)

    combiner.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(combiner.parameters(), lr=lr)
    epoch_num = 300
    scaler = torch.amp.GradScaler('cuda')
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    logging.info(f"Hyperparameters: Combiner({combiner.text_projection_layer.in_features}, {combiner.text_projection_layer.out_features}, {combiner.combiner_layer.out_features}), {total_params} params,\nlr={lr}, train_dataset_size={len(train_dataset)}, val_dataset_size={len(val_dataset)}, train_dress_types={dress_types}, val_dress_types={val_dress_types}")

    feature_dict_list = [dict(), dict(), dict()]
    
    # used to precompute validation image features, to cache
    for i in range(len(val_dress_types)):
      type = val_dress_types[i]
      basic_dataset = FashionIQBasic(args.images_path, args.split_path, [type], preprocess, "val")
      basic_loader = DataLoader(basic_dataset, batch_size=32, shuffle=False, num_workers=0)
      feature_save_path = os.path.join(args.features_path, f"{modelName}-{pretraining}-{type}_fashioniq_image_features.pt")
      logging.info(f"Extracting features for {type}...")
      curr_dict = extract_and_save_features(basic_loader, model, feature_save_path, device)
      feature_dict_list[i].update(curr_dict)

      del basic_dataset, basic_loader 
    
    # loading the precalculated features for the validation database
    for i in range(len(val_dress_types)):
        type = val_dress_types[i]
        feature_dict_list[i].update(torch.load(os.path.join(args.features_path, f"{modelName}-{pretraining}-{type}_fashioniq_image_features.pt"), map_location=device))

    torch.cuda.empty_cache()
    
    logging.info("Combiner")
    f1 = (lambda x,y : F.normalize(x)+F.normalize(y))
    f2 = (lambda x,y : x+y)
    f_t2i = (lambda x,y : y)
    for i in range(len(val_loader_list)):
        combiner.eval()
        validation_step(val_loader_list[i], model, f2, device=device, feature_dict = feature_dict_list[i], tokenizer=tokenizer, dress_type=val_dress_types[i], loss_criterion=loss_criterion)
   
    train_network(train_loader, val_loader_list, model, combiner, loss_criterion, optimizer, device, epoch_num, feature_dict_list, tokenizer, 10, args.model_save_path, scheduler=scheduler)
