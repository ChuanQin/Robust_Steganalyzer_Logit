import numpy as np
from sklearn.svm import SVC
import pickle
import torch
import models
import data
from utils import get_CNN_prob
import subprocess

def train_Rough_Filter(
    CNN_prob,
    hand_prob,
    labels,
    names,
    save_path):
    # concatenate probabilities for predictions
    prob = np.stack((hand_prob, CNN_prob), axis = 1)
    # # for the images with CNN predicting as cover while Handcrafted feature based model predicting as stego
    # suspiciousI = names[np.where((prob[:,0]<0.5)&(prob[:,1]>=0.5))]
    # # for the images being predicted stego by the CNN model
    # reliableI = names[np.where(prob[:,1]<0.5)]
    # for the images with CNN and Handcrafted feature based model predicting as cover
    both_cover = np.asarray(list(set(np.where(prob[:,0]>=0.5)[0])&set(np.where(prob[:,1]>=0.5)[0])))

    return rough_filter

def set_dataloader(args):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
    ])
    cover_data = data.ImageWithNameDataset(
                    img_dir = args.cover_dir, 
                    transform = transform)
    stego_data = data.ImageWithNameDataset(
                    img_dir = args.stego_dir, 
                    transform = transform)
    adv_data = data.ImageWithNameDataset(
                    img_dir = args.adv_dir, 
                    ref_dir = args.cover_dir, 
                    transform = transform)
    cover_loader = torch.utils.data.DataLoader(
                    cover_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    stego_loader = torch.utils.data.DataLoader(
                    stego_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)
    adv_loader = torch.utils.data.DataLoader(
                    adv_data, 
                    batch_size = args.batch_size, 
                    num_workers = args.num_workers)

    return cover_loader, stego_loader, adv_loader

def set_model(args):
    net = nn.DataParallel(models.KeNet())
    ckpt_path = args.ckpt_dir + '/model_best.pth.tar'
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['state_dict'])

    return net

net = set_model(args)

cover_loader, stego_loader, adv_loader = set_dataloader(args)

cover_cnn_probs, cover_cnn_names = get_CNN_prob(net, cover_loader)
stego_cnn_probs, stego_cnn_names = get_CNN_prob(net, stego_loader)
adv_cnn_probs, adv_cnn_names = get_CNN_prob(net, adv_loader)

cover_hand_probs, cover_hand_names = get_hand_prob(args.cover_feature_path, args.clf_path, args.cover_dir)
stego_hand_probs, stego_hand_names = get_hand_prob(args.stego_feature_path, args.clf_path, args.stego_dir)
adv_hand_probs, adv_hand_names = get_hand_prob(args.adv_feature_path, args.clf_path, args.adv_dir)

