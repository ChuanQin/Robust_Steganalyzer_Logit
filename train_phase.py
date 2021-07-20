import numpy as np
from sklearn.svm import SVC
import pickle
import torch
import torchvision
import argparse
import torch.nn as nn
import os
import models
import data
from utils import get_CNN_prob, get_hand_prob, align_cnn_hand_data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cover_dir', type=str, required=True,)
    parser.add_argument('--stego_dir', type=str, required=True,)
    parser.add_argument('--adv_dir', type=str, required=True,)
    parser.add_argument('--cover_feature_path', type=str, required=True,)
    parser.add_argument('--stego_feature_path', type=str, required=True,)
    parser.add_argument('--adv_feature_path', type=str, required=True,)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--clf_path', type=str, required=True)
    parser.add_argument('--rough_filter_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--kernel', type=str, default='rbf', required=True)
    parser.add_argument('--gamma', type=float, default=50, required=True)
    parser.add_argument('--C', type=float, default=20, required=True)

    args = parser.parse_args()
    return args

def train_Rough_Filter(
    CNN_prob,
    hand_prob,
    labels,
    save_path,
    kernel = 'rbf',
    gamma = 50,
    C = 20):
    # concatenate probabilities for predictions
    prob = np.stack((hand_prob, CNN_prob[:,0]), axis = 1)
    # # for the images with CNN predicting as cover while Handcrafted feature based model predicting as stego
    # suspiciousI = names[np.where((prob[:,0]<0.5)&(prob[:,1]>=0.5))]
    # # for the images being predicted stego by the CNN model
    # reliableI = names[np.where(prob[:,1]<0.5)]
    # for the images with CNN and Handcrafted feature based model predicting as cover
    both_cover = np.asarray(list(set(np.where(prob[:,0]>=0.5)[0])&set(np.where(prob[:,1]>=0.5)[0])))
    train_labels = labels[both_cover]
    train_probs = prob[both_cover,:]
    rough_filter = SVC(kernel = 'rbf', gamma = 50, C = 20)
    rough_filter.fit(train_probs, train_labels)
    rough_filter_dir = '/'.join(save_path.split('/')[:-1])
    if not os.path.exists(rough_filter_dir):
        os.makedirs(rough_filter_dir)
    pickle.dump(rough_filter, open(save_path, 'wb'))

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
    net.eval()

    return net

args = parse_args()
net = set_model(args)
cover_loader, stego_loader, adv_loader = set_dataloader(args)

cover_cnn_probs, cover_cnn_names = get_CNN_prob(net, cover_loader)
stego_cnn_probs, stego_cnn_names = get_CNN_prob(net, stego_loader)
adv_cnn_probs, adv_cnn_names = get_CNN_prob(net, adv_loader)

cover_hand_probs, cover_hand_names = get_hand_prob(args.cover_feature_path, args.clf_path, args.cover_dir)
stego_hand_probs, stego_hand_names = get_hand_prob(args.stego_feature_path, args.clf_path, args.stego_dir)
adv_hand_probs, adv_hand_names = get_hand_prob(args.adv_feature_path, args.clf_path, args.cover_dir)

cover_cnn_probs, cover_hand_probs, names = align_cnn_hand_data(cover_cnn_probs, cover_hand_probs, cover_cnn_names, cover_hand_names)
stego_cnn_probs, stego_hand_probs, _ = align_cnn_hand_data(stego_cnn_probs, stego_hand_probs, stego_cnn_names, stego_hand_names)
adv_cnn_probs, adv_hand_probs, _ = align_cnn_hand_data(adv_cnn_probs, adv_hand_probs, adv_cnn_names, adv_hand_names)

rough_filter = train_Rough_Filter(
        CNN_prob = np.concatenate((cover_cnn_probs, stego_cnn_probs, adv_cnn_probs), 0), 
        hand_prob = np.concatenate((cover_hand_probs, stego_hand_probs, adv_hand_probs), 0), 
        labels = np.concatenate((np.zeros((cover_cnn_probs.shape[0],1)), np.ones((stego_cnn_probs.shape[0]+adv_cnn_probs.shape[0],1))), 0),
        save_path = args.rough_filter_path,
        kernel = args.kernel, gamma = args.gamma, C = args.C)