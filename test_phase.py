import numpy as np
from sklearn.svm import SVC
import pickle
import torch
import torchvision
import argparse
import torch.nn as nn
import models
import data
from utils import get_CNN_prob, get_hand_prob, align_cnn_hand_data, get_AUC

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cover_dir', type=str, required=True,)
    parser.add_argument('--stego_dir', type=str, required=True,)
    parser.add_argument('--adv_dir', type=str, required=True,)
    parser.add_argument('--cover_feature_path', type=str, required=True,)
    parser.add_argument('--stego_feature_path', type=str, required=True,)
    parser.add_argument('--adv_feature_path', type=str, required=True,)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--base_ec_path', type=str, required=True)
    parser.add_argument('--spc_ec_path', type=str, required=True)
    parser.add_argument('--rough_filter_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

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

def test_Rough_Filter(
    cnn_probs,
    hand_probs,
    names,
    rough_filter_path):
    rough_filter = pickle.load(open(rough_filter_path, 'rb'))
    prob = np.stack((hand_probs, cnn_probs[:,0]), axis = 1)
    suspiciousI = names[np.where((prob[:,0]<0.5)&(prob[:,1]>=0.5))]
    reliableI = names[np.where(prob[:,1]<0.5)]
    both_cover = np.asarray(list(set(np.where(prob[:,0]>=0.5)[0])&set(np.where(prob[:,1]>=0.5)[0])))
    if len(both_cover)!=0:
        suspiciousII = names[both_cover[np.where(rough_filter.predict(prob[both_cover,:])==1)[0]]]
        reliableII = names[both_cover[np.where(rough_filter.predict(prob[both_cover,:])==0)[0]]]
    else:
        suspiciousII = np.asarray([])
        reliableII = np.asarray([])
    reliable = np.asarray(list(set(reliableI).union(set(reliableII))))
    suspicious = np.asarray(list(set(suspiciousI).union(set(suspiciousII))))
    return suspicious, reliable

def get_final_preds(
    pred_probs,
    pred_labels,
    cnn_probs,
    hand_probs,
    spc_probs,
    names,
    rough_filter_path):

    suspicious_names, reliable_names = test_Rough_Filter(
        cnn_probs = cnn_probs, hand_probs = hand_probs, names = names, 
        rough_filter_path = rough_filter_path)

    reliable_idx = np.searchsorted(names, reliable_names)
    for i in reliable_idx:
        pred_probs[i,0] = cnn_probs[i,0].copy()
    suspicious_idx = np.searchsorted(names, suspicious_names)
    for i in suspicious_idx:
        pred_probs[i,0] = spc_probs[i].copy()
    pred_labels[np.where(pred_probs[:,0]<0.5)] = 1
    pred_probs[:,1] = 1 - pred_probs[:,0]

    return pred_labels, pred_probs



args = parse_args()
net = set_model(args)
cover_loader, stego_loader, adv_loader = set_dataloader(args)
pred_labels = np.zeros((cover_loader.dataset.len,))
pred_probs = np.zeros((cover_loader.dataset.len,2))

cover_cnn_probs, cover_cnn_names = get_CNN_prob(net, cover_loader)
stego_cnn_probs, stego_cnn_names = get_CNN_prob(net, stego_loader)
adv_cnn_probs, adv_cnn_names = get_CNN_prob(net, adv_loader)

cover_hand_probs, cover_hand_names = get_hand_prob(args.cover_feature_path, args.base_ec_path, args.cover_dir)
stego_hand_probs, stego_hand_names = get_hand_prob(args.stego_feature_path, args.base_ec_path, args.stego_dir)
adv_hand_probs, adv_hand_names = get_hand_prob(args.adv_feature_path, args.base_ec_path, args.cover_dir)

cover_cnn_probs, cover_hand_probs, names = align_cnn_hand_data(cover_cnn_probs, cover_hand_probs, cover_cnn_names, cover_hand_names)
stego_cnn_probs, stego_hand_probs, _ = align_cnn_hand_data(stego_cnn_probs, stego_hand_probs, stego_cnn_names, stego_hand_names)
adv_cnn_probs, adv_hand_probs, _ = align_cnn_hand_data(adv_cnn_probs, adv_hand_probs, adv_cnn_names, adv_hand_names)

cover_spc_hand_probs, _ = get_hand_prob(args.cover_feature_path, args.spc_ec_path, args.cover_dir)
stego_spc_hand_probs, _ = get_hand_prob(args.stego_feature_path, args.spc_ec_path, args.cover_dir)
adv_spc_hand_probs, _ = get_hand_prob(args.adv_feature_path, args.spc_ec_path, args.cover_dir)

_, cover_spc_hand_probs, spc_names = align_cnn_hand_data(cover_cnn_probs, cover_spc_hand_probs, cover_cnn_names, cover_hand_names)
_, stego_spc_hand_probs, _ = align_cnn_hand_data(stego_cnn_probs, stego_spc_hand_probs, stego_cnn_names, stego_hand_names)
_, adv_spc_hand_probs, _ = align_cnn_hand_data(adv_cnn_probs, adv_spc_hand_probs, adv_cnn_names, adv_hand_names)

cover_pred_labels, cover_pred_probs = get_final_preds(
        pred_probs = np.zeros((cover_loader.dataset.len,2)),
        pred_labels = np.zeros((cover_loader.dataset.len,)),
        cnn_probs = cover_cnn_probs,
        hand_probs = cover_hand_probs,
        spc_probs = cover_spc_hand_probs,
        names = names,
        rough_filter_path = args.rough_filter_path)
print('Accuracy on cover images is: {:.4f}'.\
    format((cover_pred_labels==np.zeros((cover_loader.dataset.len,))).sum()/cover_loader.dataset.len))

stego_pred_labels, stego_pred_probs = get_final_preds(
        pred_probs = np.zeros((cover_loader.dataset.len,2)),
        pred_labels = np.zeros((cover_loader.dataset.len,)),
        cnn_probs = stego_cnn_probs,
        hand_probs = stego_hand_probs,
        spc_probs = stego_spc_hand_probs,
        names = names,
        rough_filter_path = args.rough_filter_path)
print('Accuracy on stego images is: {:.4f}'.\
    format((stego_pred_labels==np.ones((stego_loader.dataset.len,))).sum()/stego_loader.dataset.len))

adv_pred_labels, adv_pred_probs = get_final_preds(
        pred_probs = np.zeros((cover_loader.dataset.len,2)),
        pred_labels = np.zeros((cover_loader.dataset.len,)),
        cnn_probs = adv_cnn_probs,
        hand_probs = adv_hand_probs,
        spc_probs = adv_spc_hand_probs,
        names = names,
        rough_filter_path = args.rough_filter_path)
print('Accuracy on adv images is: {:.4f}'.\
    format((adv_pred_labels==np.ones((adv_loader.dataset.len,))).sum()/adv_loader.dataset.len))

get_AUC(cover_probs = cover_pred_probs,
    stego_probs = stego_pred_probs,
    adv_probs = adv_pred_probs)