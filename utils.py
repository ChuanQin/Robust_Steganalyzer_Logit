import torch
import models
import torch.nn as nn
import subprocess
import numpy as np
from scipy import io

def preprocess_data(images):
    # images of shape: NxCxHxW
    if images.dim() == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
    h, w = images.shape[-2:]
    ch, cw, h0, w0 = h, w, 0, 0
    cw = cw & ~1
    inputs = [
        images[..., h0:h0 + ch, w0:w0 + cw // 2],
        images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
    ]
    # if args.cuda:
    inputs = [x.cuda() for x in inputs]
    return inputs

def align_cnn_hand_data(
    cnn_probs,
    hand_probs,
    cnn_names,
    hand_names):
    cnn_names = np.asarray(cnn_names)
    hand_names = np.asarray(hand_names)
    sort_idx = np.argsort(cnn_names)
    cnn_names = cnn_names[sort_idx].copy()
    cnn_probs = cnn_probs[sort_idx].copy()
    hand_probs = np.squeeze(hand_probs[np.argsort(hand_names)], axis = 1)
    hand_names = hand_names[np.argsort(hand_names)].copy()
    hand_names_list = hand_names.tolist()
    inter_ind = [hand_names_list.index(i) for i in cnn_names]
    hand_probs = hand_probs[inter_ind]
    names = np.asarray(list(set(cnn_names)&set(hand_names)))
    names = np.sort(names)
    return cnn_probs, hand_probs, names

def get_CNN_prob(
    model,
    dataloader):
    probs = np.ndarray(shape = [dataloader.dataset.len, 2])
    names = []
    acc = 0.
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            inputs = preprocess_data(data[0]*255.)
            if isinstance(model, models.KeNet) or isinstance(model.module, models.KeNet):
                output_batch, _,_ = model(*inputs)
            else:
                output_batch = model(inputs)
            probs[idx*dataloader.batch_size:(idx+1)*dataloader.batch_size,:]\
                = nn.functional.softmax(output_batch).cpu().numpy()
            acc += models.accuracy(output_batch, data[2].cuda()).item()
            names.extend(list(data[1]))
    print('Accuracy on this dataset: {:.4f}'.format(acc/len(dataloader)))
    return probs, names

def get_hand_prob(
    feature_path, 
    clf_path, 
    ref_tst_dir):
    matlab = ['matlab']
    options = ['-nodisplay', '-nosplash', '-nodesktop', '-r']
    command = ['"cd ./matlab; get_hand_prob_RS(\'{}\',\'{}\',\'{}\'); exit;"'.\
                    format(feature_path, clf_path, ref_tst_dir)]
    subp = subprocess.Popen(' '.join(matlab + options + command), stdout = subprocess.DEVNULL, shell = True)
    subp.wait()
    hand_prob = io.loadmat('./matlab/hand_probs/hand_prob.mat')['prob']
    hand_names_tmp = io.loadmat('./matlab/hand_probs/hand_prob.mat')['names']
    hand_names = []
    for i in range(len(hand_names_tmp)):
        hand_names.append(hand_names_tmp[i][0][0])

    return np.squeeze(hand_prob), hand_names