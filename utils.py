import torch
import models
import torch.nn as nn
import subprocess

def get_CNN_prob(
    model,
    dataloader):
    probs = np.ndarray(shape = [dataloader.dataset.len, 2])
    names = []
    with torch.no_grad():
        for idx, data in dataloader:
            inputs, names = data[0].cuda(), data[1]
            if isinstance(model, models.KeNet):
                output_batch, _,_ = model(*inputs)
            else:
                output_batch = model(inputs)
            probs[idx*dataloader.batch_size:(idx+1)*dataloader.batch_size,:]\
                = nn.functional.softmax(output_batch)
            names.extend(list(data[1]))
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
    hand_prob = io.loadmat('./tmp/hand_prob.mat')['prob']
    hand_names_tmp = io.loadmat('./tmp/hand_prob.mat')['names']
    hand_names = []
    for i in range(len(hand_names_tmp)):
        hand_names.append(hand_names_tmp[i][0][0])

    return hand_prob, hand_names