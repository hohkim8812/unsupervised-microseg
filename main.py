import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from skimage import segmentation
from config import CONFIG
from model import MyNet
from utils import *

# ===========================
# Argument parser and device configuration
# ===========================

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Superpixel Refinement')
    parser.add_argument('--input', type=str, default='data/unsupervised_segmentation_input.png', help='input image')
    parser.add_argument('--output', type=str, default='data/unsupervised_segmentation_output.png', help='segmented output image')
    parser.add_argument('--nChannel', type=int, default=CONFIG["nChannel"], help='number of channels')
    parser.add_argument('--maxIter', type=int, default=CONFIG["maxIter"], help='maximum number of iterations')
    parser.add_argument('--minLabels', type=int, default=CONFIG["minLabels"], help ='minimum number of labels')
    parser.add_argument('--lr', type=float, default=CONFIG["lr"], help='learning rate')
    parser.add_argument('--nConv', type=int, default=CONFIG["nConv"], help='number of convolutional layers')
    parser.add_argument('--num_superpixels', type=int, default=CONFIG["num_superpixels"], help='number of superpixels')
    parser.add_argument('--compactness_value', type=int, default=CONFIG["compactness_value"], help='compactness value for SLIC')
    parser.add_argument('--out_dir', type=str, default=CONFIG["out_dir"], help='output directory')
    parser.add_argument('--model_path', type=str, default=CONFIG["model_path"], help='path to save the model')
    return parser.parse_args()

args = parse_args()

# Check the torch version
print('torch version: %s' %torch.__version__)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ===========================   
# Load input image and preprocessing
# ===========================

im = cv2.imread(args.input)
im_trans = np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.])
data = torch.from_numpy(im_trans)
data = data.to(device)

# ===========================
# Superpixel segmentation
# ===========================

# Extract superpixels labels and boundaries
labels = segmentation.slic(im, compactness=args.compactness_value, n_segments=args.num_superpixels)
boundaries = segmentation.mark_boundaries(im, labels)

# Reshape labels into 1D and get unique labels
labels = labels.reshape(im.shape[0]*im.shape[1])
u_labels = np.unique(labels)

# Create a list of indices for each label
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# ===========================
# Define Model
# ===========================

# Initialize model
model = MyNet(input_dim=data.size(1), nChannel=args.nChannel, nConv=args.nConv).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# Randomly generate colors for visualizing the labels
label_colours = np.random.randint(255,size=(100,3))

# ===========================
# Training
# ===========================

# Training
for iter in range(args.maxIter):
    # forward
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    _, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    # Superpixel refinement
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]

    target = torch.from_numpy( im_target )
    target = target.to(device)
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    #print (iter, '/', args.maxIter, ':', nLabels, loss.data[0])
    print (iter, '/', args.maxIter, ':', nLabels, loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# ===========================
# Save outputs
# ===========================

label_colours = np.random.randint(255,size=(100,3))
with torch.no_grad():
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    _, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    output_rgb = np.array([label_colours[ c % 100 ] for c in im_target])

path_output = make_filename(args.out_dir, args.input)+'.png'
cv2.imwrite(path_output, output_rgb.reshape(im.shape) )
print(f"output image saved: {path_output}")

path_model = path_output.replace("output", "saved_model").replace(".png", ".pth")
torch.save(model.state_dict(), path_model)
print(f"model weights saved: {path_model}")
