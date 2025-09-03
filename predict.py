import torch
import numpy as np
import cv2
import argparse
from utils import make_filename
from model import MyNet
from config import CONFIG


# ===========================
# Argument parser and device configuration
# ===========================

parser = argparse.ArgumentParser(description="Load model path from CLI")
parser.add_argument('--input', type=str, required=True, help='Path to the input image')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
parser.add_argument('--out_dir', type=str, default=CONFIG["out_dir"], help='output directory')
parser.add_argument('--nChannel', type=int, default=CONFIG["nChannel"], help='number of channels')
parser.add_argument('--nConv', type=int, default=CONFIG["nConv"], help='number of convolutional layers')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ===========================   
# Load input image and preprocessing
# ===========================

im = cv2.imread(args.input)
im_trans = np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.])
data = torch.from_numpy(im_trans).to(device)

# ===========================   
# Define model
# ===========================

model = MyNet(input_dim=data.size(1), nChannel=args.nChannel, nConv=args.nConv).to(device)
model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()

# ===========================   
# Inference
# ===========================

output = model(data)[0]
output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
ignore, target = torch.max( output, 1 )
im_target = target.data.cpu().numpy()
label_colours = np.random.randint(255,size=(100,3))
output_rgb = np.array([label_colours[ c % 100 ] for c in im_target])

# ===========================   
# Save output image
# ===========================

path_output = make_filename(args.out_dir, args.input)+'.png'
cv2.imwrite(path_output, output_rgb.reshape(im.shape) )
print(f"output image saved: {path_output}")
