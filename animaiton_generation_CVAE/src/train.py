
"""Simple code for training an RNN for motion prediction."""
import math
import os
import random
import sys
import h5py
import logging
import numpy as np
from utils.data_utils import *
from models.motionpredictor import *
import torch
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

# Learning
parser = argparse.ArgumentParser(description='Train RNN for human pose estimation')
parser.add_argument('--learning_rate', dest='learning_rate',
				help='Learning rate',
				default=0.00001, type=float)
parser.add_argument('--learning_rate_decay_factor', 	dest='learning_rate_decay_factor',
				help='Learning rate is multiplied by this much. 1 means no decay.',default=0.9, type=float)
parser.add_argument('--learning_rate_step', dest='learning_rate_step',
				help='Every this many steps, do decay.',
				default=5000, type=int)
parser.add_argument('--batch_size', dest='batch_size',
				help='Batch size to use during training.',
				default=128, type=int)
parser.add_argument('--iterations', dest='iterations',
				help='Iterations to train for.',
				default=1e5, type=int)
parser.add_argument('--kl_factor', dest='kl_factor',
                help='KL loss factor (default: 0.5)',
                default=0.5, type=float)
parser.add_argument('--test_every', dest='test_every',
				help='',default=100, type=int)
parser.add_argument('--size', dest='size',
				help='Size of each model layer.',
				default=512, type=int)
parser.add_argument('--seq_length_in', dest='seq_length_in',
				help='Number of frames to feed into the encoder. 25 fps',
				default=50, type=int)
parser.add_argument('--seq_length_out', dest='seq_length_out',
				help='Number of frames that the decoder has to predict. 25fps',default=10, type=int)
# Directories
parser.add_argument('--data_dir', dest='data_dir', help='Data directory',
				default=os.path.normpath("./Style_data"), type=str)
parser.add_argument('--train_dir', dest='train_dir', help='Training directory',
				default=os.path.normpath("./experiments/"), type=str)
parser.add_argument('--action', dest='action',
				help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking',
				default='all', type=str)
parser.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log-file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

train_dir = os.path.normpath(os.path.join( args.train_dir, args.action,
	'out_{0}'.format(args.seq_length_out),
	'iterations_{0}'.format(args.iterations),
	'size_{0}'.format(args.size),
	'lr_{0}'.format(args.learning_rate)))

# Logging
if args.log_file=='':
    logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)
else:
    logging.basicConfig(filename=args.log_file,format='%(levelname)s: %(message)s',level=args.log_level)

# Detect device
if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    logging.info("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Train dir: "+train_dir)
os.makedirs(train_dir, exist_ok=True)

def main():
    """Train a seq2seq model on human motion"""
    # Set of actions
    actions           = define_actions( args.action )
    number_of_actions = len( actions )

    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, args.seq_length_in, args.seq_length_out, args.data_dir)

    # Create model for training only
    model = MotionPredictor(args.seq_length_in,args.seq_length_out,
        args.size, # hidden layer size
        args.batch_size, 
        args.learning_rate,
        args.learning_rate_decay_factor,
        len( actions ))
    model = model.to(device)

    # This is the training loop
    loss, val_loss = 0.0, 0.0
    current_step   = 0
    kl_losses = []
    rec_losses = []
    all_losses     = []
    all_val_losses = []

    # The optimizer
    #optimiser = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, betas = (0.9, 0.999))

    for _ in range( int(args.iterations) ):
        optimiser.zero_grad()
        # Set a flag to compute gradients
        model.train()
        # === Training step ===

        # Get batch from the training set
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(train_set, 
                                                                          actions,
                                                                          device)

        # Forward pass
        preds, mu, logvar = model(encoder_inputs, decoder_inputs, device)

        # Loss: Mean Squared Errors
        rec_loss = (preds-decoder_outputs)**2
        rec_loss = rec_loss.mean()
        kld_loss = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar))
        kld_loss = torch.mean(torch.sum(kld_loss, axis=1))
        step_loss = args.kl_factor*kld_loss + rec_loss
        val_loss = step_loss.mean()

        # Backpropagation
        step_loss.backward()
        # Gradient descent step
        optimiser.step()

        step_loss = step_loss.cpu().data.numpy()

        if current_step % 100 == 0:
            logging.info("step {0:04d}; step_loss: {1:.4f} ({2:.4f}, {3:.4f})".format(current_step, step_loss, rec_loss.cpu().data.numpy(), kld_loss.cpu().data.numpy()))
        loss += step_loss / args.test_every
        current_step += 1
        # === step decay ===
        if current_step % args.learning_rate_step == 0:
            args.learning_rate = args.learning_rate*args.learning_rate_decay_factor
            optimiser = optim.Adam(model.parameters(),lr=args.learning_rate, betas = (0.9, 0.999))
            print("Decay learning rate. New value at " + str(args.learning_rate))

        # Once in a while, save checkpoint, print statistics.
        if current_step % args.test_every == 0:
            model.eval()
            # === Validation ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(test_set,actions,device)
            preds, mu, logvar = model(encoder_inputs, decoder_inputs, device)

            step_loss = (preds - decoder_outputs)**2
            val_loss  = step_loss.mean()

            print("\n============================\n"
                "Global step:         %d\n"
                "Learning rate:       %.4f\n"
                "Train loss avg:      %.4f\n"
                "--------------------------\n"
                "Val loss:            %.4f\n"
                "============================\n" % (current_step,
                args.learning_rate, loss, val_loss))
            all_val_losses.append([current_step, val_loss.cpu().detach().numpy()])
            all_losses.append([current_step, loss])
            kl_losses.append([current_step, kld_loss.cpu().detach().numpy()])
            rec_losses.append([current_step, rec_loss.cpu().detach().numpy()])
            torch.save(model, train_dir + '/model_' + str(current_step))
            # Reset loss
            loss = 0

    vlosses = np.array(all_val_losses)
    tlosses = np.array(all_losses)
    klosses = np.array(kl_losses) # Kullback Leibler losses
    reclosses = np.array(rec_losses) # reconstruction losses

    # Plot losses
    plt.plot(vlosses[:,0],vlosses[:,1],'b', label='Validation error')
    plt.plot(tlosses[:,0],tlosses[:,1],'r', label='Training error')
    plt.legend()
    plt.savefig(train_dir + 'losses.png')
    plt.close()
    
    plt.plot(klosses[:,0],klosses[:,1],'g', label='KL loss')
    plt.plot(reclosses[:,0],reclosses[:,1],'orange', label='Reconstruction loss')
    plt.legend()
    plt.savefig('{}lambda_{}.png'.format(train_dir, args.kl_factor))
    plt.close()

if __name__ == "__main__":
	main()
