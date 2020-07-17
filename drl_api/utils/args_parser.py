import argparse
import numpy as np
import os


def parse_args(model_choices):
    '''Parse terminal arguments'''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', required=True, type=str, help='full environment name')
    parser.add_argument('--model', required=True, type=str, choices=model_choices)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=0.99, type=np.float32)
    parser.add_argument('--lr', default=1e-3, type=np.float32)
    args = parser.parse_known_args()

    # Fetch default arguments for model

    return args


