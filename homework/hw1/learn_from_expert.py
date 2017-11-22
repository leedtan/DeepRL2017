#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from Utils import ops
from Model import Model
import os
import sys
import time
import argparse

for folder in ['output_images', 'datasets', 'logs', 'Models', 'experts', 'tensorboards']:
    try:
        os.mkdir(folder)
    except:
        continue

class Logger(object):
    def __init__(self, filename="last_run_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.log.flush()
        
sys.stdout = Logger("logs/" + str(os.path.basename(sys.argv[0])) +
                    str(time.time()) + ".txt")

print(sys.version)
parser = argparse.ArgumentParser()

parser.add_argument('--envname',default='Hopper-v1', type=str)
parser.add_argument('--render', type=bool,default=False)
parser.add_argument("--max_timesteps", default=200,type=int)
parser.add_argument('--num_rollouts', type=int, default=10,
                    help='Number of expert roll outs')
args = parser.parse_args()

try:
    returns, expert_data = pickle.load( open('datasets/objs_expert_' + args.envname + '.pkl', "rb" ) )
    print('using old data')
except:
    print('making new expert data')
    import run_expert
    run_expert.run_expert(envname = args.envname, render=args.render, 
                    max_timesteps = args.max_timesteps, 
                    num_rollouts = args.num_rollouts)
    returns, expert_data = pickle.load( open('datasets/objs_expert_' + args.envname + '.pkl', "rb" ) )

obs, act = expert_data['observations'], expert_data['actions']

expert_train, expert_val = ops.split_all_train_val(obs, act)

size_obs = obs.shape[1]
size_act = act.shape[1]

model = Model(size_obs, size_act)



model.train(expert_train, expert_val, verb = 1, epochs = 10)
model.draw_learning_curve()

