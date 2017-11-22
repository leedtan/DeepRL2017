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


def dagger(expert_policy_file, envname, render, max_timesteps, num_rollouts, num_dagger):
    
    
    class Logger(object):
        def __init__(self, filename="last_run_output.txt"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
    
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()
            
        def flush(self):
            self.log.flush()
            
    sys.stdout = Logger("logs/" + str(os.path.basename(sys.argv[0])) +
                        str(time.time()) + ".txt")
    
    print(sys.version)
    
    try:
        returns, expert_data = pickle.load( open('datasets/objs_dagger_' + args.envname + '.pkl', "rb" ) )
        print('using old dagger data')
    except:
        try:
            returns, expert_data = pickle.load( open('datasets/objs_expert_' + args.envname + '.pkl', "rb" ) )
            print('using old expert only data')
        except:
            print('making new expert data')
            import run_expert
            run_expert.run_expert(envname = args.envname, render=0, 
                        max_timesteps = 1000, 
                        num_rollouts = 10)
            returns, expert_data = pickle.load( open('datasets/objs_expert_' + args.envname + '.pkl', "rb" ) )
    obs, act = expert_data['observations'], expert_data['actions']
    
    expert_train, expert_val = ops.split_all_train_val(obs, act)
    
    size_obs = obs.shape[1]
    size_act = act.shape[1]
    
    model = Model(size_obs, size_act, name=envname)
    model.ModelPath = model.ModelPath
    model.restore()
    model.last_epoch=3
    model.train(expert_train, expert_val, verb = 1, epochs = 1000)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')
    total_i = 0
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        observations = obs.tolist()
        actions = act.tolist()
        thresh = 0
        for dag in range(num_dagger):
            print('threshold:', thresh)
            returns = []
            for i in range(num_rollouts):#(num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    total_i += 1
                    expert_action = policy_fn(obs[None,:])
                    predicted_actions = model.get_yhat(np.array([obs]))
                    observations.append(obs)
                    actions.append(expert_action.flatten())
                    
                    if 0:
                        model.train_writer.add_summary(
                            tf.Summary(
                                value=[
                                    tf.Summary.Value(tag='expert_mean', simple_value=expert_action.mean()),
                                    tf.Summary.Value(tag='expert_std', simple_value=expert_action.std()),
                                    tf.Summary.Value(tag='predict_mean', simple_value=predicted_actions.mean()),
                                    tf.Summary.Value(tag='predicted_std', simple_value=predicted_actions.std()),
                                    ]
                                ),
                            total_i
                            )
                    elif 0:
                        for idx in range(predicted_actions.shape[1]):
                            model.train_writer.add_summary(
                                tf.Summary(
                                    value=[
                                        tf.Summary.Value(tag='idx' + str(idx) + 'exp mean', 
                                                         simple_value=expert_action[0,idx]),
                                        tf.Summary.Value(tag='idx' + str(idx) + 'pred mean', 
                                                         simple_value=predicted_actions[0,idx])
                                        ] ),
                                total_i
                                )
                    thresh = thresh * .999999
                    if np.random.uniform() > thresh:
                        obs, r, done, _ = env.step(predicted_actions)
                    else:
                        obs, r, done, _ = env.step(expert_action)
                    totalr += r
                    steps += 1
                    if render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                
                model.train_writer.add_summary(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(tag='reward', simple_value=totalr)
                            ]
                        ),
                    total_i
                    )
                returns.append(totalr)
    
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
    
            expert_data = {'observations': np.array(observations),
                           'actions': np.array(actions)}
            
            obs, act = np.array(observations), np.array(actions)
            #np.hstack([np.array([np.array(xii) for xii in xi]) for xi in actions])
            
            expert_train, expert_val = ops.split_all_train_val(obs, act)
            
            print('DAGGER NUMBER:', dag, 'of:', num_dagger)
            
            
            model.train(expert_train, expert_val, verb = 1, epochs = 1000)
            if (dag + 1) % 2 == 0:
                model.draw_learning_curve(show_graphs = False)
                with open('datasets/objs_dagger_' + args.envname + '.pkl', 'wb') as f:
                    pickle.dump([returns, expert_data], f)
                
                
    
    
    '''
    with open('objs.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        returns, expert_data = pickle.load(f)
    '''

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--envname',default='Hopper-v1', type=str)
parser.add_argument('--render', type=bool,default=True)
parser.add_argument("--max_timesteps", default=1000,type=int)
parser.add_argument("--num_dagger", default=200,type=int)
parser.add_argument('--num_rollouts', type=int, default=5,
                    help='Number of expert roll outs')
args = parser.parse_args()
expert_policy_file = 'experts/' + args.envname + '.pkl'
if __name__ == '__main__':
    dagger(expert_policy_file, args.envname, args.render, args.max_timesteps, args.num_rollouts, args.num_dagger)
