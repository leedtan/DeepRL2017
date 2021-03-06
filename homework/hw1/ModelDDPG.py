#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""


import tensorflow as tf
import numpy as np
import tf_util
from Utils import ops
import time

import matplotlib
import matplotlib.pyplot as plt

def drop(x):
    return tf.nn.dropout(x, .5)

class Q():
    def __init__(self, size_input, act_obs_vec):
        self.act_obs = act_obs_vec#tf.placeholder(tf.float32, shape=(None, size_input))
        self.true_reward = tf.placeholder(tf.float32, shape=(None))
        self.reward_enabled = tf.placeholder(tf.float32, shape=(None))
        hidden = self.act_obs
        prev_layer_size = size_input
        for idx in range(3):
            hidden, _ = ops.cascade_bn_relu_trn_tst(
                    hidden, prev_layer_size, size_input, name='Qlayer' + str(idx))
            prev_layer_size += size_input
        w = tf.Variable(tf.random_uniform([prev_layer_size, 1],minval = -1., maxval = 1.), name='q_output_w') * 1e-3
        b = tf.Variable(tf.random_uniform([1],minval = -1., maxval = 1.), name='q_output_bias') * 1e-3
        self.yhat = tf.nn.softplus(tf.reshape(tf.matmul(hidden, w) + b, [-1]))
        self.l2_loss = tf.reduce_mean(tf.square(self.yhat - self.true_reward) * self.reward_enabled)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.l2_loss)

class Model():
    
    def __init__(self,  size_obs, size_act, net_struct = [100, 100, 100, 100], name='dbg'):
        self.tensorboardpath = 'tensorboards/' + name
        self.train_writer = tf.summary.FileWriter(self.tensorboardpath)
        self.ModelPath = 'Models/Imitation' + name
        
        self.mse_train = []
        self.mse_val = []
        self.last_epoch = 0
        size_inpt = 200
        self.obs = tf.placeholder(tf.float32, shape=(None, size_obs))
        self.ret = tf.placeholder(tf.float32, shape=(None))
        act_trn = self.obs
        act_tst = self.obs
        prev_layer_size = size_obs
        #Hidden layers
        self.l2_reg = 1e-8
        self.Q_lr = tf.placeholder(tf.float32, shape=(None))
        self.lr = tf.placeholder(tf.float32, shape=(None))
        if 1:
            for idx, l in enumerate(net_struct):
                act_trn, act_tst = ops.cascade_bn_relu_trn_tst(
                        act_trn, prev_layer_size, l, name='layer' + str(idx), input_tst = act_tst)
                prev_layer_size += l
                
            w = tf.Variable(tf.random_uniform([prev_layer_size, size_act],minval = -1., maxval = 1.), name='net_output_w') * 1e-3
            b = tf.Variable(tf.random_uniform([size_act],minval = -1., maxval = 1.), name='net_output_bias') * 1e-3
        else:
            for idx, l in enumerate(net_struct):
                act_trn = ops.linear(act_trn, l, 'layer' + str(idx))
            w = tf.Variable(tf.random_uniform([l, size_act],minval = -1., maxval = 1.), name='net_output_w') * 1e-2
            b = tf.Variable(tf.random_uniform([size_act],minval = -1., maxval = 1.), name='net_output_bias') * 1e-2
        self.yhat = tf.reshape(tf.matmul(act_trn, w) + b, [-1, size_act])
        self.yhat_tst = tf.reshape(tf.matmul(act_tst, w) + b, [-1, size_act])
        
        self.obs_act = tf.concat((self.obs, self.yhat),1)
        self.Q = Q(size_obs + size_act, tf.stop_gradient(self.obs_act))
                
        self.act = tf.placeholder(tf.float32, shape=(None))
        
        self.l2_loss = tf.reduce_mean(tf.square(self.yhat - self.act))
        self.adv_loss = tf.reduce_mean(tf.square(self.yhat_tst - self.act))
        #-1*tf.gather_nd(output_tst, self.y_raw, axis=1)output_tst[list(np.arange(bs)),self.y_raw]
        
        self.advers = tf.gradients(self.l2_loss, self.obs)
        
        t_vars = tf.trainable_variables()
        net_vars = [var for var in t_vars if 'net_' in var.name]
        self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in net_vars])*self.l2_reg
        
        
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gvs = optimizer.compute_gradients(self.l2_loss + self.reg_loss - self.Q.yhat * self.Q_lr + self.Q.l2_loss)
        self.grad_norm = tf.reduce_mean([tf.reduce_mean(grad) for grad, var in gvs if grad is not None])
        clip_norm = 100
        clip_single = 1
        capped_gvs = [(tf.clip_by_value(grad, -1*clip_single,clip_single), var) for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in capped_gvs if grad is not None]
        self.optimizer = optimizer.apply_gradients(capped_gvs)
        
        #self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.l2_loss)
        
        self.cur_Q_lr = 0
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.Saver = tf.train.Saver()
        
        
    def restore(self):
        try:
            self.Saver.restore(self.session, self.ModelPath)
            print('using pre trained model')
        except:
            print('could not find old model. training from scratch')
            pass
        
        
        
    def train(self, expert_train, expert_val, batch_size = 256, verb = 1, epochs = 100):
        
        trn_obs = expert_train['observations']
        trn_act = expert_train['actions']
        trn_ret = expert_train['returns']
        val_obs = expert_val['observations']
        val_act = expert_val['actions']
        val_ret = expert_val['returns']
        trn_ret_enabled = trn_ret[:,1]
        trn_ret_enabled = (trn_ret_enabled > 0.5).astype(float)
        trn_ret = trn_ret[:,0]
        
        start_time = time.time()
        self.batch_size = batch_size
        n_train = trn_obs.shape[0]
        self.num_batches = np.max((n_train // self.batch_size, 1))
        
        self.best_epoch = -1
        best_val_loss = np.inf
        failed_count = 0
        
        # Restart training from correct epoch if continuing to train
        for i in range(self.last_epoch + 1, self.last_epoch + epochs + 1):
            
            #Exit once optimal generalization reached
            if failed_count > 1:
                self.Saver.save(self.session, self.ModelPath)
                return
                self.Saver.restore(self.session, self.ModelPath)
                return
            
            self.last_epoch = i
            avg_cost = 0
            avg_reg = 0
            avg_yhat = 0
            shuffled = np.arange(n_train)
            np.random.shuffle(shuffled)
            print('learning rate:', 1e-1/np.sqrt(i+1))
            pre_adv = 0
            post_adv = 0
            if i > 0:
                self.cur_Q_lr = .1
            for b_idx in range(int(round(self.num_batches))):
                batch_vals = shuffled[self.batch_size * b_idx:self.batch_size * (b_idx + 1)]
                obs_batch  = trn_obs[batch_vals]
                act_batch  = trn_act[batch_vals]
                ret_batch  = trn_ret[batch_vals]
                ret_enabled_batch  = trn_ret_enabled[batch_vals]
                if b_idx == self.num_batches - 1:
                    obs_batch  = trn_obs[shuffled[self.batch_size * b_idx:],:]
                    act_batch  = trn_act[shuffled[self.batch_size * b_idx:],:]
                    ret_batch  = trn_ret[shuffled[self.batch_size * b_idx:]]
                    ret_enabled_batch  = trn_ret_enabled[shuffled[self.batch_size * b_idx:]]
                    
                
                adv, pre_adv_loss = (self.session.run(
                        [self.advers[0], self.l2_loss],
                                   {self.obs: obs_batch, 
                                    self.act: act_batch,
                                    self.Q_lr: self.cur_Q_lr,
                                    self.Q.true_reward : ret_batch/trn_ret.max(),
                                    self.Q.reward_enabled : ret_enabled_batch
                                    }))
                obs_std = obs_batch.std(0)
                adv = adv/np.linalg.norm(adv)
                obs_batch = obs_batch + obs_std * adv * .005 * adv.shape[1]
                _, loss, yhat, reg, grad_norm = (self.session.run(
                        [self.optimizer, self.l2_loss, self.yhat, self.reg_loss,self.grad_norm],
                                   {self.obs: obs_batch, 
                                    self.act: act_batch,
                                    self.lr: 1e-1 /np.sqrt(i+1),
                                    self.Q_lr: self.cur_Q_lr,
                                    self.Q.true_reward : ret_batch/trn_ret.max(),
                                    self.Q.reward_enabled : ret_enabled_batch
                                    }))
                pre_adv += pre_adv_loss
                post_adv += loss
                avg_cost += loss*yhat.shape[0] / n_train
                avg_reg += reg / n_train
            adv_loss_diff = post_adv/self.num_batches-pre_adv/self.num_batches
            print('pre_adv_loss', pre_adv/self.num_batches, 'post_adv_loss', post_adv/self.num_batches,
                  'diff', adv_loss_diff)
            if verb > 0:
                print ("Epoch: ", i, " avg train loss:", avg_cost,  " Reg Loss:", avg_reg, 'total time:', time.time() - start_time)
                #print ("avg train yhat:", np.mean(yhat), "std train yhat:", np.std(yhat))
                
            self.mse_train += [avg_cost]
            
            yhat_val = self.get_yhat(val_obs)
            val_loss = np.mean(np.square(yhat_val - val_act))
            self.mse_val += [val_loss]
            
            self.train_writer.add_summary(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(tag='val_loss', simple_value=val_loss),
                            tf.Summary.Value(tag='adv_loss_diff', simple_value=adv_loss_diff),
                            tf.Summary.Value(tag='grad_norm', simple_value=grad_norm),
                            tf.Summary.Value(tag='reg', simple_value=reg),
                            tf.Summary.Value(tag='trn_loss', simple_value=avg_cost),
                            ]
                        ),
                    i
                    )
            
            if 1:
                    #Keep track of optimzal performance parameters
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_epoch = i
                        #self.Saver.save(self.session, self.ModelPath)
                        failed_count = 0
                    else:
                        failed_count += 1
            
            
            if verb > 0:
                print ("avg loss validation:", val_loss, 'failed count:', failed_count)
        self.Saver.save(self.session, self.ModelPath)
        #self.best_epoch = i
        #self.Saver.save(self.session, self.ModelPath)
    
    #Currently, can be done in one batch for validation. This function can be batched if needed.
    def get_yhat(self, obs):
        
        yhat = self.session.run(self.yhat_tst, {self.obs: obs})
        return yhat
    
    def draw_learning_curve(self, show_graphs = False):
        
        plt.plot(self.mse_train, label='Training Error')
        plt.plot(self.mse_val, label = 'Validation Error')
        plt.legend()
        plt.title("Learning curve")
        
        plt.savefig('output_images/Learning_curve.png')
        if show_graphs:
            plt.show()
        plt.close()
