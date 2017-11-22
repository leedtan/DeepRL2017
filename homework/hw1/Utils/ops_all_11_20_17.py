
import math
import numpy as np 
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
import copy

from tensorflow.python.framework import ops

eps = 1e-8

def lrelu(x):
        return tf.maximum(x, 1e-2*x)

def relu(x):
        return tf.maximum(x, 0)
    
def noise(shape, noise_level):
    return tf.random_normal(shape, mean=0, stddev = noise_level)

def split_trn_tst(x, y, frac_trn = .8):
    num_total = x.shape[0]
    trn = np.random.choice(num_total, int(round(num_total*frac_trn)), replace=False)
    x_trn = x[trn]
    y_trn = y[trn]
    tst = [i for i in range(num_total) if i not in trn]
    x_tst = x[tst]
    y_tst = y[tst]
    return x_trn, y_trn, x_tst, y_tst


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, k = 5, d = 2,
           k_h=None, k_w=None, d_h=None, d_w=None, stddev=0.02,
           name="conv2d", padding='SAME'):
    if k_h ==None:
        k_h = k
    if k_w ==None:
        k_w = k
    if d_h ==None:
        d_h = d
    if d_w ==None:
        d_w = d
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
    
    
    
    
def make_wb_conv2d(input_, output_dim, k_h, k_w, stddev=0.02, name='wb'):
    w = tf.get_variable(name + 'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name + 'b', [output_dim], initializer=tf.constant_initializer(0.0))
    return (w,b)

def make_wb_deconv2d(input_, output_dim, k_h, k_w, stddev=0.02, name='wb'):
    w = tf.get_variable(name + 'w', [k_h, k_w, output_dim, input_.get_shape()[-1]],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name + 'b', [output_dim], initializer=tf.constant_initializer(0.0))
    return (w,b)

def conv2d_input(input_, output_dim, wb,d_h=2, d_w=2, padding='SAME'):
    w = wb[0]
    b = wb[1]

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
    
    sh1, sh2, sh3 = conv.get_shape()[1].value, conv.get_shape()[2].value, conv.get_shape()[3].value
    
    b_exp = tf.reshape(b, [1, 1, 1, sh3])
    b_exp = tf.tile(b_exp, [1, sh1, sh2, 1])
    conv = conv + b_exp
    #conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

    return conv

def deconv2d_input(input_, output_dim, wb,d_h=2, d_w=2):
    w = wb[0]
    b = wb[1]
    s = input_.get_shape()
    sh0, sh1, sh2, sh3 = s[0].value, s[1].value, s[2].value, s[3].value
    
    conv = tf.nn.conv2d_transpose(input_, w, output_shape=[sh0, sh1*d_h, sh2*d_w, output_dim],
                                strides=[1, d_h, d_w, 1], padding='SAME')
    
    sh1, sh2, sh3 = conv.get_shape()[1].value, conv.get_shape()[2].value, conv.get_shape()[3].value
    
    
    b_exp = tf.reshape(b, [1, 1, 1, sh3])
    b_exp = tf.tile(b_exp, [1, sh1, sh2, 1])
    conv = conv + b_exp

    return conv

def conv2d_bn_relu_trn_tst(input_trn, output_dim, k, d, stddev, name='conv2d', input_tst = None, padding='SAME'):
    bn = batch_norm(name=name)
    wb = make_wb_conv2d(input_trn, output_dim, k, k, stddev, name = name + 'wb')
    trn = lrelu(bn(conv2d_input(input_trn, output_dim, wb, d, d, padding=padding)))
    if input_tst == None:
        input_tst = input_trn
    
    tst = lrelu(bn(conv2d_input(input_tst, output_dim, wb, d, d, padding=padding), train=False))
    return trn, tst

def deconv2d_bn_relu_trn_tst(input_trn, output_dim, k, d, stddev, name='conv2d', input_tst = None):
    bn = batch_norm(name=name)
    wb = make_wb_deconv2d(input_trn, output_dim, k, k, stddev, name = name + 'wb')
    trn = lrelu(bn(deconv2d_input(input_trn, output_dim, wb, d, d)))
    if input_tst == None:
        input_tst = input_trn
    
    tst = lrelu(bn(deconv2d_input(input_tst, output_dim, wb, d, d), train=False))
    return trn, tst

def deconv2d_trn_tst(input_trn, output_dim, k, d, stddev, name='conv2d', input_tst = None):
    wb = make_wb_deconv2d(input_trn, output_dim, k, k, stddev, name = name + 'wb')
    trn = deconv2d_input(input_trn, output_dim, wb, d, d)
    if input_tst == None:
        input_tst = input_trn
    
    tst = deconv2d_input(input_tst, output_dim, wb, d, d)
    return trn, tst
    

def combine_last_two_dense(arr, stride):
    last_arr = arr[-1][1]
    prev_arr = arr[-2][1]
    arr[-2] = (stride, tf.concat((prev_arr, last_arr),-1))
    arr.pop()
    return arr

def add_up_list(lst):
    tns = tf.stack(lst, -1)
    return tf.reduce_sum(tns, -1)


def dense_bn_relu_trn_tst(input_trn_list, output_dim, k, d, stddev = .001, name='conv2d', input_tst_list = None):
    new_active_list_trn = []
    new_active_list_tst = []
    output_contribution_trn = []
    output_contribution_tst = []
    act_stride = None
    if input_tst_list == None:
        input_tst_list = input_trn_list
    init = 0
    for idx, input_trn in enumerate(input_trn_list):
        act_stride = input_trn[0] * d
        if act_stride > 7 and init == 0:
            init = 1
            continue
        if act_stride > 15:
            continue
        act_vec = input_trn[1]
        act_vec_tst = input_tst_list[idx][1]
        act_kernel = act_stride + np.ceil((k + 1) / 2)
        print(idx)
        wb = make_wb_conv2d(act_vec, output_dim, act_kernel, act_kernel, stddev, name = name + str(idx) + 'wb')
        output_contribution_trn.append(conv2d_input(act_vec, output_dim, wb, act_stride, act_stride))
        output_contribution_tst.append(conv2d_input(act_vec_tst, output_dim, wb, act_stride, act_stride))
        new_active_list_trn.append((act_stride, act_vec))
        new_active_list_tst.append((act_stride, act_vec_tst))
    
    raw_output_trn = add_up_list(output_contribution_trn)
    raw_output_tst = add_up_list(output_contribution_tst)
    
    # Make network residual from previous layer. 
    # If new layer is smaller than previous layer, such as if there are multiple dense connections without stride,
    # Make it residual from the ending section of the last layer.
    if d == 1:
        if input_trn_list[-1][-1].get_shape().as_list()[-1] == output_dim:
            raw_output_trn += input_trn_list[-1][-1]
            raw_output_tst += input_tst_list[-1][-1]
        elif input_trn_list[-1][-1].get_shape().as_list()[-1] > output_dim:
            raw_output_trn += input_trn_list[-1][-1][:,:,:,-output_dim:]
            raw_output_tst += input_tst_list[-1][-1][:,:,:,-output_dim:]
    
    bn = batch_norm(name=name)
    trn = lrelu(bn(raw_output_trn))
    
    tst = lrelu(bn(raw_output_tst, train=False))
    new_active_list_trn.append((1, trn))
    new_active_list_tst.append((1, tst))
    if act_stride == 1:
        new_active_list_trn = combine_last_two_dense(new_active_list_trn, 1)
        new_active_list_tst = combine_last_two_dense(new_active_list_tst, 1)
    return new_active_list_trn, new_active_list_tst, trn, tst
    

def dense_bn_relu_trn_tst_broken(input_trn_list, output_dim, k, d, stddev = .001, name='conv2d', input_tst_list = None):
    new_active_list_trn = []
    new_active_list_tst = []
    wb_list = []
    old_active_stride = None
    act_stride = None
    if input_tst_list == None:
        input_tst_list = input_trn_list
    for idx, input_trn in enumerate(input_trn_list):
        old_active_stride = act_stride
        act_stride = input_trn[0] * d
        act_vec = input_trn[1]
        act_vec_tst = input_tst_list[idx][1]
        act_kernel = act_stride + (k + 1) / 2
        print(idx)
        wb = make_wb_conv2d(act_vec, output_dim, act_kernel, act_kernel, stddev, name = name + str(idx) + 'wb')
        wb_list.append(wb)
        new_active_list_trn.append((act_stride, act_vec))
        new_active_list_tst.append((act_stride, act_vec_tst))
    if old_active_stride == new_active_list_trn[-1][0]:
        new_active_list_trn = combine_last_two_dense(new_active_list_trn, act_stride)
        new_active_list_tst = combine_last_two_dense(new_active_list_tst, act_stride)
    output_contribution_trn = []
    output_contribution_tst = []
    for idx, input_vecs in enumerate(zip(new_active_list_trn, new_active_list_tst)):
        active_trn, active_tst = input_vecs[0], input_vecs[1]
        tensor_trn = active_trn[1]
        tensor_tst = active_tst[1]
        stride = active_trn[0]
        output_contribution_trn.append(conv2d_input(tensor_trn, output_dim, wb_list[idx], stride, stride))
        output_contribution_tst.append(conv2d_input(tensor_tst, output_dim, wb_list[idx], stride, stride))
    
    raw_output_trn = add_up_list(output_contribution_trn)
    raw_output_tst = add_up_list(output_contribution_tst)
        
    
    bn = batch_norm(name=name)
    trn = lrelu(bn(raw_output_trn))
    
    tst = lrelu(bn(raw_output_tst, train=False))
    input_trn_list.append((1, trn))
    input_tst_list.append((1, tst))
    return input_trn_list, input_tst_list, trn, tst
    
    
def cascade_bn_relu_trn_tst(input_trn, out_size, name, input_tst=None):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), name=name+'net_w_') * 1e-2
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+'net_bias_') * 1e-2
    act_trn = tf.concat((input_trn, lrelu(bn(tf.matmul(input_trn, w) + b))), 1)
    if input_tst == None:
        input_tst = input_trn
    
    act_tst = tf.concat((input_tst, lrelu(bn(tf.matmul(input_tst, w) + b, train=False))), 1)
    return act_trn, act_tst

    
def res_bn_relu_trn_tst(input_trn, hidden_size, out_size, name, input_tst=None):
    if input_tst == None:
        input_tst = input_trn
    
    prev_layer_size = input_trn.get_shape()[1].value
    hidden_trn, hidden_tst = FC_bn_relu_trn_tst(input_trn, hidden_size, name = name +'_hidden', 
                                                input_tst=input_tst)
    #FC_trn_tst(trn, tst, output_size, stddev=.001, name='output_FC_layer', bias_start = 0.0):
    out_trn, out_tst = FC_trn_tst(hidden_trn, hidden_tst, out_size, name = name +'_output')
    
    if out_size <= prev_layer_size:
        out_trn = out_trn + input_trn[:,-out_size:]
        out_tst = out_tst + input_tst[:,-out_size:]
    act_trn = lrelu(out_trn)
    act_tst = lrelu(out_tst)
    
    return act_trn, act_tst
    
def cascade_res_bn_relu_trn_tst(input_trn, out_size, name, input_tst=None):
    if input_tst == None:
        input_tst = input_trn
    
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), name=name+'net_w_') * 1e-2
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+'net_bias_') * 1e-2
    
    mul = tf.matmul(input_trn, w) + b
    if out_size <= prev_layer_size:
        res = mul + input_trn[:,-out_size:]
    else:
        res = mul
    act_trn = tf.concat((input_trn, lrelu(bn(res))), 1)
    
    mul = tf.matmul(input_tst, w) + b
    if out_size <= prev_layer_size:
        res = mul + input_tst[:,-out_size:]
    else:
        res = mul
    act_tst = tf.concat((input_tst, lrelu(bn(res, train=False))), 1)
    return act_trn, act_tst


def FC_bn(input_trn, out_size, name):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), name=name+'net_w_') * 1e-2
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+'net_bias_') * 1e-2
    act_trn = bn(tf.matmul(input_trn, w) + b)
    return act_trn

    
def FC_bn_relu_trn_tst(input_trn, out_size, name, input_tst=None):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), 
                    name=name+'net_w_') * 1e-3
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+
                    'net_bias_') * 1e-3
    act_trn = lrelu(bn(tf.matmul(input_trn, w) + b))
    if input_tst == None:
        input_tst = input_trn
    
    act_tst = lrelu(bn(tf.matmul(input_tst, w) + b, train=False))
    return act_trn, act_tst


    
def FC_bn_softplus_trn_tst(input_trn, out_size, name, input_tst=None):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), name=name+'net_w_') * 1e-2
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+'net_bias_') * 1e-2
    act_trn = tf.nn.softplus(bn(tf.matmul(input_trn, w) + b))
    if input_tst == None:
        input_tst = input_trn
    
    act_tst = tf.nn.softplus(bn(tf.matmul(input_tst, w) + b, train=False))
    return act_trn, act_tst

def res_bn_relu_bottleneck_trn_tst(input_trn, hidden_size, out_size, name, input_tst=None):
    if input_tst == None:
        input_tst = input_trn
    
    prev_layer_size = input_trn.get_shape()[1].value
    
    if out_size <= prev_layer_size:
        hidden_trn, hidden_tst = FC_bn_relu_trn_tst(input_trn, hidden_size, name = name +'_hidden', 
                                                    input_tst=input_tst)
        hidden_trn, hidden_tst = FC_bn_relu_trn_tst(hidden_trn, hidden_size, name = name +'_bottleneck', 
                                                    input_tst=hidden_tst)
        #FC_trn_tst(trn, tst, output_size, stddev=.001, name='output_FC_layer', bias_start = 0.0):
        out_trn, out_tst = FC_bn_trn_tst(hidden_trn, out_size, name = name +'_output', input_tst = hidden_tst)
        out_trn = out_trn + input_trn[:,-out_size:]
        out_tst = out_tst + input_tst[:,-out_size:]
        act_trn = lrelu(out_trn)
        act_tst = lrelu(out_tst)
    else:
        act_trn, act_tst = FC_bn_relu_trn_tst(input_trn, out_size, name = name +'_hidden', 
                                                    input_tst=input_tst)

    
    return act_trn, act_tst

    
def FC_softplus_trn_tst(input_trn, out_size, name, input_tst=None):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), name=name+'net_w_') * 1e-2
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+'net_bias_') * 1e-2
    act_trn = tf.nn.softplus(bn(tf.matmul(input_trn, w) + b))
    if input_tst == None:
        input_tst = input_trn
    
    act_tst = tf.nn.softplus(bn(tf.matmul(input_tst, w) + b, train=False))
    return act_trn, act_tst



    

def conv_regional(input_, output_dim, 
           k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
           name="conv2d"):
    
    in_len = input_.get_shape()[1].value
    inbound1, inbound2 = int(in_len//4), int(in_len//2)
    
    conv_all = conv2d(input_, output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + 'all')
    
    conv_0 = conv2d(input_[:,:inbound1, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '0')
    
    conv_1 = conv2d(input_[:,inbound1:inbound2, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '1')
    
    conv_2 = conv2d(input_[:,inbound2:, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '2')
    conv_regions = tf.concat((conv_0, conv_1, conv_2), 1)
    conv = tf.concat((conv_all, conv_regions), 3)
    
    return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv2d_audio(input_, output_shape,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d", padding='SAME'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w_1 = tf.get_variable('w', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv_1 = tf.nn.conv2d_transpose(input_, w_1, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1], padding=padding)

        w_2 = tf.get_variable('w2', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv_2 = tf.nn.conv2d_transpose(input_, w_2, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1], padding=padding)
        '''
        testing
        deconv_2 = tf.concat([tf.zeros((3,4,1,4)),tf.ones((3,4,1,4))],2)
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        deconv_3 = tf.concat([deconv_2[:,:,1:,:], deconv_2[:,:,:1,:]] , 2)
        s.run(deconv_2)
        s.run(deconv_3)
        '''
        deconv_2 = tf.concat([deconv_2[:,:,1:,:], deconv_2[:,:,:1,:]] , 2)
        deconv = deconv_1 + deconv_2
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv
    
def deconv_regional(input_, output_shape,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d", padding='SAME'):
    in_len = input_.get_shape()[1].value
    inbound1, inbound2 = int(in_len//4), int(in_len//2) 
    o1, o2 = int(output_shape[1]//4), int(output_shape[1]//2)
    
    deconv_all = deconv2d_audio(input_, output_shape,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + 'all', padding=padding)
    outs = output_shape
    outs[1] = o1
    deconv_0 = deconv2d_audio(input_[:,:inbound1, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '0', padding=padding)
    
    deconv_1 = deconv2d_audio(input_[:,inbound1:inbound2, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '1', padding=padding)
    
    outs[1] = o2
    deconv_2 = deconv2d_audio(input_[:,inbound2:, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '2', padding=padding)
    deconv_regions = tf.concat((deconv_0, deconv_1, deconv_2), 1)
    deconv = tf.concat((deconv_all, deconv_regions), 3)
    
    return deconv
    
def deconv2d_audio_local(input_, out_filters,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d"):
    '''
    deconv_11 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '11')
    deconv_12 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '12')
    deconv_21 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '21')
    deconv_22 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '22')
    deconv_1 = deconv_11 + deconv_12
    deconv_2 = deconv_21 + deconv_22
    '''
    #deconv_1 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '1')
    #deconv_2 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '2')
    input1 = input_[:,:,0,:]
    input2 = input_[:,:,1,:]
    
    '''
    keras.layers.local.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid',
    data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, bias_constraint=None)
    '''
    deconv_11 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input1)
    deconv_12 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input1)
    deconv_21 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input2)
    deconv_22 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input2)
    deconv_1 = deconv_11 + deconv_21
    deconv_2 = deconv_12 + deconv_22
    deconv = tf.concat((tf.expand_dims(deconv_1, 2), tf.expand_dims(deconv_2, 2)), 2)

    return deconv
def conv_audio_local(input_, out_filters,
             k,s, stddev=0.02,
             name="deconv2d"):
    input1 = input_[:,:,0,:]
    
    deconv = Local(filters = out_filters, kernel_size = k, strides = s)(input1)

    deconv = tf.expand_dims(deconv, 2)
    return deconv

def conv_local(input_, output_shape, k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d"):
    A_len = output_shape[1]
    F_out = output_shape[-1]
    F_in = input_.get_shape()[-1]
    bs = input_.get_shape()[0].value
    with tf.variable_scope(name):
        act_out = []
        for out_idx in range(A_len):
            act_out_slice = tf.zeros((bs,F_out))
            for in_idx in range(k_h):
                w = tf.get_variable('w_i2n_' + str(out_idx) + '_in_' + str(in_idx), [F_out, F_in],
                            initializer=tf.random_normal_initializer(stddev=stddev))
                w = tf.tile(tf.expand_dims(w, 0), [bs,1,1])
                in_slice = tf.expand_dims(input_[:,out_idx + in_idx,:],-1)
                act = tf.reshape(tf.matmul(w,in_slice), [bs, F_out])
                act_out_slice += act
                
            act_out = act_out + [act_out_slice]
        
        act_out = tf.stack(act_out, 1)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(act_out, biases), act_out.get_shape())
    
        return deconv

def deconvMany(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    deconv_out = [None]*4
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        for idx, val in enumerate([1,3,5,7]):
            w = tf.get_variable('w' + str(idx), [val, val, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[int(o) for o in output_shape],
                                    strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases' + str(idx), [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv_out[idx] = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return None
        else:
            return tf.concat(deconv_out,axis=3)

def noised(x,std=.2):
    return x + tf.random_normal(x.get_shape(), mean=0,stddev=std)

def noised_gamma(x, std=.2, alpha=.5,beta=1):
    return x + tf.minimum(tf.random_gamma([1], alpha=alpha, beta = beta)[0],2) * \
        tf.random_normal(x.get_shape(), mean=0,stddev=std)

def parametric_relu(_x, name):
    alphas = tf.Variable(tf.ones(_x.get_shape()[-1])*0.0001, name = name)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    
    return pos + neg

def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def FC_bn_trn_tst(input_trn, out_size, name, input_tst=None):
    prev_layer_size = input_trn.get_shape()[1].value
    bn = batch_norm(name=name+'bn')
    w = tf.Variable(tf.random_uniform([prev_layer_size, out_size],minval = -1., maxval = 1.), 
                    name=name+'net_w_') * 1e-3
    b = tf.Variable(tf.random_uniform([out_size],minval = -1., maxval = 1.), name=name+
                    'net_bias_') * 1e-3
    act_trn = bn(tf.matmul(input_trn, w) + b)
    if input_tst == None:
        input_tst = input_trn
    
    act_tst = bn(tf.matmul(input_tst, w) + b, train=False)
    return act_trn, act_tst


def FC_trn_tst(trn, tst, output_size, stddev=.001, name='output_FC_layer', bias_start = 0.0):
    shape = trn.get_shape().as_list()
    matrix = tf.get_variable("Matrix" + name, [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias" + name, [output_size],
        initializer=tf.constant_initializer(bias_start))
    return tf.matmul(trn, matrix) + bias, tf.matmul(tst, matrix) + bias

def FC_sigmoid_trn_tst(trn, tst, output_size, stddev=.001, name='output_FC_layer', bias_start = 0.0):
    shape = trn.get_shape().as_list()
    matrix = tf.get_variable("Matrix" + name, [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias" + name, [output_size],
        initializer=tf.constant_initializer(bias_start))
    return tf.sigmoid(tf.matmul(trn, matrix) + bias), tf.sigmoid(tf.matmul(tst, matrix) + bias)

def plot_matrix(writer, matrix, tag, log_dir, iter):
    png_file = log_dir + '/temp.png'
    plt.plot(matrix)
    plt.savefig(png_file)
    plt.close()
    with open(png_file, 'rb') as f:
        imgbuf = f.read()
    img = Image.open(png_file)
    summary = tf.Summary.Image(
            height=img.height,
            width=img.width,
            colorspace=3,
            encoded_image_string=imgbuf
            )
    summary = tf.Summary.Value(tag='%s/' % (tag), image=summary)
    writer.add_summary(tf.Summary(value=[summary]), iter)


def transform_images(self, w_img, bs):
    shape = self.img_batch_resized.get_shape().as_list()
    shape[0] = bs
    self.img_batch_resized.set_shape(shape)
    self.rotate_angle = tf.random_uniform([1], minval=-.1,maxval=.1)
    self.img_rotate = tf.stack([tf.contrib.image.rotate(self.img_batch_resized[idx, :, :, :], self.rotate_angle)
                                  for idx, self in enumerate([self]*bs)], 0)
    self.img_noised = noised(self.img_batch_resized, std=5)
    self.img_batch_cropped = tf.stack([tf.random_crop(self_w[0].img_noised[idx, :, :, :], 
                                            [self_w[1]-10, self_w[1]-10, 3]) for idx, self_w in
                                            enumerate([(self, w_img)]*bs)], 0)
    self.img_batch_cropped_resized = tf.image.resize_images(self.img_batch_cropped, [w_img, w_img])
    
    self.img_flipped = tf.stack([tf.image.random_flip_left_right(self.img_batch_cropped_resized[idx, :, :, :])
                                  for idx, self in enumerate([self]*bs)], 0)
    self.img_saturated = tf.stack([tf.image.random_saturation(self.img_flipped[idx, :, :, :], .7, 1.4)
                                  for idx, self in enumerate([self]*bs)], 0)
    self.img_hue = tf.stack([tf.image.random_hue(self.img_saturated[idx, :, :, :], .2)
                                  for idx, self in enumerate([self]*bs)], 0)
    self.img_bright = tf.stack([tf.image.random_brightness(self.img_hue[idx, :, :, :], .3)
                                  for idx, self in enumerate([self]*bs)], 0)
    self.img_contrast = tf.stack([tf.image.random_contrast(self.img_hue[idx, :, :, :], .7, 1.4)
                                  for idx, self in enumerate([self]*bs)], 0)
    return self.img_contrast




