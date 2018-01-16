#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Useful functions to perform extreme learning machines based on 
rule 90 elementary cellular automata for 2-dimensional data.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


def data_augmentation_through_elastic_distortions(images, xlen, ylen, nimages, ntransformations, alpha, sigma, random_state=None):
    """
    The same function above vectorized across all the training set and repeated 'ntransformations' times.
    'images' should have shape (nimages,28,28) for the MNIST digits.
    """
    aux = images.reshape(-1,xlen)
    all_images = np.copy(aux)
    for i in range(ntransformations):
        all_images = np.append(all_images, elastic_transform(aux, alpha, sigma, random_state=None), axis=0)
    return all_images.reshape(-1,ylen,xlen)


def get_ruleset(rule):
    aux = format(rule, '008b')
    return np.array(aux.replace('', ' ').split(), dtype=int)


def iterate_ca_cols(state):
    Z = np.copy(state)
    Z[:, 1:-1, :] = np.bitwise_xor(Z[:, :-2, :], Z[:, 2:, :])                                                           #90
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], bitwise_not(np.bitwise_or(Z[:, 2:, :], Z[:, 1:-1, :])))                #2
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(Z[:, :-2, :], Z[:, 2:, :]))                                               #3
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 2:, :]), np.bitwise_xor(Z[:, :-2, :], Z[:, 1:-1, :]))               #6
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(Z[:, 2:, :], np.bitwise_and(Z[:, 1:-1, :], Z[:, :-2, :])))                #7
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])))                #9
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], bitwise_not(Z[:, 2:, :]))                                              #10
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 2:, :]), np.bitwise_or(Z[:, :-2, :], bitwise_not(Z[:, 1:-1, :])))   #11
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 2:, :]), np.bitwise_or(Z[:, 1:-1, :], bitwise_not(Z[:, :-2, :])))   #13
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 2:, :]), np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]))                #14
    #Z[:, 1:-1, :] = bitwise_not(Z[:, 2:, :])                                                                            #15
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 1:-1, :]), np.bitwise_xor(Z[:, :-2, :], Z[:, 2:, :]))               #18
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_xor(Z[:, :-2, :], np.bitwise_and(np.bitwise_and(Z[:, :-2, :], Z[:, 2:, :]), Z[:, 1:-1, :])))) #22
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_xor(Z[:, 1:-1, :], Z[:, 2:, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])) #24
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]), np.bitwise_and(Z[:, 1:-1, :], Z[:, 2:, :]))) #25
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, :-2, :], np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]))) #26
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_and(Z[:, :-2, :], np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])))) #27
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], np.bitwise_and(Z[:, 2:, :], Z[:, :-2, :]))) #28
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]))                             #30
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 2:, :], np.bitwise_and(Z[:, :-2, :], bitwise_not(Z[:, 1:-1, :])))               #32
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], bitwise_not(Z[:, 1:-1, :]))                                            #34
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 1:-1, :]), np.bitwise_or(Z[:, :-2, :], bitwise_not(Z[:, 2:, :])))   #35
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])) #36
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]))) #37
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_or(Z[:, :-2, :], np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]))) #38
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]))                            #40
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])))) #41
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], bitwise_not(np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :])))               #42
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_xor(Z[:, 2:, :], np.bitwise_and(np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])))) #43
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])) #44
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], bitwise_not(Z[:, :-2, :])))                #45
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]))) #46
    #Z[:, 1:-1, :] = np.bitwise_and(bitwise_not(Z[:, 1:-1, :]), np.bitwise_or(Z[:, 2:, :], Z[:, :-2, :]))                #50
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_or(Z[:, 2:, :], Z[:, :-2, :]))                             #54
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 2:, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])) #56
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_or(Z[:, 2:, :], bitwise_not(Z[:, :-2, :])))                #57
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_or(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))) #58
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])                                                          #60
    #Z[:, 1:-1, :] = np.bitwise_or(np.bitwise_and(Z[:, :-2, :], bitwise_not(Z[:, 2:, :])), np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])) #62
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, 2:, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])))) #73
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])) #74
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 1:-1, :], bitwise_not(np.bitwise_and(Z[:, 2:, :], Z[:, :-2, :])))               #76
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(Z[:, :-2, :], np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]))) #78
    #Z[:, 1:-1, :] = np.bitwise_or(np.bitwise_and(Z[:, 1:-1, :], bitwise_not(Z[:, 2:, :])), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])) #94
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_xor(Z[:, :-2, :], np.bitwise_or(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]))))) #104
    #Z[:, 1:-1, :] = bitwise_not(np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])))               #105
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, :-2, :], np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]))                            #106
    #Z[:, 1:-1, :] = np.bitwise_or(np.bitwise_and(Z[:, 1:-1, :], bitwise_not(Z[:, 2:, :])), np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :])) #110
    #Z[:, 1:-1, :] = np.bitwise_or(np.bitwise_and(Z[:, 2:, :], bitwise_not(Z[:, 1:-1, :])), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])) #122
    #Z[:, 1:-1, :] = np.bitwise_or(np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])) #126
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 2:, :], np.bitwise_and(Z[:, 1:-1, :], Z[:, :-2, :]))                            #128
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], bitwise_not(np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :])))               #130
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 1:-1, :], bitwise_not(np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :])))               #132
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))) #134
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 1:-1, :], Z[:, :-2, :])                                                         #136
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], np.bitwise_or(Z[:, 1:-1, :], bitwise_not(Z[:, 2:, :])))                #138
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 1:-1, :], np.bitwise_or(Z[:, :-2, :], bitwise_not(Z[:, 2:, :])))                #140
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_or(np.bitwise_xor(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]))) #142
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 2:, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))) #146
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))                            #150
    #Z[:, 1:-1, :] = np.bitwise_xor(np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]), np.bitwise_or(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]))) #152
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, :-2, :], np.bitwise_and(Z[:, 2:, :], bitwise_not(Z[:, 1:-1, :])))               #154
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_and(Z[:, 2:, :], bitwise_not(Z[:, :-2, :])))               #156
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, 2:, :], Z[:, :-2, :])                                                           #160
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], np.bitwise_or(Z[:, 2:, :], bitwise_not(Z[:, 1:-1, :])))                #162
    #Z[:, 1:-1, :] = np.bitwise_xor(np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]), np.bitwise_or(Z[:, 2:, :], np.bitwise_or(Z[:, 1:-1, :], Z[:, :-2, :]))) #164
    #Z[:, 1:-1, :] = np.bitwise_and(Z[:, :-2, :], np.bitwise_or(Z[:, 2:, :], Z[:, 1:-1, :]))                             #168
    #Z[:, 1:-1, :] = Z[:, :-2, :]                                                                                        #170
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 1:-1, :], np.bitwise_and(Z[:, 2:, :], np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))) #172
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_and(np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]), np.bitwise_xor(Z[:, 1:-1, :], Z[:, :-2, :]))) #178
    #Z[:, 1:-1, :] = np.bitwise_xor(Z[:, 2:, :], np.bitwise_and(Z[:, 1:-1, :], np.bitwise_xor(Z[:, 2:, :], Z[:, :-2, :]))) #184
    #Z[:, 1:-1, :] = np.bitwise_and(np.bitwise_or(Z[:, 2:, :], Z[:, 1:-1, :]), np.bitwise_or(Z[:, :-2, :], np.bitwise_and(Z[:, 2:, :], Z[:, 1:-1, :]))) #232
    return Z


def iterate_ca_rows(state):
    Z = np.copy(state)
    Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, :-2], Z[:, :, 2:])                                                           #90      
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], bitwise_not(np.bitwise_or(Z[:, :, 2:], Z[:, :, 1:-1])))                #2
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(Z[:, :, :-2], Z[:, :, 2:]))                                               #3
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 2:]), np.bitwise_xor(Z[:, :, :-2], Z[:, :, 1:-1]))               #6
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(Z[:, :, 2:], np.bitwise_and(Z[:, :, 1:-1], Z[:, :, :-2])))                #7
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])))                #9
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], bitwise_not(Z[:, :, 2:]))                                              #10
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 2:]), np.bitwise_or(Z[:, :, :-2], bitwise_not(Z[:, :, 1:-1])))   #11 
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 2:]), np.bitwise_or(Z[:, :, 1:-1], bitwise_not(Z[:, :, :-2])))   #13   
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 2:]), np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]))                #14
    #Z[:, :, 1:-1] = bitwise_not(Z[:, :, 2:])                                                                            #15
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, :-2], Z[:, :, 2:]))               #18
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_xor(Z[:, :, :-2], np.bitwise_and(np.bitwise_and(Z[:, :, :-2], Z[:, :, 2:]), Z[:, :, 1:-1])))) #22
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, 2:]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])) #24
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]), np.bitwise_and(Z[:, :, 1:-1], Z[:, :, 2:]))) #25
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, :-2], np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]))) #26
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_and(Z[:, :, :-2], np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])))) #27
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], np.bitwise_and(Z[:, :, 2:], Z[:, :, :-2]))) #28
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]))                             #30
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 2:], np.bitwise_and(Z[:, :, :-2], bitwise_not(Z[:, :, 1:-1])))               #32
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], bitwise_not(Z[:, :, 1:-1]))                                            #34
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 1:-1]), np.bitwise_or(Z[:, :, :-2], bitwise_not(Z[:, :, 2:])))   #35
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])) #36
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]))) #37
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_or(Z[:, :, :-2], np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]))) #38
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]))                            #40
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])))) #41
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], bitwise_not(np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1])))               #42
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_xor(Z[:, :, 2:], np.bitwise_and(np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])))) #43
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])) #44
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], bitwise_not(Z[:, :, :-2])))                #45
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]))) #46
    #Z[:, :, 1:-1] = np.bitwise_and(bitwise_not(Z[:, :, 1:-1]), np.bitwise_or(Z[:, :, 2:], Z[:, :, :-2]))                #50
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_or(Z[:, :, 2:], Z[:, :, :-2]))                             #54
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 2:], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])) #56
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_or(Z[:, :, 2:], bitwise_not(Z[:, :, :-2])))                #57
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_or(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))) #58
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])                                                          #60
    #Z[:, :, 1:-1] = np.bitwise_or(np.bitwise_and(Z[:, :, :-2], bitwise_not(Z[:, :, 2:])), np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])) #62
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_or(np.bitwise_and(Z[:, :, 2:], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])))) #73
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])) #74
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 1:-1], bitwise_not(np.bitwise_and(Z[:, :, 2:], Z[:, :, :-2])))               #76
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(Z[:, :, :-2], np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]))) #78
    #Z[:, :, 1:-1] = np.bitwise_or(np.bitwise_and(Z[:, :, 1:-1], bitwise_not(Z[:, :, 2:])), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])) #94
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_xor(Z[:, :, :-2], np.bitwise_or(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]))))) #104
    #Z[:, :, 1:-1] = bitwise_not(np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])))               #105
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, :-2], np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]))                            #106
    #Z[:, :, 1:-1] = np.bitwise_or(np.bitwise_and(Z[:, :, 1:-1], bitwise_not(Z[:, :, 2:])), np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2])) #110
    #Z[:, :, 1:-1] = np.bitwise_or(np.bitwise_and(Z[:, :, 2:], bitwise_not(Z[:, :, 1:-1])), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])) #122
    #Z[:, :, 1:-1] = np.bitwise_or(np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])) #126
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 2:], np.bitwise_and(Z[:, :, 1:-1], Z[:, :, :-2]))                            #128
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], bitwise_not(np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1])))               #130
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 1:-1], bitwise_not(np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2])))               #132
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))) #134
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 1:-1], Z[:, :, :-2])                                                         #136
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], np.bitwise_or(Z[:, :, 1:-1], bitwise_not(Z[:, :, 2:])))                #138
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 1:-1], np.bitwise_or(Z[:, :, :-2], bitwise_not(Z[:, :, 2:])))                #140
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_or(np.bitwise_xor(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]))) #142    
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 2:], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))) #146
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))                            #150
    #Z[:, :, 1:-1] = np.bitwise_xor(np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]), np.bitwise_or(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]))) #152
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, :-2], np.bitwise_and(Z[:, :, 2:], bitwise_not(Z[:, :, 1:-1])))               #154
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_and(Z[:, :, 2:], bitwise_not(Z[:, :, :-2])))               #156
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, 2:], Z[:, :, :-2])                                                           #160
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], np.bitwise_or(Z[:, :, 2:], bitwise_not(Z[:, :, 1:-1])))                #162
    #Z[:, :, 1:-1] = np.bitwise_xor(np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]), np.bitwise_or(Z[:, :, 2:], np.bitwise_or(Z[:, :, 1:-1], Z[:, :, :-2]))) #164
    #Z[:, :, 1:-1] = np.bitwise_and(Z[:, :, :-2], np.bitwise_or(Z[:, :, 2:], Z[:, :, 1:-1]))                             #168
    #Z[:, :, 1:-1] = Z[:, :, :-2]                                                                                        #170
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 1:-1], np.bitwise_and(Z[:, :, 2:], np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))) #172
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_and(np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]), np.bitwise_xor(Z[:, :, 1:-1], Z[:, :, :-2]))) #178
    #Z[:, :, 1:-1] = np.bitwise_xor(Z[:, :, 2:], np.bitwise_and(Z[:, :, 1:-1], np.bitwise_xor(Z[:, :, 2:], Z[:, :, :-2]))) #184
    #Z[:, :, 1:-1] = np.bitwise_and(np.bitwise_or(Z[:, :, 2:], Z[:, :, 1:-1]), np.bitwise_or(Z[:, :, :-2], np.bitwise_and(Z[:, :, 2:], Z[:, :, 1:-1]))) #232
    return Z


def get_evolution(binary_images, num_of_images, image_length, ca_iterations):
    evolution = binary_images.reshape((num_of_images, image_length, image_length))
    state_cols = np.copy(evolution)
    state_rows = np.copy(evolution)
    for it in range(ca_iterations):
        state_cols = iterate_ca_cols(state_cols)
        state_rows = iterate_ca_rows(state_rows)
        evolution = np.append(evolution, np.bitwise_xor(state_cols, state_rows), axis=1)
    return evolution


def get_accuracy(predictions, targets):
    accuracy = np.abs(predictions - targets)
    accuracy[accuracy != 0] = 1
    accuracy = 1 - np.mean(accuracy)
    return accuracy


def bitwise_not(n, numbits=8):
    return (1 << numbits) - 1 - n


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)


    














