#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:47:11 2020

@author: vivek
"""

import tensorflow as tf
    

def compute_exclusion_loss(img1,img2,level=1):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)
               
        alphax=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))
        alphay=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))
        
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1

        gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
        grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)

        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
    
    loss_gradxy=tf.reduce_sum(sum(gradx_loss)/3.0)+tf.reduce_sum(sum(grady_loss)/3.0)
    loss_grad= loss_gradxy/2.0
    return loss_grad

def compute_gradient(img):
    gradx=img[:,1:,:,:]-img[:,:-1,:,:]
    grady=img[:,:,1:,:]-img[:,:,:-1,:]
    return gradx,grady


def multires_spectral_loss(img1, img2, level=1):
    loss = []
    for l in range(level):
        c = tf.reshape(img2, (img2.shape[0] , img2.shape[1]*img2.shape[2]))
        b = tf.reshape(img1, (img1.shape[0] , img1.shape[1]*img1.shape[2]))
        loss.append(1.0*tf.reduce_mean(tf.reduce_sum(tf.abs(c-b), axis=1), axis=0))
        
        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
    
    return sum(loss)


def freq_loss(img1, img2):
    c = []
    
    for i in range(img1.shape[1]):
        
        a = tf.reduce_sum(tf.log(1.0 + img1[:,i,:,:])+1e-6,axis=1) 
        b = tf.reduce_sum(tf.log(1.0 + img2[:,i,:,:])+1e-6,axis=1)
        c.append(tf.reduce_mean(a/b, axis=0))
    return sum(c)

        
