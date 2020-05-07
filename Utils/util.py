#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:49:42 2020

@author: vivek
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import glob
import librosa
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from natsort import natsorted 


def normalize(X):
#Input : X : audio signal of len = 16384
#Output : Normalized audio signal of len = 16384
#This function returns the normalized version of the audio signal [-1,1]
    
    X = X / np.max(np.abs(X))
    
    return X


def make_batch(file_dir, start, end):
#Input : file_dir containing the audio files
#        start : start index, end: end_index (for batching purposes)
#Output : Returns the batched audio files G_audio of shape [batch_size, 16384, 1]    

    #Initializing an empty list for appending the audio loaded from the file_dir
    audio = []
    
    #k is a flag variable that takes care of the batching of data
    #For eg. if start = 0, k = -1 (OR) if start = 500, k = 499. 
    #In the loop we check whether this k falls between start and end. If yes, we append the audio files with 
    #indice k, otherwise we do not append them to the list
    k = -1
    
    #Running the loop. We also ensure that the file_dir is sorted in ascending order
    for i in natsorted(os.listdir(file_dir)):
        #Incrementing k
        k += 1
        
        #Obtaining the path for the current file i
        f = glob.glob(os.path.join(file_dir,i))
           
        
        #Checking whether k is between start and end. If true, we load and append the data to the list audio
        if k >= start and k <= end:
            
            #Printing the name of the selected file (For reference)
            print(i)
            
            #Loading the audio file at a sampling rate of 16kHz (To be checked)
            #s is of shape (16384)
            s = librosa.load(f[0], 16000)[0]
            
            #Manually expanding the dimension so that when we convert the list to an array
            #we automatically obtain a 3D ndarray
            
            #Now s is of shape (16384,1)                        
            s = np.expand_dims(s, axis=1)
            
            #Appending s to the audio list
            audio.append(s)
    
    #Converting the list to a numpy array
    #audio is of shape [batch_size, 16384, 1]
    audio = np.array(audio)
    
    return audio



def save_audio_two(audio, args, iter_=0):
#Inputs: audio [batch_size, 16384, 1], iter_: Iteration index,  args. The passed audio is already normalized
#This function saves the generated audio in the respective folders
    
    #iter_ is incremented in irder to take care of the Python indexing
    iter_ += 1
    
    if args.expt_name == 'PGD':
        #Creating the folders to store the audio data
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_)))
        
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_)))
            
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_'+str(iter_)))
        
        
        
        #These commands save the generated audio files in the specified folders at a sampling rate of 16kHz
        #Here audio is a list of length 2. Every element of the list is of dimension [batch_size, 16384, 1]
        #For eg. audio[0][0,:,0] implies that we are accessing source s1 and the first generated audio for that source. Similarly for source s2
        
        
        [librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_))+'/'+ args.sname1+'_'+ str(i+args.start)+'.wav', audio[0][i,:,0], 16000) for i in range(args.batch_size)]
        
        [librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_))+'/'+ args.sname2+'_'+ str(i+args.start)+'.wav', audio[1][i,:,0], 16000) for i in range(args.batch_size)]
                
        
        #In these steps we create the mixture by directly adding the two generated sources
        for i in range(args.batch_size):
            #Initializing the mixture with zeros
            recon_mix = np.zeros(args.dim_m)
            
            for j in range(len(audio)):
                #Forming the mixtures
                recon_mix += audio[j][i,:,0]
            #Saving the mixture    
            librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_' + str(iter_))+'/'+'mix'+'_'+ str(i+args.start)+'.wav', recon_mix, 16000)

    else:
        
        k = iter_
        #Creating the folders to store the audio data
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1)):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1))
        
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2)):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2))
            
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix')):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix'))
        
        
        
        #These commands save the generated audio files in the specified folders at a sampling rate of 16kHz
        #Here audio is a list of length 2. Every element of the list is of dimension [batch_size, 16384, 1]
        #For eg. audio[0][0,:,0] implies that we are accessing source s1 and the first generated audio for that source. Similarly for source s2
        
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1)+'/'+ args.sname1+'_'+ str(args.start+k)+'.wav', audio[0][0,:,0], 16000)
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2)+'/'+ args.sname2+'_'+ str(args.start+k)+'.wav', audio[1][0,:,0], 16000)
        
           
        #In these steps we create the mixture by directly adding the two generated sources
        
        #Initializing the mixture with zeros
        recon_mix = np.zeros(args.dim_m)
            
        for j in range(len(audio)):
            #Forming the mixtures
            recon_mix += audio[j][0,:,0]
            #Saving the mixture    
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix')+'/'+'mix'+'_'+ str(k+args.start)+'.wav', recon_mix, 16000)



def save_audio_three(audio, args, iter_=0):
#Inputs: audio [3, batch_size, 16384, 1], iter_: Iteration index,  args. The passed audio is already normalized
#This function saves the generated audio in the respective folders
    
    #iter_ is incremented in irder to take care of the Python indexing
    iter_ += 1
    
    if args.expt_name == 'PGD':
        #Creating the folders to store the audio data
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_)))
        
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_)))
            
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3, 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3, 'iter_'+str(iter_)))
            
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_'+str(iter_))):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_'+str(iter_)))
        
        
        
        #These commands save the generated audio files in the specified folders at a sampling rate of 16kHz
        #Here audio is a list of length 2. Every element of the list is of dimension [batch_size, 16384, 1]
        #For eg. audio[0][0,:,0] implies that we are accessing source s1 and the first generated audio for that source. Similarly for source s2
        
        
        [librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1, 'iter_'+str(iter_))+'/'+ args.sname1+'_'+ str(i+args.start)+'.wav', audio[0][i,:,0], 16000) for i in range(args.batch_size)]
        
        [librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2, 'iter_'+str(iter_))+'/'+ args.sname2+'_'+ str(i+args.start)+'.wav', audio[1][i,:,0], 16000) for i in range(args.batch_size)]
        
        [librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3, 'iter_'+str(iter_))+'/'+ args.sname3+'_'+ str(i+args.start)+'.wav', audio[2][i,:,0], 16000) for i in range(args.batch_size)]
                
        
        #In these steps we create the mixture by directly adding the two generated sources
        for i in range(args.batch_size):
            #Initializing the mixture with zeros
            recon_mix = np.zeros(args.dim_m)
            
            for j in range(len(audio)):
                #Forming the mixtures
                recon_mix += audio[j][i,:,0]
            #Saving the mixture    
            librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix', 'iter_' + str(iter_))+'/'+'mix'+'_'+ str(i+args.start)+'.wav', recon_mix, 16000)

    else:
        
        k = iter_
        #Creating the folders to store the audio data
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1)):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1))
        
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2)):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2))
        
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3)):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3))
                
        if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix')):
            os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix'))
        
        
        
        #These commands save the generated audio files in the specified folders at a sampling rate of 16kHz
        #Here audio is a list of length 2. Every element of the list is of dimension [batch_size, 16384, 1]
        #For eg. audio[0][0,:,0] implies that we are accessing source s1 and the first generated audio for that source. Similarly for source s2
        
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname1)+'/'+ args.sname1+'_'+ str(args.start+k)+'.wav', audio[0][0,:,0], 16000)
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname2)+'/'+ args.sname2+'_'+ str(args.start+k)+'.wav', audio[1][0,:,0], 16000)
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_'+args.sname3)+'/'+ args.sname3+'_'+ str(args.start+k)+'.wav', audio[2][0,:,0], 16000)
        
           
        #In these steps we create the mixture by directly adding the two generated sources
        
        #Initializing the mixture with zeros
        recon_mix = np.zeros(args.dim_m)
            
        for j in range(len(audio)):
            #Forming the mixtures
            recon_mix += audio[j][0,:,0]
            #Saving the mixture    
        librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, args.expt_name, 'est_mix')+'/'+'mix'+'_'+ str(k+args.start)+'.wav', recon_mix, 16000)        



def invert_spectra_griffin_lim(X_mag, nfft=512, nhop=128, ngl=16):
#Reference : https://github.com/chrisdonahue/wavegan
#This function performs the Griffin Lim inversion to obtain raw audio waveform
#from magnitude spectrogram 
    
    X = tf.complex(X_mag, tf.zeros_like(X_mag))

    def b(i, X_best):
        x = tf.contrib.signal.inverse_stft(X_best, nfft, nhop)
        X_est = tf.contrib.signal.stft(x, nfft, nhop)
        phase = X_est / tf.cast(tf.maximum(1e-8, tf.abs(X_est)), tf.complex64)
        X_best = X * phase
        return i + 1, X_best

    i = tf.constant(0)
    c = lambda i, _: tf.less(i, ngl)
    _, X = tf.while_loop(c, b, [i, X], back_prop=False)

    x = tf.contrib.signal.inverse_stft(X, nfft, nhop)
    x = x[:, :16384]

    return x




def magnitude_spectrogram(audio, args):
#Inputs: audio [batch_size, 16384, 1], args
#Output : Magnitude spectrogram [batch_size, frames(T), fft_unique_bins, 1]
#         fft_unique_bins = fft_length // 2 + 1    

#This function calculates the magnitude spectrogram of raw audio

#References
#https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/signal/shape_ops.py#L56
#http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/contrib/signal/stft.html

    #audio is reshaped so that it can be utilized by the contrib.signal.stft version (This step can be avoided if using higher versions of TF)
    audio_reshaped = tf.reshape(audio, [args.batch_size, args.dim_m])    
    spec_audio_reshaped = tf.contrib.signal.stft(audio_reshaped, frame_length=256, frame_step=128, fft_length=256, window_fn=tf.contrib.signal.hann_window, pad_end=False, name=None)
    mag_spec_audio_reshaped = tf.expand_dims(tf.abs(spec_audio_reshaped), axis=3)
    
    return mag_spec_audio_reshaped


def log_magnitude_spectrogram(mag_spec):
#Input : Magnitude spectrogram [batch_size, frames(T), fft_unique_bins, 1]
#Output : Log Magnitude spectrogram [batch_size, frames(T), fft_unique_bins, 1]

#The number '1' is added for imrpoved perceptual quality [https://arxiv.org/pdf/1810.09785.pdf]
    return tf.log(1+(mag_spec)**2)



def spec_plot(s, title):
#Input : Input raw audio s of 16384 samples, title of the plot
#This function is used to plot the spectrogram of audio in a separate figure
    
    #To create a separate plot
    plt.figure(figsize=(12, 8))
    
    #Obtaining the magnitude spectrogram in dB
    D = librosa.amplitude_to_db(np.abs(librosa.stft(np.reshape(s,-1))), ref=np.max)
    
    #Displaying the spectrogram
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()
    

def get_random_snippets(audio_list):

#Input : audio_list: A list containing multiple audio clips >1s
#This function randomly snips 16384 samples from every respective audio clips and stores it back in the array
    
    for i in range(len(audio_list)):
        rand_index = np.random.choice(range(len(audio_list[i])-16384),1, replace=False)
        audio_list[i] = audio_list[i][int(rand_index):int(rand_index)+16384]
    
    return audio_list


def check_vars_of_ckpt(args, name):
    #We first take in the checkpoint name and directory for either digits, drums or piano
    ckpt_name = os.path.join(args.model_path, args.sname1, name+'.ckpt')
    
    #In-built command for printing only the names of the variables contained within the checkpoint file
    #This command can also be used to print the name of a specific tensor 
    print_tensors_in_checkpoint_file(ckpt_name, all_tensors=True, tensor_name='')

        
        
        
    