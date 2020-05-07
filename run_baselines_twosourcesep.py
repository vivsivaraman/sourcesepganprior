#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:29:55 2020

@author: vivek
"""
import numpy as np
from sklearn.decomposition import FastICA, PCA, KernelPCA, NMF
import argparse
from Utils.util import normalize, make_batch, save_audio_two
from evaluate import compute_SDR_SIR_two, compute_SDR_mixture_two, compute_SNR_spec, envelope_two

#All the baseline models require atleast two observations in order to 
#estimate the sources. In other words, these methods work well for over-determined cases. 
#So, in order to compare them to our work, we artificially create mixtures that have a similar additive 
#mixing pattern.

#The first mixture = s1 + s2 ; The second mixture = s1 + s2 + noise of very small variance  


#Baseline 1 (FastICA)
def compute_fastICA(s1, s2 , args, k):
    
#Inputs : s1 and s2 are the gt_source signals each of len 16384.
#Outputs : SDRs, SIRs, ENV and Spectral SNR for every example mixture 
    
    #Reshaping s1 and s2 into (16384, 1), so that we can concatenate along axis =1
    #It is done in order to support the input shape for the sklearn functions
    #source_1 and source_2 are of shape [16384, 1]
    source_1 = np.reshape(s1, (args.dim_m, 1))
    source_2 = np.reshape(s2, (args.dim_m, 1))
    
    #Observation 1 (Additive mixing)
    #x1 is of shape [16384, 1]
    x1 = 1.0*source_1 + 1.0*source_2
    
    #Observation 2 (Additive mixing + noise of small variance)
    #x2 is of shape [16384, 1]
    x2 = 1.0*source_1 + 1.0*source_2 + args.noise_var*np.random.uniform(args.dim_m, 1)
    
    #Concatenating along axis 1 to obtain the observation matrix
    #X is of shape [16384, 2]
    X = np.concatenate((x1, x2), axis=1)
    
    #Using sklearn FastICA library (2 components as we have only two sources)
    transformer = FastICA(n_components = 2,random_state=0)
    
    #Obtain the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed = transformer.fit_transform(X)
    
    #Normalizing the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed[:,0] = normalize(X_transformed[:, 0])
    X_transformed[:,1] = normalize(X_transformed[:, 1])
    
    #Transposing X_transformed so that it is in a format that can be used by SDR evaluation funtion
    #X_transformed is of shape [2, 16384]    
    X_transformed = X_transformed.T
    
    #These steps make X_transformed a matrix of shape [2, 1, 16384, 1]
    #This can be directly used by the evaluate.py function
    X_transformed = np.expand_dims(X_transformed, axis=2)
    X_transformed = np.expand_dims(X_transformed, axis=1)
    
    #Save the estimated audio files. Here k is an index used for correctly naming the estimated sources and mixtures
    save_audio_two(X_transformed, args, k)
    
    #Calling the compute_SDR_SIR_two function to compute SDR and SIR per example
    #Computing Spectral SNR as well as Envelope distance
    #s1 and s2 are of shape [1, 16384, 1]
    #X_transformed is of shape [2, 1,16384, 1]
    
    sdr_s1, sdr_s2, sir_s1, sir_s2 = compute_SDR_SIR_two(s1, s2, X_transformed, args)
    sdr_m = compute_SDR_mixture_two((s1+s2), X_transformed, args)    
    
    snr_spec_s1 = compute_SNR_spec(s1, X_transformed[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(s2, X_transformed[1,:,:,:], args, args.sname2)
    snr_spec_mix = compute_SNR_spec(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(s1, X_transformed[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(s2, X_transformed[1,:,:,:], args, args.sname2)
    env_mix = envelope_two(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    
    return sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix



#Baseline 2 (PCA)
def compute_PCA(s1, s2 , args, k):
        
#Inputs : s1 and s2 are the gt_source signals each of len 16384.
#Outputs : SDRs, SIRs, ENV and Spectral SNR for every example mixture 
    
    #Reshaping s1 and s2 into (16384, 1), so that we can concatenate along axis =1
    #It is done in order to support the input shape for the sklearn functions
    #source_1 and source_2 are of shape [16384, 1]
    source_1 = np.reshape(s1, (args.dim_m, 1))
    source_2 = np.reshape(s2, (args.dim_m, 1))
    
    #Observation 1 (Additive mixing)
    #x1 is of shape [16384, 1]
    x1 = 1.0*source_1 + 1.0*source_2
    
    #Observation 2 (Additive mixing + noise of small variance)
    #x2 is of shape [16384, 1]
    x2 = 1.0*source_1 + 1.0*source_2 + args.noise_var*np.random.uniform(args.dim_m, 1)
    
    #Concatenating along axis 1 to obtain the observation matrix
    #X is of shape [16384, 2]
    X = np.concatenate((x1, x2), axis=1)
    
    #Using sklearn PCA library (2 components as we have only two sources)
    pca = PCA(n_components=2)
    
    #Obtain the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed = pca.fit_transform(X)
        
    #Normalizing the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed[:,0] = normalize(X_transformed[:, 0])
    X_transformed[:,1] = normalize(X_transformed[:, 1])
    
    #Transposing X_transformed so that it is in a format that can be used by SDR evaluation funtion
    #X_transformed is of shape [2, 16384]    
    X_transformed = X_transformed.T
    
    #These steps make X_transformed a matrix of shape [2, 1, 16384, 1]
    #This can be directly used by the evaluate.py function
    X_transformed = np.expand_dims(X_transformed, axis=2)
    X_transformed = np.expand_dims(X_transformed, axis=1)
    
    #Save the estimated audio files. Here k is an index used for correctly naming the estimated sources and mixtures
    save_audio_two(X_transformed, args, k)
    
    #Calling the compute_SDR_SIR_two function to compute SDR and SIR per example
    #Computing Spectral SNR as well as Envelope distance
    #s1 and s2 are each of shape [1, 16384, 1]
    #X_transformed is of shape [2, 1,16384, 1]
    
    sdr_s1, sdr_s2, sir_s1, sir_s2 = compute_SDR_SIR_two(s1, s2, X_transformed, args)
    sdr_m = compute_SDR_mixture_two((s1+s2), X_transformed, args)    
    
    snr_spec_s1 = compute_SNR_spec(s1, X_transformed[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(s2, X_transformed[1,:,:,:], args, args.sname2)
    snr_spec_mix = compute_SNR_spec(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(s1, X_transformed[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(s2, X_transformed[1,:,:,:], args, args.sname2)
    env_mix = envelope_two(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    
    return sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix


#Baseline 3 (Kernel PCA)
def compute_kPCA(s1, s2 , args, k):
#Inputs : s1 and s2 are the gt_source signals each of len 16384.
#Outputs : SDRs, SIRs, ENV and Spectral SNR for every example mixture 
    
    #Reshaping s1 and s2 into (16384, 1), so that we can concatenate along axis =1
    #It is done in order to support the input shape for the sklearn functions
    #source_1 and source_2 are of shape [16384, 1]
    source_1 = np.reshape(s1, (args.dim_m, 1))
    source_2 = np.reshape(s2, (args.dim_m, 1))
    
    #Observation 1 (Additive mixing)
    #x1 is of shape [16384, 1]
    x1 = 1.0*source_1 + 1.0*source_2
    
    #Observation 2 (Additive mixing + noise of small variance)
    #x2 is of shape [16384, 1]
    x2 = 1.0*source_1 + 1.0*source_2 + args.noise_var*np.random.uniform(args.dim_m, 1)
    
    #Concatenating along axis 1 to obtain the observation matrix
    #X is of shape [16384, 2]
    X = np.concatenate((x1, x2), axis=1)
    
    #Using sklearn kernelPCA library (2 components as we have only two sources)
    kpca = KernelPCA(n_components=2)
    
    #Obtain the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed = kpca.fit_transform(X)
        
    #Normalizing the predicted sources
    #X_transformed is of shape [16384, 2]
    X_transformed[:,0] = normalize(X_transformed[:, 0])
    X_transformed[:,1] = normalize(X_transformed[:, 1])
    
    #Transposing X_transformed so that it is in a format that can be used by SDR evaluation funtion
    #X_transformed is of shape [2, 16384]    
    X_transformed = X_transformed.T
    
    #These steps make X_transformed a matrix of shape [2, 1, 16384, 1]
    #This can be directly used by the evaluate.py function
    X_transformed = np.expand_dims(X_transformed, axis=2)
    X_transformed = np.expand_dims(X_transformed, axis=1)
    
    #Save the estimated audio files. Here k is an index used for correctly naming the estimated sources and mixtures
    save_audio_two(X_transformed, args, k)
    
    #Calling the compute_SDR_SIR_two function to compute SDR and SIR per example
    #Computing Spectral SNR as well as Envelope distance
    #s1 and s2 are each of shape [1, 16384, 1]
    #X_transformed is of shape [2, 1,16384, 1]
    
    sdr_s1, sdr_s2, sir_s1, sir_s2 = compute_SDR_SIR_two(s1, s2, X_transformed, args)
    sdr_m = compute_SDR_mixture_two((s1+s2), X_transformed, args)    
    
    snr_spec_s1 = compute_SNR_spec(s1, X_transformed[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(s2, X_transformed[1,:,:,:], args, args.sname2)
    snr_spec_mix = compute_SNR_spec(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(s1, X_transformed[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(s2, X_transformed[1,:,:,:], args, args.sname2)
    env_mix = envelope_two(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    
    return sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix

#Baseline 4 (NMF)
def compute_NMF(s1, s2, args, k):
#Inputs : s1 and s2 are the gt_source signals each of len 16384.
#Outputs : SDRs, SIRs, ENV and Spectral SNR for every example mixture 
    
    #Reshaping s1 and s2 into (16384, 1), so that we can concatenate along axis =1
    #It is done in order to support the input shape for the sklearn functions
    #source_1 and source_2 are of shape [16384, 1]
    source_1 = np.reshape(s1, (args.dim_m, 1))
    source_2 = np.reshape(s2, (args.dim_m, 1))
    
    #Observation 1 (Additive mixing)
    #x1 is of shape [16384, 1]
    x1 = 1.0*source_1 + 1.0*source_2
    
    #Observation 2 (Additive mixing + noise of small variance)
    #x2 is of shape [16384, 1]
    x2 = 1.0*source_1 + 1.0*source_2 + args.noise_var*np.random.uniform(args.dim_m, 1)
    
    #Concatenating along axis 1 to obtain the observation matrix
    #X is of shape [16384, 2]
    X = np.concatenate((x1, x2), axis=1)
    
    #Using sklearn NMF library (2 components as we have only two sources)
    model = NMF(n_components=2, init='random', random_state=0)
    
    #Adding an arbitrary constant to the data samples in order to make the entire dataset positive (>=0)
    #As NMF only deals with non negative data
    W = model.fit_transform(X + 3.0)
    H = model.components_
    
    #Estimating the sources and subtracting 3.0 to obtain our original sources    
    source_1_hat = np.expand_dims(np.matmul(W, H[:,0:1]), axis=0)-3.0
    source_2_hat = np.expand_dims(np.matmul(W, H[:,1:2]), axis=0)-3.0
        
    #Normalizing estimated source 1 and 2
    source_1_hat = normalize(source_1_hat)
    source_2_hat = normalize(source_2_hat)
    
    #X_transformed has a shape [2, 16384, 1]    
    X_transformed = np.concatenate((source_1_hat, source_2_hat), axis=0)
    
    #X_transformed now has a shape of [2,1,16384,1]
    X_transformed = np.expand_dims(X_transformed, axis=1)
    
        
    #Save the estimated audio files. Here k is an index used for correctly naming the estimated sources and mixtures
    save_audio_two(X_transformed, args, k)
    
    #Calling the compute_SDR_SIR_two function to compute SDR and SIR per example
    #Computing Spectral SNR as well as Envelope distance
    #s1 and s2 are each of shape [1, 16384, 1]
    #X_transformed is of shape [2, 1,16384, 1]
    
    sdr_s1, sdr_s2, sir_s1, sir_s2 = compute_SDR_SIR_two(s1, s2, X_transformed, args)
    sdr_m = compute_SDR_mixture_two((s1+s2), X_transformed, args)    
    
    snr_spec_s1 = compute_SNR_spec(s1, X_transformed[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(s2, X_transformed[1,:,:,:], args, args.sname2)
    snr_spec_mix = compute_SNR_spec(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(s1, X_transformed[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(s2, X_transformed[1,:,:,:], args, args.sname2)
    env_mix = envelope_two(s1+s2, X_transformed[0,:,:,:]+X_transformed[1,:,:,:], args, 'mix')
    
    
    return sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix





def run_baseline_expt(args):
    
    gt_s1 = make_batch(args.results_dir+'/gt_'+args.sname1, args.start, args.end)
    gt_s2 = make_batch(args.results_dir+'/gt_'+args.sname2, args.start, args.end)
    

    SDR_s1, SDR_s2, SDR_mix = [], [], []
    
    SIR_s1, SIR_s2 = [], []
    
    ENV_s1, ENV_s2, ENV_mix = [], [], []
    
    SNR_SPEC_s1, SNR_SPEC_s2, SNR_SPEC_mix = [], [], []
    
    for i in range(gt_s1.shape[0]):
        
        if args.expt_name == 'FastICA':
            sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix  = compute_fastICA(gt_s1[i:i+1], gt_s2[i:i+1], args, i)
        elif args.expt_name == 'PCA':
            sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix  = compute_PCA(gt_s1[i:i+1], gt_s2[i:i+1], args, i)
        elif args.expt_name == 'KernelPCA':
            sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix  = compute_kPCA(gt_s1[i:i+1], gt_s2[i:i+1], args, i)
        elif args.expt_name == 'NMF':
            sdr_s1, sdr_s2, sir_s1, sir_s2, sdr_m, snr_spec_s1, snr_spec_s2, snr_spec_mix, env_s1, env_s2, env_mix  = compute_NMF(gt_s1[i:i+1], gt_s2[i:i+1], args, i)
        
        SDR_s1.append(sdr_s1)
        SDR_s2.append(sdr_s2)
        SDR_mix.append(sdr_m)
        
        SIR_s1.append(sir_s1)
        SIR_s2.append(sir_s2)
               
        ENV_s1.append(env_s1) 
        ENV_s2.append(env_s2)
        ENV_mix.append(env_mix)
        
        SNR_SPEC_s1.append(snr_spec_s1)
        SNR_SPEC_s2.append(snr_spec_s2)
        SNR_SPEC_mix.append(snr_spec_mix)
        
    
    print(('Experiment {}').format(args.expt_name))    
    print(('Median SDR {} = {}').format(args.sname1, np.nanmedian(SDR_s1)))
    print(('Median SDR {} = {}').format(args.sname2, np.nanmedian(SDR_s2)))
    print(('Median SDR {} = {}').format('Mixture', np.nanmedian(SDR_mix)))
    print('########')
    
    print(('Median SIR {} = {}').format(args.sname1, np.nanmedian(SIR_s1)))
    print(('Median SIR {} = {}').format(args.sname2, np.nanmedian(SIR_s2)))
    print('########')
    
    print(('Median Spectral SNR {} = {}').format(args.sname1, np.nanmedian(SNR_SPEC_s1)))
    print(('Median Spectral SNR {} = {}').format(args.sname2, np.nanmedian(SNR_SPEC_s2)))
    print(('Median Spectral SNR {} = {}').format('Mixture', np.nanmedian(SNR_SPEC_mix)))
    print('########')
    
    print(('Median Envelope Distance: {} = {}').format(args.sname1, np.nanmedian(ENV_s1)))
    print(('Median Envelope Distance: {} = {}').format(args.sname2, np.nanmedian(ENV_s2)))
    print(('Median Envelope Distance: {} = {}').format('Mixture', np.nanmedian(ENV_mix)))
    
   

parser = argparse.ArgumentParser()
    
parser.add_argument('--model_dir', type=str, default = './ckpts', help = 'Pre-trained model (checkpoint) dir for source 1 and 2')
parser.add_argument('--data_dir', type=str, default = './data', help = 'Data directory for the sources')
parser.add_argument('--results_dir', type=str, default = './results', help = 'Results directory')
parser.add_argument('--sname1', type=str, default = 'digit', help = 'Name of source 1') #Either "digit" or "drums" or "piano'
parser.add_argument('--sname2', type=str, default = 'drums', help = 'Name of source 2') #Either "digit" or "drums" or "piano'
parser.add_argument('--mix_type', type=str, default = 'digit_drums', help = 'Mix Type')
parser.add_argument('--expt_name', type=str, default = 'NMF', help = 'Experiment Name')
parser.add_argument('--normalize', type=str, default = 'True', help = 'Normalize or not')
parser.add_argument('--bokeh_plot_filename', type=str, default = 'loss_plot', help = 'Filename for storing bokeh plot of Loss vs Iterations')

parser.add_argument('--dim_m', type=int, default = 16384, help = 'Dimension of mixture (OR) No. of samples in the mixture')
parser.add_argument('--lambda1', type=float, default = 0.1, help = 'Weight for Inclusion Loss')
parser.add_argument('--lambda2', type=float, default = 0.3, help = 'Weight for Multiresolution Spectral Loss')
parser.add_argument('--lambda3', type=float, default = 0.8, help = 'Weight for Ratio frequency Loss')
parser.add_argument('--lambda4', type=float, default = 0.4, help = 'Weight for Exclusion Loss')
parser.add_argument('--batch_size', type=int, default = 1000, help = 'Batch size')
parser.add_argument('--start', type=int, default = 0, help = 'Starting Mixture Number')
parser.add_argument('--end', type=int, default = 1, help = 'Ending Mixture Number')
parser.add_argument('--learning_rate', type=float, default = 0.05, help = 'learning rate')
parser.add_argument('--num_mixtures', type=int, default = 1000, help = 'No. of Mixtures')
parser.add_argument('--noise_var', type=float, default = 0.0001, help = 'Noise variance for forming the mixtures')

run_baseline_expt(parser.parse_args())
