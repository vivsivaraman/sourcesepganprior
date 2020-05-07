#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:35:54 2020

@author: vivek
"""

import numpy as np
import librosa
import museval
from Utils.util import make_batch
from scipy.signal import hilbert
import os


def compute_metrics_two(X, est, args):
    
    #est is of shape [2, batch_size, 16384, 1]
    est = np.array(est)
    
    #gt_s1 and gt_s2 are both of shape [batch_size, 16384,1]
    gt_s1 = make_batch(args.results_dir+'/gt_'+args.sname1, args.start, args.end)
    gt_s2 = make_batch(args.results_dir+'/gt_'+args.sname2, args.start, args.end)
    
    sdr_s1, sdr_s2, sir_s1, sir_s2 = compute_SDR_SIR_two(gt_s1, gt_s2, est, args)
    sdr_m = compute_SDR_mixture_two(X, est, args)
    
    snr_spec_s1 = compute_SNR_spec(gt_s1, est[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(gt_s2, est[1,:,:,:], args, args.sname2)
    snr_spec_mix = compute_SNR_spec(X, est[0,:,:,:]+est[1,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(gt_s1, est[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(gt_s2, est[1,:,:,:], args, args.sname2)
    env_mix = envelope_two(X, est[0,:,:,:]+est[1,:,:,:], args, 'mix')
    
    print(('Median SDR {} = {}').format(args.sname1, sdr_s1))
    print(('Median SDR {} = {}').format(args.sname2, sdr_s2))
    print(('Median SDR {} = {}').format('Mixture', sdr_m))
    print('########')
    
    print(('Median SIR {} = {}').format(args.sname1, sir_s1))
    print(('Median SIR {} = {}').format(args.sname2, sir_s2))
    print('########')
    
    print(('Median Spectral SNR {} = {}').format(args.sname1, snr_spec_s1))
    print(('Median Spectral SNR {} = {}').format(args.sname2, snr_spec_s2))
    print(('Median Spectral SNR {} = {}').format('Mixture', snr_spec_mix))
    print('########')
    
    print(('Median Envelope Distance: {} = {}').format(args.sname1, env_s1))
    print(('Median Envelope Distance: {} = {}').format(args.sname2, env_s2))
    print(('Median Envelope Distance: {} = {}').format('Mixture', env_mix))
    
    
    
    
    
    
    
def compute_SDR_SIR_two(gt_s1, gt_s2, est, args):

    SDR, SIR = [], []
    
    for i in range(gt_s1.shape[0]):
        sources = np.concatenate((gt_s1[i:i+1], gt_s2[i:i+1]), axis=0)
        metrics = museval.metrics.bss_eval(sources, est[:,i,:,:], window=args.dim_m)
        SDR.append(metrics[0])
        SIR.append(metrics[2])
            
        
    SDR = np.array(SDR)
    SDR = np.reshape(SDR, (SDR.shape[0], SDR.shape[1]))
        
    SIR = np.array(SIR)
    SIR = np.reshape(SIR, (SIR.shape[0], SIR.shape[1]))
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    source_names = str.split(args.mix_type, '_')
    
    for n in range(len(source_names)):
    
        with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SDR_'+source_names[n]+'.txt', "a") as f:
            np.savetxt(f, SDR[:,n])
        
        f.close()
        
        with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SIR_'+source_names[n]+'.txt', "a") as f:
            np.savetxt(f, SIR[:,n])
        
        f.close()     

    return np.nanmedian(SDR[:,0]), np.nanmedian(SDR[:,1]), np.nanmean(SIR[:,0]), np.nanmean(SIR[:,1])




def compute_SDR_mixture_two(mix, est, args):
    
    SDR = []
    for i in range(mix.shape[0]):
        est_mix = est[0,i:i+1,:,:] + est[1,i:i+1,:,:]
        metrics = museval.metrics.bss_eval(mix[i:i+1], est_mix)
        SDR.append(metrics[0])
            
    SDR = np.array(SDR)
    SDR = np.reshape(SDR, (SDR.shape[0], SDR.shape[1]))
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SDR_mix.txt', "a") as f:
        np.savetxt(f, SDR)
    
    f.close()
    
    
    return np.nanmedian(SDR[:,0])



def compute_SNR_spec(s, est_s, args, name):
    
    #Reference : http://www.ient.rwth-aachen.de/services/bib2web/pdf/SpGn09a.pdf
    
    SNR = []
    
    for j in range(s.shape[0]):
        ref_source = np.reshape(s[j],-1)
        est_source = np.reshape(est_s[j],-1)
            
        ref_source_spec = np.abs(librosa.core.stft(ref_source, n_fft=256, hop_length=128, win_length=256))
        est_source_spec = np.abs(librosa.core.stft(est_source, n_fft=256, hop_length=128, win_length=256))
            
        snr = np.sum(ref_source_spec**2) / np.sum((ref_source_spec - est_source_spec)**2)
        SNR.append(10.0*np.log10(snr))
        
    
    SNR = np.array(SNR)
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SNR_spec_'+name+'.txt', "a") as f:
        np.savetxt(f, SNR)
    
    f.close()
    
    
    return np.nanmedian(SNR)




def compute_envelope(signal):
    
    #Reference : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    
    signal = np.reshape(signal, -1)
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
        
    return amplitude_envelope
    
    
def compute_distance_envelope(s1, s2):
    
    #Reference : https://arxiv.org/pdf/1809.02587.pdf
    
    env_dist = np.sqrt(np.mean((s1 - s2)**2))
        
    return env_dist

    
def envelope_two(s, est_s, args, name):
    
    env = []
    for j in range(s.shape[0]):
        s1 = np.reshape(s[j],-1)
        est_s1 = np.reshape(est_s[j],-1)
            
        env_s1 = compute_envelope(s1)
        env_est_s1 = compute_envelope(est_s1)
            
        env.append(compute_distance_envelope(env_s1, env_est_s1))
    
    env = np.array(env)
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'env_'+name+'.txt', "a") as f:
        np.savetxt(f, env)
    
    f.close()
        
    return np.nanmedian(env)
    
    
    
          
def compute_metrics_three(X, est, args):
    
    #est is of shape [3, batch_size, 16384, 1]
    est = np.array(est)
    
    #gt_s1 and gt_s2 and gt_s3 are both of shape [batch_size, 16384,1]
    gt_s1 = make_batch(args.results_dir+'/gt_'+args.sname1, args.start, args.end)
    gt_s2 = make_batch(args.results_dir+'/gt_'+args.sname2, args.start, args.end)
    gt_s3 = make_batch(args.results_dir+'/gt_'+args.sname3, args.start, args.end)
    
    sdr_s1, sdr_s2, sdr_s3, sir_s1, sir_s2, sir_s3 = compute_SDR_SIR_three(gt_s1, gt_s2, gt_s3, est, args)
    sdr_m = compute_SDR_mixture_three(X, est, args)
    
    snr_spec_s1 = compute_SNR_spec(gt_s1, est[0,:,:,:], args, args.sname1)
    snr_spec_s2 = compute_SNR_spec(gt_s2, est[1,:,:,:], args, args.sname2)
    snr_spec_s3 = compute_SNR_spec(gt_s3, est[2,:,:,:], args, args.sname3)
    snr_spec_mix = compute_SNR_spec(X, est[0,:,:,:]+est[1,:,:,:]+est[2,:,:,:], args, 'mix')
    
    env_s1 = envelope_two(gt_s1, est[0,:,:,:], args, args.sname1)
    env_s2 = envelope_two(gt_s2, est[1,:,:,:], args, args.sname2)
    env_s3 = envelope_two(gt_s3, est[2,:,:,:], args, args.sname3)
    env_mix = envelope_two(X, est[0,:,:,:]+est[1,:,:,:]+est[2,:,:,:], args, 'mix')
    
    print(('Median SDR {} = {}').format(args.sname1, sdr_s1))
    print(('Median SDR {} = {}').format(args.sname2, sdr_s2))
    print(('Median SDR {} = {}').format(args.sname3, sdr_s3))
    print(('Median SDR {} = {}').format('Mixture', sdr_m))
    print('########')
    
    print(('Median SIR {} = {}').format(args.sname1, sir_s1))
    print(('Median SIR {} = {}').format(args.sname2, sir_s2))
    print(('Median SIR {} = {}').format(args.sname3, sir_s3))
    print('########')
    
    print(('Median Spectral SNR {} = {}').format(args.sname1, snr_spec_s1))
    print(('Median Spectral SNR {} = {}').format(args.sname2, snr_spec_s2))
    print(('Median Spectral SNR {} = {}').format(args.sname3, snr_spec_s3))
    print(('Median Spectral SNR {} = {}').format('Mixture', snr_spec_mix))
    print('########')
    
    print(('Median Envelope Distance: {} = {}').format(args.sname1, env_s1))
    print(('Median Envelope Distance: {} = {}').format(args.sname2, env_s2))
    print(('Median Envelope Distance: {} = {}').format(args.sname3, env_s3))
    print(('Median Envelope Distance: {} = {}').format('Mixture', env_mix))
    
    
    
    
    
    
    
def compute_SDR_SIR_three(gt_s1, gt_s2, gt_s3, est, args):

    SDR, SIR = [], []
    
    for i in range(gt_s1.shape[0]):
        sources = np.concatenate((gt_s1[i:i+1], gt_s2[i:i+1], gt_s3[i:i+1]), axis=0)
        metrics = museval.metrics.bss_eval(sources, est[:,i,:,:], window=args.dim_m)
        SDR.append(metrics[0])
        SIR.append(metrics[2])
            
        
    SDR = np.array(SDR)
    SDR = np.reshape(SDR, (SDR.shape[0], SDR.shape[1]))
        
    SIR = np.array(SIR)
    SIR = np.reshape(SIR, (SIR.shape[0], SIR.shape[1]))
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    
    source_names = str.split(args.mix_type, '_')
    
    for n in range(len(source_names)):
    
        with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SDR_'+source_names[n]+'.txt', "a") as f:
            np.savetxt(f, SDR[:,n])
        
        f.close()
        
        with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SIR_'+source_names[n]+'.txt', "a") as f:
            np.savetxt(f, SIR[:,n])
        
        f.close()    

    return np.nanmedian(SDR[:,0]), np.nanmedian(SDR[:,1]), np.nanmedian(SDR[:,2]), np.nanmean(SIR[:,0]), np.nanmean(SIR[:,1]), np.nanmean(SIR[:,2])


def compute_SDR_mixture_three(mix, est, args):
    
    SDR = []
    for i in range(mix.shape[0]):
        est_mix = est[0,i:i+1,:,:] + est[1,i:i+1,:,:] + est[2,i:i+1,:,:]
        metrics = museval.metrics.bss_eval(mix[i:i+1], est_mix)
        SDR.append(metrics[0])
            
    SDR = np.array(SDR)
    SDR = np.reshape(SDR, (SDR.shape[0], SDR.shape[1]))
    
    if not os.path.exists(args.results_dir+'/'+args.mix_type+'/'+args.expt_name):
        os.makedirs(args.results_dir+'/'+args.mix_type+'/'+args.expt_name)
    
    with open(args.results_dir+'/'+args.mix_type+'/'+args.expt_name+'/'+'SDR_mix.txt', "a") as f:
        np.savetxt(f, SDR)
    
    f.close()
    
    
    return np.nanmedian(SDR[:,0])    
    
    
           
