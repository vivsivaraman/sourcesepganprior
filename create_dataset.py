#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:45:26 2020

@author: vivek
"""

import numpy as np
import os
import librosa
import argparse
from Utils.util import get_random_snippets, normalize



def create_folders(args):
    
    # This function creates all the folders that are required for the experiments

    # Creating the Ground truth digits (gt_digits) folder
    if not os.path.exists(os.path.join(args.results_dir, 'gt_digit')):
        os.makedirs(os.path.join(args.results_dir, 'gt_digit'))
    
    # Creating the Ground truth drums (gt_drums) folder
    if not os.path.exists(os.path.join(args.results_dir, 'gt_drums')):
        os.makedirs(os.path.join(args.results_dir, 'gt_drums'))
        
    # Creating the Ground truth piano (gt_piano) folder
    if not os.path.exists(os.path.join(args.results_dir, 'gt_piano')):
        os.makedirs(os.path.join(args.results_dir, 'gt_piano'))
    
    # Creating the main folder for the current experiment
    if not os.path.exists(os.path.join(args.results_dir, args.mix_type)):
        os.makedirs(os.path.join(args.results_dir, args.mix_type))
        
    # Creating the folder that contains the ground truth mixtures within the current experiment
    if not os.path.exists(os.path.join(args.results_dir, args.mix_type, 'gt_mix')):
        os.makedirs(os.path.join(args.results_dir, args.mix_type, 'gt_mix'))
    
    # Creating the folder for the type of method within the current experiment
    if not os.path.exists(os.path.join(args.results_dir, args.mix_type, args.expt_name)):
        os.makedirs(os.path.join(args.results_dir, args.mix_type, args.expt_name))
    
 
def create_data(args):    
    
    #This function returns the mixture audio depending on args.mix_type
    
       
    #if args.mix_type == ('digit_drums' or 'digit_piano' or 'drums_piano'), the function randomly chooses a digit recording and a 
    #drums recording and naiively adds them to  produce the mixture 
    #recording. It also saves the randomly chosen original digit recording in the given directory. 
    
    if args.sample_gt_audio == 'True':
        
        #This command returns an array of strings (Array of str832) containing the names of the audio files within the specified directory
        digit_selected = np.random.choice(os.listdir(os.path.join(args.data_dir, 'digit', 'test')), args.num_mixtures, replace=False)
        
        #This command creates a list containing the relative paths of the selected audio clips
        digit_selected = [os.path.join(args.data_dir, 'digit', 'test')+'/'+digit_selected[i] for i in range(len(digit_selected))]
        
        #This command returns an array of strings (Array of str832) containing the names of the audio files within the specified directory
        drums_selected = np.random.choice(os.listdir(os.path.join(args.data_dir, 'drums', 'test')), args.num_mixtures, replace=True)
        
        #This command creates a list containing the relative paths of the selected audio clips
        drums_selected = [os.path.join(args.data_dir, 'drums', 'test')+'/'+drums_selected[i] for i in range(len(drums_selected))]
        
        #This command returns an array of strings (Array of str832) containing the names of the audio files within the specified directory            
        #We give replace = True for piano because the number of wav files within the test directory is relatively small. 
        piano_selected = np.random.choice(os.listdir(os.path.join(args.data_dir, 'piano', 'test')), args.num_mixtures, replace=True)
        
        #This command creates a list containing the relative paths of the selected audio clips
        piano_selected = [os.path.join(args.data_dir, 'piano', 'test')+'/'+piano_selected[i] for i in range(len(piano_selected))]
        
        #Reference : https://librosa.github.io/librosa/generated/librosa.util.fix_length.html
        #digit_audio, drums_audio and piano_audio are list containing the gt_audio samples
        digit_audio = [librosa.util.fix_length(librosa.load(i, 16000)[0], 16384) for i in digit_selected]
        drums_audio = [librosa.util.fix_length(librosa.load(i, 16000)[0], 16384) for i in drums_selected]
        piano_audio = [librosa.load(i, 16000)[0] for i in piano_selected]
        
        #Since piano audio has a duration >1s, we use the following function to randomly snip 16384 samples from the audio given
        piano_audio = get_random_snippets(piano_audio)
        
        #Saving the audio
        if args.normalize == 'True':
            print('Saving Digits')
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_digit')+'/'+'digit_'+str(k+args.start)+'.wav'), normalize(digit_audio[k]), 16000) for k in range(len(digit_audio))]
            
            print('Saving Drums')
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_drums')+'/'+'drums_'+str(k+args.start)+'.wav'), normalize(drums_audio[k]), 16000) for k in range(len(drums_audio))]
            
            print('Saving Piano')
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_piano')+'/'+'piano_'+str(k+args.start)+'.wav'), normalize(piano_audio[k]), 16000) for k in range(len(piano_audio))]
          
        else:
            
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_digit')+'/'+'digit_'+str(k+args.start)+'.wav'), (digit_audio[k]), 16000) for k in range(len(digit_audio))]
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_drums')+'/'+'drums_'+str(k+args.start)+'.wav'), (drums_audio[k]), 16000) for k in range(len(drums_audio))]
            [librosa.output.write_wav((os.path.join(args.results_dir,'gt_piano')+'/'+'piano_'+str(k+args.start)+'.wav'), (piano_audio[k]), 16000) for k in range(len(piano_audio))]
            
    
    if args.create_mix == 'True':
        #Splits args.mix_type into the two or three words , depending on what was given
        source_names = str.split(args.mix_type, '_')
    
        #Creating mixtures and saving them
        for i in range(args.num_mixtures):
            mix = np.zeros(args.dim_m)
            print(('Creating Mixture {}').format(i+args.start))
            for name in source_names:
                mix += librosa.load(os.path.join(args.results_dir,'gt_'+name)+'/'+name+'_'+str(i+args.start)+'.wav', 16000)[0]
        
            librosa.output.write_wav(os.path.join(args.results_dir, args.mix_type, 'gt_mix')+'/'+'mix_'+str(i+args.start)+'.wav', mix, 16000)
                


parser = argparse.ArgumentParser()
    
parser.add_argument('--data_dir', type=str, default = './data', help = 'Data directory for the sources')
parser.add_argument('--results_dir', type=str, default = './results', help = 'Results directory')
parser.add_argument('--sname1', type=str, default = 'digit', help = 'Name of source 1') #Either "digit" or "drums" or "piano'
parser.add_argument('--sname2', type=str, default = 'drums', help = 'Name of source 2') #Either "digit" or "drums" or "piano'
parser.add_argument('--mix_type', type=str, default = 'digit_drums', help = 'Mix Type')
parser.add_argument('--expt_name', type=str, default = 'PGD', help = 'Experiment Name')
parser.add_argument('--normalize', type=str, default = 'True', help = 'Normalize or not')
parser.add_argument('--sample_gt_audio', type=str, default = 'True', help = 'Sample Ground Truth audio')
parser.add_argument('--create_mix', type=str, default = 'True', help = 'Create mixtures or not')
parser.add_argument('--start', type=int, default = 800, help = 'Indexing for the audio')
parser.add_argument('--dim_m', type=int, default = 16384, help = 'Dimension of mixture (OR) No. of samples in the mixture')
parser.add_argument('--num_mixtures', type=int, default = 200, help = 'No. of Mixtures')
     

create_folders(parser.parse_args())
create_data(parser.parse_args())    