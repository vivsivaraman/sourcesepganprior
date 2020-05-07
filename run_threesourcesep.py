import tensorflow as tf
import os
import argparse
from wavegan import WaveGANGenerator as generator
from Utils.log import ResultsLog
from Utils.losses import compute_exclusion_loss, multires_spectral_loss, freq_loss
from Utils.util import magnitude_spectrogram, log_magnitude_spectrogram, make_batch, save_audio_three
from evaluate import compute_metrics_three 
   

##################################################################################
def main(args):
    
    # We only consider 3 sources in the mixture
    n_sources = 3
    
    #true_mixtures is a TF placeholder of shape [batch_size, 16384, 1]
    true_mixture = tf.placeholder(tf.float32,[args.batch_size, args.dim_m, 1])
    
    #Initializing a list for three z's where the z vectors get updated during every TF iteration
    z_placeholder = []
    
    #We execute the 'for loop' n_sources no.of times 
    for i in range(n_sources):
        #Every z is of dimension [batch_size, 100]. (Since the z dimension used in the pre-trained WaveGAN model is = 100)
        #Every z is initialized by 0's of shape [batch_size, 100]
        z_placeholder.append(tf.Variable(tf.zeros([args.batch_size, 100],dtype=tf.float32),trainable=True,name="z_prior_"+str(i)))
    
        
    # Initializing a list to store the audio obtained from the pre-trained generator model
    G_audio = []
    
    # G_mixture is used to sum up the sources obtained from the respective generator models to produce the synthesized mixture
    # G_mixture has a shape [batch_size, 16384, 1]
    G_mixture = tf.zeros_like(true_mixture)
    
    #We use assert to ensure that both the source names are not the same. And also a random source name is not entered
    #Also we ensure that the batch_size is the same as the difference between args.end and args.start+1
    assert args.sname1 != args.sname2
    assert args.sname2 != args.sname3
    assert args.sname1 in ['digit', 'drums', 'piano']
    assert args.sname2 in ['digit', 'drums', 'piano']
    assert args.sname3 in ['digit', 'drums', 'piano']
    assert args.batch_size == args.end - args. start+ 1
    
    #We choose the name of the tf.variable_scope based on the names of the original scope while training the digit/drums/piano dataset using the WaveGAN model
    #This first set of if-else conditions are for source 1
    if args.sname1 == 'digit':
        scope1 = 'G'
    elif args.sname1 == 'drums':
        scope1 = 'G1'
    else:  #Piano
        scope1 = 'G2'
    
    #This second set of if-else conditions are for source 2
    if args.sname2 == 'digit':
        scope2 = 'G'
    elif args.sname2 == 'drums':
        scope2 = 'G1'
    else:  #Piano
        scope2 = 'G2'
        
    #This third set of if-else conditions are for source 3
    if args.sname3 == 'digit':
        scope3 = 'G'
    elif args.sname3 == 'drums':
        scope3 = 'G1'
    else:  #Piano
        scope3 = 'G2'
        
    
    #######################################################################################
    #Generating source 1 from the pre-trained model for source 1 using z_placeholder[0]
    #Since we are generating source 1, we use scope1
    with tf.variable_scope(scope1):
        #We pass z_placeholder[0] to the generator of the WaveGAN model to produce source 1
        #gen_s1 is of size [batch_size, 16384, 1]
        gen_s1 = generator(z_placeholder[0], train=False)
        
        if args.normalize == 'True':
        #Normalize the generated audio,  gen_s1 such that each example of gen_s1 is in the range of [-1, 1]
        #gen_s1 is of size [batch_size, 16384, 1]. Normalization scheme adapted from the WaveGAN model 
            gen_s1 = gen_s1/tf.expand_dims(tf.reduce_max(tf.abs(gen_s1), axis=1), axis=2)
        
        #Appending the normalized gen_s1 to the empty list G_audio
        #The len(G_audio) is now 1, where the first element is gen_s1
        G_audio.append(gen_s1)
        
        #Synthesizing mixture by first adding gen_s1
        #G_mixture is of shape [batch_size, 16384, 1]
        G_mixture += gen_s1
        
        #Collecting the global variables of the generator under this scope1
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope1)  
        
        #Using tf.train.Saver so that the pre-trained weights of the generator under this scope can be restored. 
        saver1 = tf.train.Saver(gen_vars)
    
    #Compute magnitude spectrogram of gen_s1
    #mag_spec_gen_s1 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    mag_spec_gen_s1 = magnitude_spectrogram(gen_s1, args)
    
    #Compute log magnitude spectrogram of gen_s1
    #log mag_spec_gen_s1 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    log_mag_spec_gen_s1 = log_magnitude_spectrogram(mag_spec_gen_s1)
        
    
    
    #Generating source 2 from the pre-trained model for source 2 using z_placeholder[1]
    #Since we are generating source 2, we use scope2
    with tf.variable_scope(scope2):
        #We pass z_placeholder[1] to the generator of the WaveGAN model to produce source 2
        #gen_s2 is of size [batch_size, 16384, 1]
        gen_s2 = generator(z_placeholder[1], train=False)
        
        if args.normalize == 'True':
        #Normalize the generated audio,  gen_s2 such that each example of gen_s1 is in the range of [-1, 1]
        #gen_s2 is of size [batch_size, 16384, 1]. Normalization scheme adapted from the WaveGAN model 
            gen_s2 = gen_s2/tf.expand_dims(tf.reduce_max(tf.abs(gen_s2), axis=1), axis=2)
        
        #Appending the normalized gen_s2 to the list G_audio
        #The len(G_audio) is now 2, where the first element is gen_s1 and the second element is gen_s2
        G_audio.append(gen_s2)
        
        #Synthesizing mixture by last adding gen_s2
        #G_mixture is of shape [batch_size, 16384, 1] (I have also checked whether the addition takes place 'in-place')
        G_mixture += gen_s2
        
        #Collecting the global variables of the generator under this scope2
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope2) 
        
        #Using tf.train.Saver so that the pre-trained weights of the generator under this scope can be restored. 
        saver2 = tf.train.Saver(gen_vars)
        
    #Compute magnitude spectrogram of gen_s2
    #mag_spec_gen_s2 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    mag_spec_gen_s2 = magnitude_spectrogram(gen_s2, args)
    
    #Compute log magnitude spectrogram of gen_s2
    #log mag_spec_gen_s2 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    log_mag_spec_gen_s2 = log_magnitude_spectrogram(mag_spec_gen_s2)
    
    
    
    #Generating source 3 from the pre-trained model for source 3 using z_placeholder[2]
    #Since we are generating source 3, we use scope3
    with tf.variable_scope(scope3):
        #We pass z_placeholder[2] to the generator of the WaveGAN model to produce source 3
        #gen_s3 is of size [batch_size, 16384, 1]
        gen_s3 = generator(z_placeholder[2], train=False)
        
        if args.normalize == 'True':
        #Normalize the generated audio,  gen_s3 such that each example of gen_s3 is in the range of [-1, 1]
        #gen_s3 is of size [batch_size, 16384, 1]. Normalization scheme adapted from the WaveGAN model 
            gen_s3 = gen_s3/tf.expand_dims(tf.reduce_max(tf.abs(gen_s3), axis=1), axis=2)
        
        #Appending the normalized gen_s3 to the list G_audio
        #The len(G_audio) is now 3, where the first element is gen_s1 and the second element is gen_s2 and the third element is gen_s3
        G_audio.append(gen_s3)
        
        #Synthesizing mixture by last adding gen_s3
        #G_mixture is of shape [batch_size, 16384, 1] (I have also checked whether the addition takes place 'in-place')
        G_mixture += gen_s3
        
        #Collecting the global variables of the generator under this scope3
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope3) 
        
        #Using tf.train.Saver so that the pre-trained weights of the generator under this scope can be restored. 
        saver3 = tf.train.Saver(gen_vars)
        
    #Compute magnitude spectrogram of gen_s3
    #mag_spec_gen_s3 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    mag_spec_gen_s3 = magnitude_spectrogram(gen_s3, args)
    
    #Compute log magnitude spectrogram of gen_s3
    #log mag_spec_gen_s3 is of shape [batch_size, 127(Time), 129(Frequency), 1]
    log_mag_spec_gen_s3 = log_magnitude_spectrogram(mag_spec_gen_s3)
        
     
    
    
    #We now determine the magnitude and log magnitude spectrogram for the synthesized mixture
    #mag_spec_G_mixture is of shape [batch_size, 127(Time), 129(Frequency), 1]        
    mag_spec_G_mixture = magnitude_spectrogram(G_mixture, args) #(b_1)
    
    #log mag_spec_G_mixed is of shape [batch_size, 127(Time), 129(Frequency), 1]
    log_mag_spec_G_mixture = log_magnitude_spectrogram(mag_spec_G_mixture)  #(b)
    
    
    #We now determine the magnitude and log magnitude spectrogram for the ground truth mixture
    #mag_spec_true_mixture is of shape [batch_size, 127(Time), 129(Frequency), 1]        
    mag_spec_true_mixture = magnitude_spectrogram(true_mixture, args) #(c_1)
    
    #log mag_spec_true_mixture is of shape [batch_size, 127(Time), 129(Frequency), 1]
    log_mag_spec_true_mixture = log_magnitude_spectrogram(mag_spec_true_mixture) #(c)
    
    
    
    
    #############################################################################################
    #Using loss functions
    
    #Not included for the current work (May be included if necessary)        
    #l_MSE = 1.0*tf.reduce_mean(tf.reduce_sum((true_mixture - G_mixture)**2, axis=1), axis=0)
    #l_MAE = 1.0*tf.reduce_mean(tf.reduce_sum(tf.abs(true_mixture - G_mixture), axis=1), axis=0)
    
    #All loss functions return a single value
    
    #l_inc is the Inclusion loss which is the negative of the exclusion loss. Hence the -ve sign in front
    l_inc = -compute_exclusion_loss(log_mag_spec_true_mixture, log_mag_spec_G_mixture, level=3)
    
    l_spec = multires_spectral_loss(log_mag_spec_true_mixture, log_mag_spec_G_mixture, level=3)
    
    l_freq = freq_loss(mag_spec_true_mixture, mag_spec_G_mixture)
    
    l_exc1 = compute_exclusion_loss(log_mag_spec_gen_s1, log_mag_spec_gen_s2, level=3)
    l_exc2 = compute_exclusion_loss(log_mag_spec_gen_s1, log_mag_spec_gen_s3, level=3)
    l_exc3 = compute_exclusion_loss(log_mag_spec_gen_s2, log_mag_spec_gen_s3, level=3)
    
    l_exc = (l_exc1 + l_exc2 + l_exc3)/3.0
        
     
    #Total loss    
    l = args.lambda1*l_inc + args.lambda2*l_spec + args.lambda3*l_freq + args.lambda4*l_exc     
    
    #############################################################################################    
    
    #Dataset Preparation
    print('Preparing Dataset')
    # Make a batch of the true_mixtures
    # X is of shape [batch_size, 16384, 1] ; X is a numpy ndarray
    X = make_batch(args.results_dir+'/'+args.mix_type+'/'+'gt_mix', args.start, args.end)
    print('Dataset prepared')
    
    #############################################################################################    
    
    #Results logging
    #For storing the results in the given directory
    results_file = os.path.join(args.results_dir,args.mix_type, args.expt_name, args.bokeh_plot_filename)
    results = ResultsLog(results_file)
    
    #############################################################################################    
    
    
    ## Define Optimizer and perform projection onto the generator input manifold
    
    #ADAM Optimizer with a learning rate 0.05 (default)    
    optimizer_PGD = tf.train.AdamOptimizer(learning_rate = args.learning_rate)
    
    #We minimize our loss function l wrt z (Projected Gradient Descent) 
    g_opt_PGD = optimizer_PGD.minimize(l, var_list= [z_placeholder])
    
    #To check whether there is a gradient flow between l and z
    grad = tf.gradients(l, z_placeholder)
    
    #Clipping z (projection)
    z_placeholder[0] = tf.clip_by_value(z_placeholder[0], clip_value_min=-0.9, clip_value_max=0.9)
    z_placeholder[1] = tf.clip_by_value(z_placeholder[1], clip_value_min=-0.9, clip_value_max=0.9)
        
    
    #############################################################################################
    #############################################################################################       
    
    #Starting the TF session
    sess = tf.Session()
    
    #Initializing the TF variables
    sess.run(tf.global_variables_initializer())
    
    #Restoring the pre-trained models
    saver1.restore(sess, args.model_dir+'/'+args.sname1+'/'+scope1+'.ckpt')
    saver2.restore(sess, args.model_dir+'/'+args.sname2+'/'+scope2+'.ckpt')
    saver3.restore(sess, args.model_dir+'/'+args.sname3+'/'+scope3+'.ckpt')
    
    print(('Restored {} checkpoint').format(args.sname1))
    print(('Restored {} checkpoint').format(args.sname2))
    print(('Restored {} checkpoint').format(args.sname3))
    
    
    ##############################################################################################       
       
    # Execute the TF session  
    
    #Running the algorithm for args.iterations times
    for iter_ in range(args.iterations):
        
        # Here loss is the total loss of shape ()
        # loss_inc, loss_spec, loss_freq, loss_exc are the individual losses each of shape ()
        # audio_gen_mixture is a list of [gen_s1, gen_s2, gen_s3] which is updated during every iteration
        # audio_true_mixture is an array containing the originally passed input data X (For verification)
        # z is a list of [z_placeholder[0], z_placeholder[1], z_placeholder[2]] which gets updated during every iteration
        # g is the gradient of l wrt z
        _, loss, loss_inc, loss_spec, loss_freq, loss_exc, audio_gen_mixture, audio_true_mixture, z, g = sess.run([g_opt_PGD, l, l_inc, l_spec, l_freq, l_exc, G_audio, true_mixture, z_placeholder, grad],feed_dict={true_mixture:X})
        
        print(('Iteration {} Total Loss {}').format(iter_, loss))
        
        
        #We store the respective loss values every 10 iterations of the algorithm (Can be modified)
        #These lines of code create a .csv file and a bokeh plot to tabulate the loss values.
        if ((iter_+1) % 10) == 0:
            
            res = {'iter': iter_+1, 'Total Loss l': loss, 'Inclusion Loss l1': loss_inc, 'Spectral Loss l2': loss_spec, 'Freq Loss l3': loss_freq, 'Exclusion Loss l4': loss_exc}
            plot_loss = ['Total Loss l']+['Inclusion Loss l1']+['Spectral Loss l2']+['Freq Loss l3']+['Exclusion Loss l4']
            results.add(**res)
            
            results.plot(x='iter', y=plot_loss, title='PGD '+args.mix_type+' loss plot', ylabel='Loss')
            results.save()
        
        
        # We store the generated sources s1 and s2 and s3 after every 200 iterations.
        # We also save the synthesized mixture as a sum the two generated audio sources
        if ((iter_+1) % 200) == 0: 
            
            save_audio_three(audio_gen_mixture, args, iter_) 
            
            
    #Terminating the TF session        
    sess.close()
        
    

    #Compute Metrics (SDR, SIR, Spectral SNR, Envelope Distance)
    compute_metrics_three(X, audio_gen_mixture, args)   
        



if __name__ == '__main__':
    
    #This work supports only mono-channel audio recordings
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default = './ckpts', help = 'Pre-trained model (checkpoint) dir for source 1 and 2')
    parser.add_argument('--data_dir', type=str, default = './data', help = 'Data directory for the sources')
    parser.add_argument('--results_dir', type=str, default = './results', help = 'Results directory')
    parser.add_argument('--sname1', type=str, default = 'digit', help = 'Name of source 1') #Either "digit" or "drums" or "piano'
    parser.add_argument('--sname2', type=str, default = 'drums', help = 'Name of source 2') #Either "digit" or "drums" or "piano'
    parser.add_argument('--sname3', type=str, default = 'piano', help = 'Name of source 3') #Either "digit" or "drums" or "piano'
    parser.add_argument('--mix_type', type=str, default = 'digit_drums_piano', help = 'Mix Type')
    parser.add_argument('--expt_name', type=str, default = 'PGD', help = 'Experiment Name')
    parser.add_argument('--bokeh_plot_filename', type=str, default = 'loss_plot', help = 'Filename for storing bokeh plot of Loss vs Iterations')
    parser.add_argument('--normalize', type=str, default = 'True', help = 'Normalize or not')
    
    parser.add_argument('--dim_m', type=int, default = 16384, help = 'Dimension of mixture (OR) No. of samples in the mixture')
    parser.add_argument('--lambda1', type=float, default = 0.1, help = 'Weight for Inclusion Loss')
    parser.add_argument('--lambda2', type=float, default = 0.3, help = 'Weight for Multiresolution Spectral Loss')
    parser.add_argument('--lambda3', type=float, default = 0.8, help = 'Weight for Ratio frequency Loss')
    parser.add_argument('--lambda4', type=float, default = 0.4, help = 'Weight for Exclusion Loss')
    parser.add_argument('--batch_size', type=int, default = 200, help = 'Batch size')
    parser.add_argument('--start', type=int, default = 0, help = 'Starting Mixture Number')
    parser.add_argument('--end', type=int, default = 1, help = 'Ending Mixture Number')
    parser.add_argument('--learning_rate', type=float, default = 0.05, help = 'learning rate')
    parser.add_argument('--iterations', type=int, default = 1000, help = 'No. of iterations')
         
    
    main(parser.parse_args())


