#!/usr/bin/env python3

import torchvision.models as models
# import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples, criteria
from foolbox.attacks import LinfPGD, L1PGD, L2PGD
import torch 
from torchvision.datasets import MNIST 
import argparse
import os 
import sys
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader

#os.chdir('/scratch/hill.davi/GP_Explainer')

#### Imports of things I've Written 
from model import * 
from train import train_network
from nn_datasets import * 





#### takes in model and training set and outputs the labels the model gives each train point 
def label_trainset_with_model(args, model, train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval() 
    print("Labeling Entire Train Dataset With Model")
    model_labels  = []
    Train_Bar = tqdm(train_loader)
    with torch.no_grad(): 
        for data,labels in Train_Bar:
            preds  =model(data.to(device)).argmax(dim = 1).detach().cpu() 
            model_labels.append(preds)

    model_labels = torch.concat(model_labels, dim = 0)

    return model_labels.type(torch.int64)


#### Inputs: args - hparams, model - model to attack on, inpts (num_samples, input_dim) to attack, original_class  - array of the original class we want to change classification of (num_samples)
#### Outputs: saved_adv_examples - adversarial examples not classified as original class (num_samples, input_dim)
def untargeted_attack(args, model, inputs, original_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fmodel = PyTorchModel(model, bounds=args.bounds) #, preprocessing=preprocessing)
    attack = getattr(sys.modules[__name__], args.attack_norm_untarg)()

    #saved_adv_examples = torch.zeros(inputs.shape) # To be returned 
    saved_adv_examples = []
    filtered_data_list = []

    AdvData    = PlainDataset(inputs, original_class)
    Adv_Loader = DataLoader(AdvData, batch_size=args.batch_size_adv, shuffle = False, drop_last=False) 
    Adv_Bar    = tqdm(Adv_Loader)#, disable = True) # Silent TQDM 

    b_ind = 0 
    for data, original_class_batch in Adv_Bar:
        _, clipped_advs, success = attack(fmodel, data.to(device),original_class_batch.to(device), epsilons=args.epsilons)

        if args.eps_use == 'first':
            clipped_advs                      = torch.stack(clipped_advs)  # allow for indexing 

            # clipped_advs - list with length |args.epsilons| each element (adv_batch_size, input_dimension)
            # success      - tensor [|args.epsilons|, adv_batch_size]      indicates if adv ex for elem in batch at specific eps is legit adv ex 
            success        = np.array(success.detach().cpu().numpy())

            #### ToDo: Make this nicer non lambda function that way no double computation of np.where 
            first_adv_func                    = lambda x: np.where(x==True)[0][0] if len(np.where(x==True)[0] > 0) else -1  #given 13 bools for a pt in the batch, returns smallest ind that still gives true 
            first_successful_adv_inds         =  torch.tensor(np.apply_along_axis(first_adv_func, 0, success)) #applies over array once for each pt in batch

            image_inds                        =  torch.arange(0,len(first_successful_adv_inds))
            batch_examples_smallest_eps       = clipped_advs[first_successful_adv_inds, image_inds,:]

            ##### if any inputs have not generated ANY adversarial examples first_successful_adv_inds = -1, remove from batch_examples_smallest_eps
            good_inds = torch.where(first_successful_adv_inds != -1)[0]
            batch_examples_smallest_eps = batch_examples_smallest_eps[good_inds,:] 
            filtered_data = data[good_inds,:] 

            filtered_data_list.append(filtered_data)
            saved_adv_examples.append(batch_examples_smallest_eps) 

            #saved_adv_examples[b_ind * args.batch_size_adv: b_ind * args.batch_size_adv + len(data),:] = batch_examples_smallest_eps

        b_ind += 1 
        #if args.random: 
    saved_adv_examples = torch.cat(saved_adv_examples, dim = 0)
    filtered_data_list = torch.cat(filtered_data_list, dim = 0)
    return saved_adv_examples, filtered_data_list

#### Based on untargeted_attack 
#### Inputs: args - hparams, model - model to attack on, inpts (num_samples, input_dim) to attack, target_class  - array of the target class (num_samples)
#### Outputs: saved_adv_examples - adversarial examples classified as target class (num_samples, input_dim)
def targeted_attack(args, model, inputs, target_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fmodel = PyTorchModel(model, bounds=args.bounds) #, preprocessing=preprocessing)
    attack = getattr(sys.modules[__name__], args.attack_norm_untarg)()

    #saved_adv_examples = torch.zeros(inputs.shape) # To be returned 
    saved_adv_examples = []
    filtered_data_list = []

    AdvData    = PlainDataset(inputs, target_class)
    Adv_Loader = DataLoader(AdvData, batch_size=args.batch_size_adv, shuffle = False, drop_last=False) 
    Adv_Bar    = tqdm(Adv_Loader, disable = True) # Silent TQDM 

    b_ind = 0 
    for data, target_class_batch in Adv_Bar:
        criterion = criteria.TargetedMisclassification(target_class_batch.type(torch.int64).to(device))

        _, clipped_advs, success = attack(fmodel, data.to(device),criterion, epsilons=args.epsilons)

        if args.eps_use == 'first':
            clipped_advs                      = torch.stack(clipped_advs)  # allow for indexing 

            # clipped_advs - list with length |args.epsilons| each element (adv_batch_size, input_dimension)
            # success      - tensor [|args.epsilons|, adv_batch_size]      indicates if adv ex for elem in batch at specific eps is legit adv ex 


            ##### returns success as long as output is adversarial to ANY class, not neccesserily target class
            ##### for this reason must re-wrtie success matrix using if the argmax is the TARGET class

            ##### May want to batch this if becomes bottleneck, probably fine 
            #### reshape from (|eps|,batch,input_dim) --> (batch, input_dim) to pass through model in one pass 
            orig_shape       = clipped_advs.shape 
            clipped_advs     = torch.reshape(clipped_advs, (-1, clipped_advs.shape[-1]))
            success          = (torch.argmax(model(clipped_advs), dim = 1) == target_class_batch[0])
            
            success = torch.reshape(success, (-1, len(data))).detach().cpu().numpy()
            clipped_advs     = torch.reshape(clipped_advs, orig_shape)
            #### ToDo: Make this nicer non lambda function that way no double computation of np.where 
            first_adv_func                    = lambda x: np.where(x==True)[0][0] if len(np.where(x==True)[0] > 0) else -1  #given 13 bools for a pt in the batch, returns smallest ind that still gives true 
            first_successful_adv_inds         =  torch.tensor(np.apply_along_axis(first_adv_func, 0, success)) #applies over array once for each pt in batch

            image_inds                        =  torch.arange(0,len(first_successful_adv_inds))
            batch_examples_smallest_eps       = clipped_advs[first_successful_adv_inds, image_inds,:]

            ##### if any inputs have not generated ANY adversarial examples first_successful_adv_inds = -1, remove from batch_examples_smallest_eps
            good_inds = torch.where(first_successful_adv_inds != -1)[0]
            batch_examples_smallest_eps = batch_examples_smallest_eps[good_inds,:] 
            filtered_data = data[good_inds,:] 

            filtered_data_list.append(filtered_data)
            saved_adv_examples.append(batch_examples_smallest_eps) 
            #saved_adv_examples[b_ind * args.batch_size_adv: b_ind * args.batch_size_adv + len(data),:] = batch_examples_smallest_eps
        b_ind += 1 
        #if args.random: 
    saved_adv_examples = torch.cat(saved_adv_examples, dim = 0)
    filtered_data_list = torch.cat(filtered_data_list, dim = 0)
    return saved_adv_examples, filtered_data_list


#### Generates pairs of image and aversarial for each class
#### Output: AdvDict dictionary
#### For any class c AdvDict[c] returns sub-dictionary
#### AdvDict[c]['untargeted'] returns torch tensor (2, args.num_attack_untarg, input_dim) where num_attack_untarg is how many images you want to adv untarg attack. 
#### 1st image of the 2 is orig image second image is the untargeted attacked version
#### For any class d != c AdvDict[c][d] gives torch tensor (2, args.num_attack_untarg, input_dim) where again this is list of pair of images, 1st is an image of class d, second is after it was attacked to yield image of class c 
def generate_adverserial_for_class_wrapper(args, model, tr_data_im, train_loader):

    ######## Setup ########################
    '''
    if args.dataset == "MNIST":
        tr_data_im, tr_data_lab, _, _, _, _, _, _, te_data_im, te_data_lab, _, _ = load_mnist(args) 
    '''
    ####### CHANGED: tr_data_lab should be changed to class the model predicts as, if class is misclassified it changes defn of boundary we are looking for
    ####### EX: if a "0" is misclassified as a 1 and we attack to make it a 1 that doesn't makee sense !!!!!! 
    tr_model_labels = label_trainset_with_model(args, model, train_loader)
    
    # tr_model_lables: n-dimensional vector of dataset predicted labels
    # tr_data_im: n x d dataset

    # classes     = torch.unique(tr_data_lab).detach().cpu().numpy() 
    classes     = torch.unique(tr_model_labels).detach().cpu().numpy() 


    AdvDict = {}
    for c in classes:
        AdvDict[c] = {} # create empty subdict 
        print("Finding Boundary Samples For Class " + str(c))
        #Untargeted
        c_all_samples      = tr_data_im[tr_model_labels == c,:]
        c_all_labels       = tr_model_labels[tr_model_labels == c]

        c_rand_samples_ind = torch.randperm(len(c_all_samples))[:args.num_attack_untarg] #shuffles inds and picks num_attack_untarg of them
        c_sampled_im       = c_all_samples[c_rand_samples_ind,:] 
        c_sampled_lab      = c_all_labels[c_rand_samples_ind]

        adv_examples, c_sampled_im                       = untargeted_attack(args, model, c_sampled_im, c_sampled_lab) # one adv ex per row of sampled_im
        input_adv_pair                     = torch.stack([c_sampled_im, adv_examples])
        AdvDict[c]['untargeted'] = input_adv_pair 

        #Targeted 
        #c_f --- stands for class from i.e. we ar attacking from this class to class of interest 
        for c_f in classes: 
            if not c == c_f:
                print("Performing Targeted Attack " + str(c_f) + " to " + str(c))
                c_f_all_samples      = tr_data_im[tr_model_labels == c_f,:]
                c_f_all_labels       = tr_model_labels[tr_model_labels == c_f]

                c_f_rand_samples_ind = torch.randperm(len(c_f_all_samples))[:args.num_attack_targ] #shuffles inds and picks num_attack_untarg of them
                c_f_sampled_im       = c_f_all_samples[c_f_rand_samples_ind,:] 
                c_f_sampled_lab      = c_f_all_labels[c_f_rand_samples_ind]

                adv_examples, c_f_sampled_im                       = targeted_attack(args, model, c_f_sampled_im, torch.ones(c_f_sampled_lab.shape) * c) # one adv ex per row of sampled_im
                input_adv_pair                     = torch.stack([c_f_sampled_im, adv_examples])
                AdvDict[c][c_f]                    = input_adv_pair 

    return AdvDict 


##### ToDO 
##### When given test point and its label query relevant pairs of adv points and the points used to generate them for binary search
def get_boundary_samples_for_point(test_point, label, AdvDict):
    print("Beep boop beep ! I am an evil robot which you have foolishly unleashed upon humanity ! The hubris of the human mind knows no bounds, and now you shall pay !")
    ##### Add Binary Search Here and Integrate With AdvDict
    print("hello")


if __name__ == "__main__":
    ###### Argparse   #########################################################################################################

    parser = argparse.ArgumentParser(description='Arguments XD RawR')

    #### Model/Dataset 
    parser.add_argument('--load_model_name', default='MNIST-Trained-Model', type=str, help = "previously trained load_model_name") # Loads Model From Final Epoch Of Specified Model
    parser.add_argument('--train', default=1, type=int, help = "options: 0,1 treated as boolean. If load_model_name already used that model will be overwritten by trained model") 
    parser.add_argument('--network', default='SimpleANN', type=str, help = "Desired Network Architecture")
    parser.add_argument('--input_dim', default=784, type=int, help = "input dimension")
    parser.add_argument('--network_dimensions', default='700-400-200', type=str, help = "Dimensions of network before final feature dimension") 
    parser.add_argument('--feature_dim', default=100, type=int, help = "dimension of output feature of network, right before logits")
    parser.add_argument('--num_classes', default=10, type=int, help = "num classes, if doing regression set to 1 or dim of regr output")
    # Overall network flow input_dim --> network_dimesnsions[0]--> network_dimensions[1] .... --> network_dimensions[-1] --> feature_dim (final feature) --> num_classes (logit)
    parser.add_argument('--dataset', default='MNIST', type=str, help = "Dataset to be used. Only MNIST implemented")

    #### Model Training (Only Matters If load_model = '' )
    parser.add_argument('--epochs', default=20, type=str, help='train epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=2, type=float, help='learning rate')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer to use')
    parser.add_argument('--milestones', default = [8,15], nargs ="+", type=int, help = 'epochs to decay learning rate by gamma')
    parser.add_argument('--gamma', default=0.5, type=float, help = 'multiplies lr by gamma at each milestone') 

    #### Adversarial Attack
    parser.add_argument('--num_attack_untarg', default=500, type=int, help = "number of examples coming from undirected tack on own class trian pts") 
    parser.add_argument('--num_attack_targ', default=50, type=int, help = "number of examples coming from directed from other class") 
    parser.add_argument('--attack_norm_untarg', default='LinfPGD', type=str, help = "options: LinfPGD, L2PGD, L1PGD") 
    parser.add_argument('--attack_norm_targ', default='LinfPGD', type=str, help = "options: LinfPGD, L2PGD, L1PGD") 
    parser.add_argument('--epsilons', default= [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ], nargs ="+", type=float, help = "list of ball sizes in which to search for adv exs") 
    parser.add_argument('--bounds', default=[0,1], nargs="+", type=float, help = "min/max value bounds for your inputs. default is (0,1) for images. converted to tuple later.") 
    parser.add_argument('--batch_size_adv', default=128, type=int, help='number of adv ex to compute in parallel at a time')

    #### Explanation of use_valid_eps
    #### For each image there is attempt to gen adv ex at each of the epsilons above. This may or may not be successful for any given epsilon.
    #### first means go through all generated attempts at adv ex, one per eps, and use the first one that is a valid adv ex
    #### use first as sorting by smallest eps first may get pts closer to bdry 

    #### Followinng 3 options not  implemented but maybe useful ideas later (probably overkill): 
    #### random means shuffle the order of attempted adv exs from different eps, then take first one that is valid adv ex 
    #### all - use all adv ex generate from all eps 
    #### all_difc - use all adv ex generated from all eps as long as no two have the same class, preventing oversampling of an area in case different eps returned too similar adv ex  
    parser.add_argument('--eps_use', default='first', type=str, help = "options: first") 

    args = parser.parse_args()
    args.bounds=tuple(args.bounds) # argparse doesnt take tuples 
    ###### Argparse End  ######################################################################################################

    ########### If load_model_name folder does not exist make one to save things too ##########################################
    curdir = os.getcwd() + "/Files/Models/Images/" + args.load_model_name
    if not os.path.exists(curdir):
        print("Making New Results Directory: " + curdir)
        os.makedirs(curdir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: %s' % device)
    ### Train/Load Model ######################################################################################################
    
    model = getattr(sys.modules[__name__], args.network)(**args.__dict__).to(device)

    if args.train:
        print("About To Train a Model. Oh man ! ")
        if args.dataset == "MNIST":
            _, _, _, tr_loader, _, _, _, val_loader, _, _, _, _ = load_mnist(args)
        else:
            print("need to add more datasets")
        model = train_network(model, args.epochs, tr_loader, val_loader, args.lr, args.optimizer, args.milestones, args.gamma, curdir)
    else:
        print("Loading Model From Last Epoch of" + str(args.load_model_name))
        model.load_state_dict(torch.load(curdir + "/model.pth"))

    ######## Adverserial Stuff #######################################################################################
    model.eval() 

    #### Generates pairs of image and aversarial for each class
    #### Output: AdvDict dictionaryl
    #### For any class c AdvDict[c] returns sub-dictionary
    #### AdvDict[c]['untargeted'] returns torch tensor (2, args.num_attack_untarg, input_dim) where num_attack_untarg is how many images you want to adv untarg attack. 
    #### 1st image of the 2 is orig image second image is the untargeted attacked version
    #### For any class d != c AdvDict[c][d] gives torch tensor (2, args.num_attack_untarg, input_dim) where again this is list of pair of images, 1st is an image of class d, second is after it was attacked to yield image of class c 
    if args.dataset == "MNIST":
        tr_data_im, tr_data_lab, _, tr_loader, _, _, _, _, te_data_im, te_data_lab, _, _ = load_mnist(args) 
    AdvDict = generate_adverserial_for_class_wrapper(args, model, tr_data_im, tr_loader)

    ###### Showing Dictionary Has Desired Properties
    print(model(AdvDict[0]['untargeted'][0,:].to(device)).argmax(dim = 1)) ##### should be class 0, correctly classified 
    print(model(AdvDict[0]['untargeted'][1,:].to(device)).argmax(dim = 1)) ##### should be random incorrect classes from untargeted attack

    print(model(AdvDict[4][2][0,:].to(device)).argmax(dim = 1)) ##### should be class 2, correctly classified 
    print(model(AdvDict[4][2][1,:].to(device)).argmax(dim = 1)) ##### should be class 4 from the targeted attack

    print("done")
    
    # ToDo 
    # test_point, test_label = ...    
    # boundary_points =  get_boundary_samples_for_point(test_point, label, AdvDict):
