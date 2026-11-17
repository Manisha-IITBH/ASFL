"""
Efficient Split Learning on CIFAR/FMNIST/DR/ISIC-2019 dataset.
"""

"""
IMPORTS
"""

# inbuilt
import os
import sys
import random
from math import ceil
import string
import requests, threading, time, socket, datetime
import multiprocessing
import copy
import importlib
import gc
from pathlib import Path
import hashlib

import pickle

# usual
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmetrics.functional import f1_score


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# discriminator model
from .models.discriminator import Discriminator


#plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# PFSL
from utils.random_clients_generator import generate_random_clients
from utils.connections import send_object
from utils.argparser import parse_arguments
from utils.merge import merge_weights
# from ImageClassification_Task.cifarbuilder import CIFAR10DataBuilder
from ImageClassification_Task.cifar_contrastivebuilder import CIFAR10_ContrastiveDataBuilder
# from ImageClassification_Task.ic_client import Client
from ImageClassification_Task.ic_client_combinedloss import Client_Contrastive
# from ImageClassification_Task.ic_server import ConnectedClient
from ImageClassification_Task.ic_server_combinedloss import ConnectedClient_Contrastive

from config import WANDB_KEY

# flags
os.environ['CUDA_LAUNCH_BLOCKING']='1'



"""
ImageClassification IC CLASS
"""
def hash_model(model):
    # clone = copy.deepcopy()
    model_bytes = pickle.dumps(model.state_dict())  # Save the model to a byte buffer
    return hashlib.sha256(model_bytes).hexdigest()

class ICTrainer_CombinedLoss_IP:

    def seed(self):
        """seed everything along with cuDNN"""
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed) 
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    def init_clients_with_data(self,):
        """
        initialize PFSL clients: (id, class: Client)
        along with their individual data based on Flamby splits
        """
        assert self.args.number_of_clients <= 10, 'max clients for isic is 6'
        self.num_clients = self.args.number_of_clients if not self.pooling_mode else 1
        self.clients = generate_random_clients(self.num_clients,Client_Contrastive)
        if self.pooling_mode:
            key = list(self.clients.keys())[0]
            self.clients['pooled_client'] = self.clients.pop(key) 
        self.client_ids = list(self.clients.keys())
        
        self.clients_threshold = dict()
        for c_id, _ in self.clients.items():
            self.clients_threshold[c_id]=0
        
        # -------------- for individual personalization of clients -------------
        self.best_client_per_val_accs = dict()
        self.client_per_early_stop_counter = dict()
        self.best_client_per_test_accs = dict() # it has test acc of epoch when the client performs best on its test dataset
        self.client_per_train_accs = dict()
        self.best_client_per_train_accs = dict() # it has train acc of epoch when the client performs best on its validation dataset
        self.client_per_early_stop = dict()
        for c_id, _ in self.clients.items():
            self.best_client_per_val_accs[c_id]=0
            self.client_per_early_stop_counter[c_id]=0
            self.best_client_per_test_accs[c_id]=0
            self.client_per_early_stop[c_id]=False
        #-----------------------------------------------------------------------
        
        data = []
        for idx, (c_id, client) in enumerate(self.clients.items()):
            train_ds, test_ds, main_test_ds = self.cifar_contrastive_builder.get_datasets(client_id=idx, alpha=0.3, num_clients=len(self.clients), ood=self.args.ood, seed=self.args.seed, pool=self.pooling_mode)
            client.train_dataset = train_ds
            client.test_dataset = test_ds
            client.main_test_dataset = main_test_ds
            print(f"client {c_id} -> #train {len(train_ds)} #valid: {len(test_ds)} #test: {len(main_test_ds)}")
            client.create_DataLoader(
                self.train_batch_size,
                self.test_batch_size
            )
            try:
                train_class_counts = Counter(item['label'].item() for item in train_ds)
                test_class_counts = Counter(item['label'].item() for item in test_ds)
                main_test_class_counts = Counter(item['label'].item() for item in main_test_ds)
            except KeyError as e:
                print(f"KeyError: {e} not found in dataset items. Inspecting item structure for alternative keys or attributes...")
                raise
            
            # Calculate class proportions
            def calculate_proportions(class_counts):
                total = sum(class_counts.values())
                return {k: v / total for k, v in class_counts.items()}

            train_class_proportions = calculate_proportions(train_class_counts)
            test_class_proportions = calculate_proportions(test_class_counts)
            main_test_class_proportions = calculate_proportions(main_test_class_counts)
            
            # print(f"Client {c_id} class distribution (proportions):")
            # print(f"  Train: {train_class_proportions}")
            # print(f"  Test: {test_class_proportions}")
            # print(f"  Main Test: {main_test_class_proportions}")

            # Collect data for plotting
            for class_label, freq in train_class_counts.items():
                data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'train'})
            #for class_label, freq in test_class_counts.items():
            #    data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'test'})
            #for class_label, freq in main_test_class_counts.items():
            #    data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'main_test'})
            
            print(f"Client {c_id} class distribution:")
            print(f"  Train: {train_class_counts}")
            print(f"  Test: {test_class_counts}")
            print(f"  Main Test: {main_test_class_counts}")
        print(f'generated {self.num_clients} clients with data')
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        # Plotting
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=df, x='client', y='class', size='frequency', hue='dataset', sizes=(20, 200), alpha=0.6, palette='muted')
        #sns.scatterplot(data=df, x='client', y='class')
        plt.title('Class Distribution per Client')
        plt.xlabel('Client')
        plt.ylabel('Class')
        plt.legend(title='Dataset')
        plt.grid(True)
        plt.savefig('class_distribution_plot.png')
        plt.show()


    def init_client_models_optims(self, input_channels=1):
        """
        Initialize client-side model splits: front, back, center.
        Splits are made from Resnet50/18, effnetv2_small,
        and optimizer for each split: Adam / AdamW / Novograd.
        """
        try:
            model = importlib.import_module(self.import_module)
        except ImportError as e:
            print(f"Error importing module: {e}")
            return
        
        pretrained = self.args.pretrained
        lr = self.args.client_lr

        for c_id, client in self.clients.items():
            client.device = self.device

            try:
                client.front_model = model.front().to(self.device)
                client.front_model.eval()
            except AttributeError as e:
                print(f"Error initializing front model for client {c_id}: {e}")
                continue

            try:
                client.back_model = model.back().to(self.device)
                #client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr,weight_decay=1e-5)
                client.back_optimizer = Adam(client.back_model.parameters(), lr=lr) #same as previous one
                #client.back_scheduler = CosineAnnealingWarmRestarts(
                #    client.back_optimizer,
                #    T_0=20,
                #    T_mult=1,
                #    eta_min=1e-10
                #)
            except AttributeError as e:
                print(f"Error initializing back model or optimizer for client {c_id}: {e}")
                continue

        print(f'Initialized client-side model splits front & back and their optimizers')
        
        
    def init_clients_server_copy(self):
        """
        For each client, there is a server copy of the center model
        initialized using class: ConnectedClient(client_id, connection).
        The center model and its optimizer are initialized as well.
        """
        try:
            model = importlib.import_module(self.import_module)
        except ImportError as e:
            print(f"Error importing module: {e}")
            return
        
        pretrained = self.args.pretrained
        lr = self.args.server_lr

        self.sc_clients = dict()
        for c_id in self.client_ids:
            self.sc_clients[c_id] = ConnectedClient_Contrastive(id=c_id, conn=None)

        for c_id, sc_client in self.sc_clients.items():
            sc_client.device = self.device

            try:
                sc_client.center_front_model = model.center_front().to(self.device)
                sc_client.center_front_model.eval()
            except AttributeError as e:
                print(f"Error initializing center front model for server copy of client {c_id}: {e}")
                continue
            
            try:
                discriminator = Discriminator().to(self.device)
                sc_client.discriminator = discriminator
                sc_client.discriminator_loss_fn = nn.MSELoss()
                sc_client.discriminator_optimizer = Adam(sc_client.discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))
            except AttributeError as e:
                print(f"Error initializing discriminator model for server copy of client {c_id}: {e}")
            
            try:
                sc_client.center_back_model = model.center_back().to(self.device)
                #s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)
                sc_client.center_optimizer = Adam(sc_client.center_back_model.parameters(), lr=lr)
                #sc_client.center_scheduler = CosineAnnealingWarmRestarts(
                #    sc_client.center_optimizer,
                #    T_0=20,
                #    T_mult=1,
                #    eta_min=1e-10
                #)
            except AttributeError as e:
                print(f"Error initializing center back model or optimizer for server copy of client {c_id}: {e}")
                continue

        print(f'Initialized server-side model splits center_front, center_back, and center_back optimizer')
    
    
    def _create_save_dir(self):
        """
        Create the directory for saving models. The directory is created based on
        the dataset name and model split specified in the arguments. If the directory
        already exists, it won't be recreated.

        Raises:
            OSError: If there is an error creating the directory.
        """
        try:
            self.save_dir = Path(f'./saved_models/{self.args.dataset}/key_value_mode/model_split{self.args.split}/{self.args.exp_name}')
            self.save_dir.mkdir(exist_ok=True, parents=True)
            print(f"Directory created at {self.save_dir}")
        except OSError as e:
            print(f"Error creating directory {self.save_dir}: {e}")
            raise

    
    def remove_frozen_models(self,):
        """
        - the client-side forward model and the server-side center_front model are unused after
        the key-value store mappings are generated.
        - the models are rather saved to disk and moved to CPU during runtime to save GPU
        """
        for c_id in self.client_ids:
            # client-side front model
            front_state_dict = self.clients[c_id].front_model.state_dict()
            torch.save(front_state_dict, self.save_dir / f'client_{c_id}_front.pth')
            self.clients[c_id].front_model.cpu()
            # server-side center_front model
            center_front_state_dict = self.sc_clients[c_id].center_front_model.state_dict()
            torch.save(center_front_state_dict, self.save_dir / f'client_{c_id}_center_front.pth')
            self.sc_clients[c_id].center_front_model.cpu()

    def personalize(self,epoch):
        """
        personalization: 
            freeze all layers of center model in server copy clients
        """
        print(f'personalizing, freezing server copy center model @ epoch {epoch}')
        for c_id, sc_client in self.sc_clients.items():
            sc_client.center_back_model.freeze(epoch,pretrained=self.args.pretrained)
            
    def merge_model_weights(self, epoch):
        """
        Merge weights and distribute over all server-side center_back models.
        In the personalization phase, merging of weights of the back model layers is stopped:
            - Merge weights and distribute over all client-side back models if not in personalization mode.

        Args:
            epoch (int): The current epoch during which merging is applied.
        """
        print(f'Merging model weights at epoch {epoch}')

        params = []
        sample_lens = []

        # Collect the state dictionaries and sample lengths for all server-side center_back models
        for c_id, sc_client in self.sc_clients.items():
            try:
                params.append(copy.deepcopy(sc_client.center_back_model.state_dict()))
                sample_lens.append(len(self.clients[c_id].train_dataset) * self.args.kv_factor)
            except Exception as e:
                print(f"Error collecting weights for server copy client {c_id}: {e}")
                continue

        try:
            # Merge weights using a custom utility function
            w_glob = merge_weights(params, sample_lens)

            # Distribute the merged weights to all server-side center_back models
            for c_id, sc_client in self.sc_clients.items():
                sc_client.center_back_model.load_state_dict(w_glob)
                #print(f"Merged weights loaded to server copy client {c_id}")
        except Exception as e:
            print(f"Error merging or distributing server-side weights: {e}")

        if True:
            params = []
            sample_lens = []

            # Collect the state dictionaries and sample lengths for all client-side back models
            for c_id, client in self.clients.items():
                try:
                    params.append(copy.deepcopy(client.back_model.state_dict()))
                    sample_lens.append(len(client.train_dataset))
                except Exception as e:
                    print(f"Error collecting weights for client {c_id}: {e}")
                    continue

            try:
                # Merge weights using a custom utility function
                w_glob_cb = merge_weights(params, sample_lens)
        
                # Distribute the merged weights to all client-side back models
                for c_id, client in self.clients.items():
                    client.back_model.load_state_dict(w_glob_cb)
                    #print(f"Merged weights loaded to client {c_id}")
            except Exception as e:
                print(f"Error merging or distributing client-side weights: {e}")

        # Clean up to free memory
        del params, sample_lens

    
    def create_iters(self, dl='train'):
        """
        - Append 0 for train_f1/test_f1 per client list.
        - Assign iterators per client from train/test dataloader.
        """
        num_iters = {c_id: 0 for c_id in self.client_ids}
        for c_id, client in self.clients.items():
            if dl == 'train':
                client.train_f1.append(0)
                client.flag = False
            elif dl == 'test':
                client.test_f1.append(0)
                #client.test_iterator = iter(client.test_DataLoader)
                #client.num_test_iterations = len(client.test_DataLoader)
                #num_iters[c_id] = len(client.test_DataLoader)
        print(f"Created list to store f1 score for {dl} data.")
        return num_iters
    

    def store_forward_mappings_kv(self,mode='train'):
        """
        kv: key-value store {data_key:np.Array}\n
        since client-side front model & server-side center-front model is "frozen"
        - we only need the outputs from both the models once
        - the outputs are stored in activation mappings, each output with its own index
        - the targets are stored in target mappings, each target with its own index
        
        these values are reused every epoch / refreshed once in a while, this is done for both train and test mappings
        """    
        if mode=='train':
            # create iterators for initial forward pass of training phase
            for c_id, client in self.clients.items():
                client.kv_flag=1
                self.sc_clients[c_id].kv_flag=1
                client.num_iterations = len(client.train_DataLoader)
                client.iterator = iter(client.train_DataLoader)
            # forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                #print(c_id,client.num_iterations)
                for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="client front"):
                    client.forward_front_key_value_contrastive()
                    self.sc_clients[c_id].remote_activations1 = client.remote_activations1
                    self.sc_clients[c_id].batchkeys = client.key
                    self.sc_clients[c_id].forward_center_front_contrastive()
                print(f"Training Set Key Value Store Created for Client {c_id}")
                print("Training Set Key Value Store Length is :", len((list(self.sc_clients[c_id].activation_mappings.keys()))))
                client.kv_flag=0
                self.sc_clients[c_id].kv_flag=0
        else:
            # create iterators for initial forward pass of testing phase
            for c_id, client in self.clients.items():
                client.kv_test_flag=1
                self.sc_clients[c_id].kv_test_flag=1
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                
            # forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                #print(c_id,client.num_test_iterations)
                for it in tqdm(range(client.num_test_iterations * self.args.kv_factor),desc="server front"):
                    #if client.data_key % len(client.train_dataset) == 0:
                    client.forward_front_key_value_test_contrastive()
                    self.sc_clients[c_id].remote_activations1 = client.remote_activations1
                    self.sc_clients[c_id].test_batchkeys = client.test_key
                    self.sc_clients[c_id].forward_center_front_test_contrastive()
                print(f"Validation Set Key Value Store Created for Client {c_id}")
                print("Validation Set Key Value Store Length is :", len(list(self.sc_clients[c_id].test_activation_mappings.keys())))
                client.kv_test_flag=0
                self.sc_clients[c_id].kv_test_flag=0

    def populate_key_value_store(self,):
        """
        - resets key-value store for client and server
        - populates key-value store kv_factor no. of times
        """
        print('generating training samples in key-value store...')
        self.store_forward_mappings_kv(mode='train')
        print('generating testing samples in key-value store...')
        self.store_forward_mappings_kv(mode='test')


    def train_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - step back optimizer
                - merge grads
                - step center_back optimizer
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
            - merge model weights across clients (center_back & back)
        """

        print(f"\n\nGeneralisation Phase Training {epoch}..........................................................................................")
        num_iters = self.create_iters(dl='train')
        self.overall_f1['train'].append(0)
        self.overall_acc['train'].append(0)
                
        for client_id, client in tqdm(self.clients.items()):
                client.iterator = iter(client.train_DataLoader)
                client.num_iterations = len(client.train_DataLoader)
                #for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="Training"):
                for iteration in tqdm(range(client.num_iterations),desc="Generalization Phase Training"):
                    client.forward_front_key_value_contrastive()
                    self.sc_clients[client_id].batchkeys = client.key
                    
                    self.sc_clients[client_id].forward_center_front_contrastive()
                    self.sc_clients[client_id].forward_center_back()
                    client.remote_activations2 = self.sc_clients[client_id].remote_activations2
                    
                    client.forward_back_contrastive()
                    client.calculate_loss_contrastive(mode='train')
                    
                    wandb.log({'train step loss': client.loss.item()})
                    
                    client.loss.backward()
                    
                    self.sc_clients[client_id].remote_activations2 = client.remote_activations2
                    self.sc_clients[client_id].backward_center()
                    
                    client.step_back()
                    #client.back_scheduler.step()
                    client.zero_grad_back()
                    
                    self.sc_clients[client_id].center_optimizer.step()
                    self.sc_clients[client_id].center_optimizer.zero_grad()
                    
                    f1=client.calculate_train_metric()
                    client.train_f1[-1] += f1 
                    
                    #print("train f1 per iteration: ",iteration,f1)
                    wandb.log({f'train f1 / iter: client {client_id}':f1.item()})

        # calculate per epoch metrics
        bal_accs, f1_macros = [], []
        avg_loss = 0
        for c_id, client in self.clients.items():
            client.train_f1[-1] /= client.num_iterations
            client.train_loss /= client.num_iterations
            avg_loss += client.train_loss
            self.overall_f1['train'][-1] += client.train_f1[-1]
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='train') 
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'train accuracy {c_id}':bal_acc_client})
            wandb.log({f'tarin f1 macro {c_id}':f1_macro_client})
            wandb.log({f'train loss {c_id}': client.train_loss})
            client.train_loss = 0 # reset for next epoch


        # calculate per epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['train'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['train'][-1] /= self.num_clients
        print("avg train f1 all clients: ", self.overall_f1['train'][-1].item())
        print("avg train accuracy all clients: ", self.overall_acc['train'][-1])
        wandb.log({'avg train f1 all clients': self.overall_f1['train'][-1].item()})
        wandb.log({'avg train bal acc all clients': bal_acc})
        wandb.log({'avg train f1 macro all clients': f1_macro})
        wandb.log({'avg train loss all clients': avg_loss / self.num_clients})
        self.merge_model_weights(epoch)
        
        
        
        #if not self.pooling_mode:
            # merge model weights (center and back)
            #print()
            #self.merge_model_weights(epoch)
    
    def train_one_epoch_discriminator(self, epoch):
        '''
        In this epoch:
            -for every batch of data available:
                - forward the self.middle_activations to the auto encoder for training
                - back propagate the loss gradient
                - step the optimizer
                - calculate the loss metric
        '''
        print(f"\n\n Discriminator Phase Training {epoch}..........................................................................................")
        
        self.disc_batchwise_losses=dict()

        for client_id, client in tqdm(self.clients.items()):
            client.iterator = iter(client.train_DataLoader)
            client.num_iterations = len(client.train_DataLoader)
            self.disc_batchwise_losses[client_id]=[]

            for iteration in tqdm(range(client.num_iterations), desc="Discriminator Phase Training"):
                client.forward_front_key_value_contrastive()
                
                self.sc_clients[client_id].batchkeys=client.key
                
                self.sc_clients[client_id].forward_center_front_contrastive()
                self.sc_clients[client_id].forward_center_back()
                self.sc_clients[client_id].forward_discriminator_contrastive()
                self.sc_clients[client_id].calculate_discriminator_loss_contrastive(mode="train")
                wandb.log({'discriminator step loss': self.sc_clients[client_id].disc_loss.item()})

                # store the batch wise train loss values
                self.disc_batchwise_losses[client_id].append(self.sc_clients[client_id].disc_loss.item())

                self.sc_clients[client_id].disc_loss.backward()
                self.sc_clients[client_id].discriminator_step()
                self.sc_clients[client_id].zero_grad_back()
            
        avg_loss = 0
            
        for c_id, client in self.clients.items():
            self.sc_clients[c_id].discriminator_train_loss /= client.num_iterations
            avg_loss+=self.sc_clients[c_id].discriminator_train_loss
            # wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'discriminator train loss of discriminator for {c_id}': self.sc_clients[c_id].discriminator_train_loss})
            self.clients_threshold[c_id]=self.sc_clients[c_id].discriminator_train_loss
            self.sc_clients[c_id].discriminator_train_loss = 0

        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Thresholds(based on training data) : {self.clients_threshold}')  
        wandb.log({f'avg discriminator train loss of clients': avg_loss / self.num_clients})       
    
    def train_one_epoch_personalise(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - step back optimizer
                - merge grads
                - step center_back optimizer
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
            - merge model weights across clients (center_back & back)
        """
        print(f" \n\n Personalisation Phase Training {epoch}.........................................................................................")                
        self.overall_f1_personalize['train'].append(0)
        self.overall_acc_personalize['train'].append(0)
        
        for client_id, client in tqdm(self.clients.items()):
            if self.client_per_early_stop[client_id]!=True:
                client.iterator = iter(client.train_DataLoader)
                client.num_iterations = len(client.train_DataLoader)
                #for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="Training"):
                for iteration in tqdm(range(client.num_iterations),desc="Personlaisation Phase Training"):
                    client.forward_back_personalise_contrastive()
                    client.calculate_loss_contrastive(mode='train')
                    client.loss.backward()
                    wandb.log({'train step loss': client.loss.item()})
                    client.step_back()
                    client.zero_grad_back()
                    #client.loss.backward()
                    f1=client.calculate_train_metric()
                    client.train_f1[-1] += f1
                    #print("train f1 per iteration: ",iteration,f1)
                    wandb.log({f'train f1 / iter: client {client_id}':f1.item()})

        # calculate per epoch metrics
        bal_accs, f1_macros = [], []
        avg_loss = 0
        active_clients_train_acc=dict()
        
        for c_id, client in self.clients.items():
            if self.client_per_early_stop[c_id]!=True:
                client.train_f1[-1] /= client.num_iterations
                client.train_loss /= client.num_iterations
                avg_loss += client.train_loss
                self.overall_f1_personalize['train'][-1] += client.train_f1[-1]
                bal_acc_client, f1_macro_client = client.get_main_metric(mode='train') 
                bal_accs.append(bal_acc_client)
                active_clients_train_acc[c_id]=bal_acc_client
                self.client_per_train_accs[c_id]=bal_acc_client
                f1_macros.append(f1_macro_client)
                wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
                wandb.log({f'train accuracy {c_id}':bal_acc_client})
                wandb.log({f'tarin f1 macro {c_id}':f1_macro_client})
                wandb.log({f'train loss {c_id}': client.train_loss})
                client.train_loss = 0 # reset for next epoch


        # calculate per epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc_personalize['train'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1_personalize['train'][-1] /= self.num_clients
        print("avg train f1 of active clients: ", self.overall_f1_personalize['train'][-1].item())
        print("avg train accuracy of active clients: ", self.overall_acc_personalize['train'][-1])
        print("Active Clients train acc:", active_clients_train_acc)
        
        # wandb.log({'avg train f1 all clients': self.overall_f1_personalize['train'][-1].item()})
        # wandb.log({'avg train bal acc all clients': bal_acc})
        # wandb.log({'avg train f1 macro all clients': f1_macro})
        # wandb.log({'avg train loss all clients': avg_loss / self.num_clients})
        
    
    @torch.no_grad()
    def test_one_epoch_personalise(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
        """
        self.overall_f1_personalize['test'].append(0)
        self.overall_acc_personalize['test'].append(0)
        
        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        for client_id, client in tqdm(self.clients.items()):
            if self.client_per_early_stop[client_id]!=True:
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Personalised Validation"):
                    client.forward_back_personalise_test_contrastive()
                    client.calculate_loss_contrastive(mode='test')
                    wandb.log({'Validation step loss': client.loss.item()})
                    f1=client.calculate_test_metric()
                    client.test_f1[-1] += f1
                    wandb.log({f'Validation f1 / iter: client {client_id}':f1.item()})
                    
        # calculate per epoch metrics
        avg_loss = 0
        bal_accs,f1_macros = [], []
        # save each client current validation acc
        client_val_accs = dict()
        
        for c_id, client in self.clients.items():
            if self.client_per_early_stop[c_id]!=True:
                client.test_f1[-1] /= len(client.test_DataLoader)
                client.test_loss /= len(client.test_DataLoader)
                # calculate remaining metrics
                avg_loss += client.test_loss
                bal_acc_client, f1_macro_client = client.get_main_metric(mode='test')
                bal_accs.append(bal_acc_client)
                client_val_accs[c_id]=bal_acc_client
                f1_macros.append(f1_macro_client)
                self.overall_f1_personalize['test'][-1] += client.test_f1[-1]
                wandb.log({f'Validation f1 {c_id}': client.test_f1[-1].item()})
                wandb.log({f'Validation accuracy {c_id}':bal_acc_client})
                wandb.log({f'Validation macro f1 {c_id}':f1_macro_client})
                wandb.log({f'Validation loss {c_id}': client.test_loss})
                client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc_personalize['test'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1_personalize['test'][-1] /= self.num_clients
        print("active clients val acc:", client_val_accs)
        print("active clients validation f1: ", self.overall_f1_personalize['test'][-1])
        print("active validation acc: ", self.overall_acc_personalize['test'][-1])
        
        for c_id in self.client_ids:
            if self.client_per_early_stop[c_id]!=True:
                if client_val_accs[c_id]>self.best_client_per_val_accs[c_id]:
                    self.best_client_per_val_accs[c_id]=client_val_accs[c_id]
                    self.client_per_early_stop_counter[c_id]=0
                    print(f"MAX Personalisation Validation Accuracy Score: {self.best_client_per_val_accs[c_id]} @ epoch {self.best_epoch}")
                    # storing the train acc when we got the best val acc
                    self.best_client_per_train_accs[c_id]=self.client_per_train_accs[c_id]
                else:
                    self.client_per_early_stop_counter[c_id]+=1

    @torch.no_grad()
    def test_one_epoch_disc(self, epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
        """

        # num_iters = self.create_iters(dl='test')
        #self.overall_f1['test'].append(0)
        #self.overall_acc['test'].append(0)

        #for c_id, client in self.clients.items():
        #    client.pred = []
        #    client.y = []

        for client_id, client in tqdm(self.clients.items()):
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Validation"):
                    client.forward_front_key_value_test_contrastive()
                    self.sc_clients[client_id].test_batchkeys = client.test_key
                    self.sc_clients[client_id].forward_center_front_test_contrastive()
                    self.sc_clients[client_id].forward_center_back()
                    self.sc_clients[client_id].forward_discriminator_test_contrastive()
                    self.sc_clients[client_id].calculate_discriminator_loss_contrastive(mode="test")
                    
                    wandb.log({'discriminator Validation step loss': self.sc_clients[client_id].disc_loss.item()})

                    
        # calculate per epoch metrics
        avg_disc_test_loss = 0
        for c_id, client in self.clients.items():
            self.sc_clients[c_id].discriminator_test_loss /= len(client.test_DataLoader)
            avg_disc_test_loss += self.sc_clients[c_id].discriminator_test_loss
            wandb.log({f'Validation loss of {c_id} discriminator': self.sc_clients[c_id].discriminator_test_loss})
            # self.clients_threshold[c_id]=self.sc_clients[c_id].discriminator_test_loss
            self.sc_clients[c_id].discriminator_test_loss=0
            
        # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Thresholds : {self.clients_threshold}')
        avg_disc_test_loss = avg_disc_test_loss/self.num_clients
        wandb.log({'Discriminator validation avg loss of all clients': avg_disc_test_loss / self.num_clients})
        if avg_disc_test_loss < self.best_disc_loss:
            self.best_disc_loss = avg_disc_test_loss
            self.best_disc_epoch = epoch
            self.early_stop_counter_disc = 0
            print(f"MIN VALIDATION LOSS OF DISCRIMINATOR: {self.best_disc_loss} @ epoch {self.best_disc_epoch}")
            wandb.log({
                'min_test_loss_epoch':self.best_disc_loss
            })
            return True
        else:
            self.early_stop_counter_disc += 1
            return False
    
    @torch.no_grad()
    def test_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
        """

        num_iters = self.create_iters(dl='test')
        self.overall_f1['test'].append(0)
        self.overall_acc['test'].append(0)

        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        for client_id, client in tqdm(self.clients.items()):
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Validation"):
                    client.forward_front_key_value_test_contrastive()
                    self.sc_clients[client_id].test_batchkeys = client.test_key
                    self.sc_clients[client_id].forward_center_front_test_contrastive()
                    self.sc_clients[client_id].forward_center_back()
                    client.remote_activations2 = self.sc_clients[client_id].remote_activations2
                    client.forward_back_contrastive()
                    client.calculate_loss_contrastive(mode='test')
                    wandb.log({'Validation step loss': client.loss.item()})
                    f1=client.calculate_test_metric()
                    client.test_f1[-1] += f1 
                    #print("validation f1 per iteration: ",iteration,f1)
                    wandb.log({f'Validation f1 / iter: client {client_id}':f1.item()})
                    
        # calculate per epoch metrics
        avg_loss = 0
        bal_accs,f1_macros = [], []
        for c_id, client in self.clients.items():
            client.test_f1[-1] /= len(client.test_DataLoader)
            client.test_loss /= len(client.test_DataLoader)
            # calculate remaining metrics
            avg_loss += client.test_loss
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='test')
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            self.overall_f1['test'][-1] += client.test_f1[-1]
            wandb.log({f'Validation f1 {c_id}': client.test_f1[-1].item()})
            wandb.log({f'Validation accuracy {c_id}':bal_acc_client})
            wandb.log({f'Validation macro f1 {c_id}':f1_macro_client})
            wandb.log({f'Validation loss {c_id}': client.test_loss})
            client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['test'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['test'][-1] /= self.num_clients
        print("validation f1: ", self.overall_f1['test'][-1])
        print("validation acc: ", self.overall_acc['test'][-1])
        wandb.log({'Validation avg f1 all clients': self.overall_f1['test'][-1].item()})
        wandb.log({'validation avg accuracy all clients': bal_acc})
        wandb.log({'Validation avg f1 macro all clients': bal_acc})
        wandb.log({'Validation avg loss all clients': avg_loss / self.num_clients}) 
        if self.overall_acc['test'][-1] > self.best_acc:
            print(self.best_acc)
            self.best_acc = self.overall_acc['test'][-1]
            self.best_epoch = epoch
            self.early_stop_counter = 0
            print(f"MAX Validation Accuracy Score: {self.best_acc} @ epoch {self.best_epoch}")
            wandb.log({
                'max validation accuracy score':self.best_acc,
                'max_validation_accuarcy_epoch':self.best_epoch
            })
            return True
        else:
            self.early_stop_counter += 1
            return False
                    
    def save_models(self,epoch):
        """
        save client-side back and server-side center_back models to disk
        """
        print("Save Model at epoch", epoch)
        if self.personalization_mode ==False:
            print("Save Best Model for Generalisation Phase")
            for c_id in self.client_ids:
                # client-side front model
                front_state_dict = copy.deepcopy(self.clients[c_id].front_model.state_dict())
                torch.save(front_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_front.pth')
                # torch.save(self.clients[c_id].front_model, self.save_dir / f'client_{c_id}_{self.args.model}_front_model.pth')
                # server-side center_front model
                center_front_state_dict = copy.deepcopy(self.sc_clients[c_id].center_front_model.state_dict())
                torch.save(center_front_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_center_front.pth')
                # torch.save(self.sc_clients[c_id].center_front_model, self.save_dir / f'client_{c_id}_{self.args.model}_center_front_model.pth')
                # server-side center_back model
                center_back_state_dict = copy.deepcopy(self.sc_clients[c_id].center_back_model.state_dict())
                torch.save(center_back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_center_back.pth')
                # torch.save(self.sc_clients[c_id].center_back_model, self.save_dir / f'client_{c_id}_{self.args.model}_center_back_model.pth')
                # client-side back model
                back_state_dict = copy.deepcopy(self.clients[c_id].back_model.state_dict())
                torch.save(back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_back.pth')
                # torch.save(self.clients[c_id].back_model, self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth')
        else:
            for c_id in self.client_ids:
                print("Save Best Model for Personalisation Phase")
                # client-side back model
                back_state_dict = copy.deepcopy(self.clients[c_id].back_model.state_dict())
                torch.save(back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth')
                # torch.save(self.clients[c_id].back_model, self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth')
    
    def save_per_models(self, epoch, c_id):
        print(f"Save {c_id} best model for Personalisation Phase")
        # client-side back model
        back_state_dict = copy.deepcopy(self.clients[c_id].back_model.state_dict())
        torch.save(back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth')
     
    def load_best_models(self,):
        """
        replaces the latest models with the best models on server and client-side
        """
        print("Loaded Best Model")

        model = importlib.import_module(self.import_module)

        if self.personalization_mode==False:
            print("Load Best Model for Generalisation Phase")
            for c_id in self.client_ids:
                # front = model.front()
                # front_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front.pth')
                # front.load_state_dict(front_sd, weights_only=True)
                # front=front.to(self.device)
                # front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front_model.pth').to(self.device)
                # center_front = model.center_front()
                # center_front_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front.pth')
                # center_front.load_state_dict(center_front_sd, weights_only=True)
                # center_front=center_front.to(self.device)
                # center_front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front_model.pth').to(self.device)
                # center_back = model.center_back()
                # center_back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back.pth')
                # center_back.load_state_dict(center_back_sd, weights_only=True)
                # center_back=center_back.to(self.device)
                # center_back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back_model.pth').to(self.device)
                # back = model.back()
                # back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth')
                # back.load_state_dict(back_sd, weights_only=True)
                # back=back.to(self.device)
                # back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth').to(self.device)
                
                # self.clients[c_id].front_model = front
                # self.clients[c_id].back_model = back
                # self.sc_clients[c_id].center_front_model = center_front
                # self.sc_clients[c_id].center_back_model = center_back
                self.clients[c_id].front_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front.pth', weights_only=True, map_location=self.device))
                self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth', weights_only=True, map_location=self.device))
                self.sc_clients[c_id].center_front_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front.pth', weights_only=True, map_location=self.device))
                self.sc_clients[c_id].center_back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back.pth', weights_only=True, map_location=self.device))
                
                self.clients[c_id].front_model.eval()
                self.clients[c_id].back_model.eval()
                self.sc_clients[c_id].center_front_model.eval()
                self.sc_clients[c_id].center_back_model.eval()
        else:
            for c_id in self.client_ids:
                # back = model.back().to(self.device)
                # print("Load Best Model for Personalisation Phase")
                # back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth')
                # back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth').to(self.device)
                # back.load_state_dict(back_sd)
                # self.clients[c_id].back_model = back
                self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
                self.clients[c_id].back_model.eval()
                
    def load_best_per_models(self):
        for c_id in self.client_ids:
            if self.client_per_early_stop_counter==0:
                self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
                self.clients[c_id].back_model.eval()
                
    @torch.no_grad()
    def inference(self,):
        """
        run inference on the main test dataset
        """
        print("RUNNING INFERENCE from the best models on test dataset")
        
        # self.load_best_models()
        avg_acc=0
        for c_id in tqdm(self.client_ids,desc="Testing"):
            trues = []
            preds = []

            for batch in self.clients[c_id].main_test_DataLoader:
                image, label = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device)

                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)

                trues.append(label.cpu())
                preds.append(x4.cpu())

            preds = torch.cat(preds)
            targets = torch.cat(trues)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            #bacc = balanced_accuracy_score(targets, preds)
            wandb.log({
                'inference cfm': wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=targets,
                    class_names=[f'{i}' for i in range(10)]
                )
            })

            print(f'inference score {c_id}: {accuracy}')
            avg_acc+=accuracy
            wandb.log({f'inference score {c_id}': accuracy})
        print(f'Average inference score: {avg_acc/len(self.clients)}')

    @torch.no_grad()
    def inference_personalisation(self, c_id):
        """
        run inference on the main test dataset
        """
         
        # self.load_best_models()
        # for c_id in tqdm(self.client_ids,desc="Testing"):
        #     if self.client_per_early_stop_counter[c_id]==0:
        trues = []
        preds = []

        for batch in self.clients[c_id].main_test_DataLoader:
            image, label = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device)

            x1 = self.clients[c_id].front_model(image)
            x2 = self.sc_clients[c_id].center_front_model(x1)
            x3 = self.sc_clients[c_id].center_back_model(x2)
            x4, _ = self.clients[c_id].back_model(x3)

            trues.append(label.cpu())
            preds.append(x4.cpu())

        preds = torch.cat(preds)
        targets = torch.cat(trues)
        
        targets = targets.reshape(-1).numpy()
        preds = np.argmax(preds.numpy(), axis=1)
        correct = np.sum(preds == targets)
        total = len(targets)
        accuracy = correct / total

        self.best_client_per_test_accs[c_id]=accuracy
        print(f'personalisation inference score {c_id}: {accuracy}')
    
    @torch.no_grad()
    def inference_new(self,):
        '''
        run inference individually on every data point from main_test_dataset
        '''
        print('running inference_new on main_test_dataset')
        avg_acc=0
        # for c_id in tqdm(self.client_ids,desc="Testing_New"):
        
        # self.load_best_models()
        for idx, (c_id, client) in enumerate(self.clients.items()):
            trues=[]
            preds=[]
            generalized=[]
            personalized=[]
            c=0
            c1=0
            c2=0
            for data in client.main_test_dataset:
                c+=1
                self.sc_clients[c_id].discriminator.eval()
                image = data['image'].to(self.device)
                image = torch.unsqueeze(image, 0)
                label = data['label'].to(self.device)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                
                reconstruction = self.sc_clients[c_id].discriminator(x2)
                loss = self.sc_clients[c_id].discriminator_loss_fn(reconstruction, x2)
                
                if loss > self.clients_threshold[c_id]:
                    c1+=1
                    # print(f"{loss}>{self.clients_threshold[c_id]}")
                    generalized.append([image, label])
                else:
                    c2+=1
                    # print(f"less")
                    personalized.append([image, label])
            print(f'test images in  {c_id} : {c} ; {c1},{c2}')
            
            print(f"------------no of data points predicted ood in {c_id}: {len(generalized)}")
            self.clients[c_id].back_model = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth')
            for image, label in generalized:
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4 = self.clients[c_id].back_model(x3)
                
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            
            print(f"------------no of data points predicted id in {c_id}: {len(personalized)}")
            self.clients[c_id].back_model = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth')
            for image, label in personalized:
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4 = self.clients[c_id].back_model(x3)
                
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            trues= [torch.unsqueeze(t, 0) for t in trues]
            preds = torch.cat(preds)
            targets = torch.cat(trues)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.detach().numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            
            wandb.log({
                'inference cfm': wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=targets,
                    class_names=[f'{i}' for i in range(10)]
                )
            })
            
            avg_acc+=accuracy
            wandb.log({f'inference score {c_id}': accuracy})
        print(f'Average inference score: {avg_acc/len(self.clients)}')
        
    def calculate_score(self, x, c_id, num_classes):
        """
        Function to calculate scores for the batch.
        """
        temperature=1
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(self.device)
        inputs = Variable(x.to(self.device), requires_grad=True)
        #zero gradients
        self.sc_clients[c_id].center_back_model.zero_grad()
        self.clients[c_id].back_model.zero_grad()
        
        x1 = self.clients[c_id].front_model(x)
        x2 = self.sc_clients[c_id].center_front_model(x1)
        x3 = self.sc_clients[c_id].center_back_model(x2)
        outputs, _ = self.clients[c_id].back_model(x3)
        
        targets = torch.ones((inputs.shape[0]//2, num_classes)).to(self.device)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        layer_grad = self.clients[c_id].back_model.fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()

        return layer_grad_norm
    
    def iterate_traindata_gradnorm(self, data_loader, c_id, num_classes):
        confs = []
        temperature = 1
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(self.device)
        for batch in data_loader:
            inputs=torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device)
            # inputs = Variable(x.cuda(), requires_grad=True)
            #zero gradients
            # self.sc_clients[c_id].center_back_model.zero_grad()
            self.clients[c_id].back_model.zero_grad()
            # forward pass
            x1 = self.clients[c_id].front_model(inputs)
            x2 = self.sc_clients[c_id].center_front_model(x1)
            x3 = self.sc_clients[c_id].center_back_model(x2)
            outputs, _ = self.clients[c_id].back_model(x3)
            
            targets = torch.ones((inputs.shape[0]//2, num_classes)).to(self.device)
            outputs = outputs / temperature
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

            loss.backward()

            layer_grad = self.clients[c_id].back_model.fc.weight.grad.data

            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)

    def inference_hybrid_batchwise(self, num_classes):
        client_thresh={}
        for idx, (c_id, client) in enumerate(self.clients.items()):
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            train_ood_scores = self.iterate_traindata_gradnorm(client.thresh_DataLoader, c_id, num_classes)
            mean = np.mean(train_ood_scores)
            std = np.std(train_ood_scores)
            client_thresh[c_id]=mean-std
            print(f"Threshold mean of client {c_id}: {client_thresh[c_id]}, std:{std}")
        for idx, (c_id, client) in enumerate(self.clients.items()):
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            train_ood_scores = self.iterate_traindata_gradnorm(client.hybrid_test_DataLoader, c_id, num_classes)
            mean = np.mean(train_ood_scores)
            std = np.std(train_ood_scores)
            print(f"score stats of {c_id} on its test:- mean={mean}, std={std}")
            
        avg_acc=0
        for c_id in tqdm(self.client_ids,desc="Hybrid Testing Batchwise"):
        # for idx, (c_id, client) in enumerate(self.clients.items()):
            trues=[]
            preds=[]
            generalized=[]
            personalized=[]
            c=0
            c1=0
            c2=0
            
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            for batch in self.clients[c_id].hybrid_test_DataLoader:
                image, label = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device)
                
                score = self.calculate_score(image, c_id, num_classes)
                batch_size=image.shape[0]//2
                c+=batch_size
                if score > client_thresh[c_id]:
                    c2+=batch_size
                    personalized.append([image, label])
                else:
                    c1+=batch_size
                    generalized.append([image, label])
            
            print(f'test images in  {c_id} : {c} ; general-{c1}, personal-{c2}')
            
            print(f"------------no of data points predicted ood in {c_id}: {len(generalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for image, label in generalized:
                # print(image.shape, label.shape)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                # print(x4.shape)
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            print(f"------------no of data points predicted id in {c_id}: {len(personalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for image, label in personalized:
                # print(image.shape, label.shape)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                # print(x4.shape)
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            # trues= [torch.unsqueeze(t, 0) for t in trues]
            preds = torch.cat(preds)
            targets = torch.cat(trues)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.detach().numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            
            avg_acc+=accuracy
            print(f'inference score {c_id}:', accuracy)
        print(f'Hybrid Average inference score: {avg_acc/len(self.clients)}')
        
    @torch.no_grad()
    def inference_new_contrastive_samplewise(self,):
        '''
        run inference individually on every data point from main_test_dataset
        '''
        print('running inference_new on main_test_dataset')
        avg_acc=0
        # for c_id in tqdm(self.client_ids,desc="Testing_New"):
        
        # self.load_best_models()
        for c_id in tqdm(self.client_ids,desc="Hybrid Testing"):
        # for idx, (c_id, client) in enumerate(self.clients.items()):
            trues=[]
            preds=[]
            generalized=[]
            personalized=[]
            c=0
            c1=0
            c2=0
            for data in self.clients[c_id].main_test_dataset:
                c+=1
                self.sc_clients[c_id].discriminator.eval()
                image1 = torch.cat([torch.unsqueeze(data['image'][0],0), torch.unsqueeze(data['image'][1],0)], dim=0).to(self.device)
                image = data['image'][0].to(self.device)
                image = torch.unsqueeze(image, 0)
                label = data['label'].to(self.device)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                # x2_copy = x2.clone()
                # batch_size = x2_copy.shape[0] // 2
                # disc_inp, _ = torch.split(x2_copy, [batch_size, batch_size], dim=0) 
                reconstruction = self.sc_clients[c_id].discriminator(x3)
                loss = self.sc_clients[c_id].discriminator_loss_fn(reconstruction, x3)
                
                if loss > self.clients_threshold[c_id]:
                    c1+=1
                    # print(f"{loss}>{self.clients_threshold[c_id]}")
                    generalized.append([image1, label])
                else:
                    c2+=1
                    # print(f"less")
                    personalized.append([image1, label])
            print(f'test images in  {c_id} : {c} ; {c1},{c2}')
            
            print(f"------------no of data points predicted ood in {c_id}: {len(generalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for data in generalized:
                # print(data[0].shape, data[1].shape)
                x1 = self.clients[c_id].front_model(data[0])
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                
                trues.append(data[1].cpu())
                preds.append(x4.cpu())
            
            
            print(f"------------no of data points predicted id in {c_id}: {len(personalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for image, label in personalized:
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            trues= [torch.unsqueeze(t, 0) for t in trues]
            preds = torch.cat(preds)
            targets = torch.cat(trues)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.detach().numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            
            avg_acc+=accuracy
            print(f'inference score {c_id}:', accuracy)
        print(f'Average inference score: {avg_acc/len(self.clients)}')
    
    @torch.no_grad()
    def inference_new_contrastive_batchwise(self,):
        '''
        run inference individually on every data point from main_test_dataset
        '''
        print('running inference_new on main_test_dataset')
        avg_acc=0
        # for c_id in tqdm(self.client_ids,desc="Testing_New"):
        
        # self.load_best_models()
        for c_id in tqdm(self.client_ids,desc="Hybrid Testing Batchwise"):
        # for idx, (c_id, client) in enumerate(self.clients.items()):
            trues=[]
            preds=[]
            generalized=[]
            personalized=[]
            c=0
            c1=0
            c2=0
            for batch in self.clients[c_id].main_test_DataLoader:
                image, label = image, label = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device)

                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x3_copy = x3.clone()
                batch_size = x3_copy.shape[0] // 2
                disc_inp, _ = torch.split(x3_copy, [batch_size, batch_size], dim=0) 
                reconstruction = self.sc_clients[c_id].discriminator(disc_inp)
                loss = self.sc_clients[c_id].discriminator_loss_fn(reconstruction, disc_inp)
                
                c+=batch_size
                if loss > self.clients_threshold[c_id]:
                    c1+=batch_size
                    # print(f"{loss}>{self.clients_threshold[c_id]}")
                    generalized.append([image, label])
                else:
                    c2+=batch_size
                    # print(f"{loss}<{self.clients_threshold[c_id]}")
                    personalized.append([image, label])
                
            print(f'test images in  {c_id} : {c} ; {c1},{c2}')
            
            print(f"------------no of data points predicted ood in {c_id}: {len(generalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for image, label in generalized:
                # print(image.shape, label.shape)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                # print(x4.shape)
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            print(f"------------no of data points predicted id in {c_id}: {len(personalized)}")
            self.clients[c_id].back_model.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth', weights_only=True, map_location=self.device))
            self.clients[c_id].back_model.eval()
            
            for image, label in personalized:
                # print(image.shape, label.shape)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4, _ = self.clients[c_id].back_model(x3)
                # print(x4.shape)
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            # trues= [torch.unsqueeze(t, 0) for t in trues]
            preds = torch.cat(preds)
            targets = torch.cat(trues)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.detach().numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            
            avg_acc+=accuracy
            print(f'inference score {c_id}:', accuracy)
        print(f'Average inference score: {avg_acc/len(self.clients)}')

    def clear_cache(self,):
        gc.collect()
        torch.cuda.empty_cache()
        
    def save_kv(self,):
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Training"):
                for batch in self.clients[c_id].train_DataLoader:
                    image, label, batchkeys = batch['image'].to(self.device), batch['label'].to(self.device), batch['id']
                    #print(batchkeys)
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys = [key for key in batchkeys if key in self.sc_clients[c_id].activation_mappings]
                    #print(valid_keys)
                    activations_list = [self.sc_clients[c_id].activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys)):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[i]]=local_middle_activations[i]
                print(f"Training Set Key Value Store Created for Client {c_id}")
                print("Training Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys()))))
                
                
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Testing"):
                for batch in self.clients[c_id].test_DataLoader:
                    image, label, batchkeys = batch['image'].to(self.device), batch['label'].to(self.device), batch['id']
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys = [key for key in batchkeys if key in self.sc_clients[c_id].test_activation_mappings]
                    activations_list = [self.sc_clients[c_id].test_activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys)):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[i]]=local_middle_activations[i]
                print(f"Validation Set Key Value Store Created for Client {c_id}")
                print("Validation Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys())))) 

    def save_kv_contrastive(self,):
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Training"):
                for batch in self.clients[c_id].train_DataLoader:
                    image, label, batchkeys = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device), batch['id']
                    #print(batchkeys)
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys=[]
                    valid_keys.extend([key for key in batchkeys[0] if key in self.sc_clients[c_id].activation_mappings])
                    valid_keys.extend([key for key in batchkeys[1] if key in self.sc_clients[c_id].activation_mappings])
                    #print(valid_keys)
                    activations_list = [self.sc_clients[c_id].activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys[0])):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[0][i]]=local_middle_activations[i]
                        self.clients[c_id].activation_mappings[batchkeys[1][i]]=local_middle_activations[i]
                        # self.sc_clients[c_id].activation_mappings_serverend[batchkeys[0][i]]=local_middle_activations[i] # added to save the activations of the best server model to train the discriminator at the end of server
                        # self.sc_clients[c_id].activation_mappings_serverend[batchkeys[1][i]]=local_middle_activations[i] # added to save the activations of the best server model to train the discriminator at the end of server
                        
                print(f"Training Set Key Value Store Created for Client {c_id}")
                print("Training Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys()))))
            
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Testing"):
                for batch in self.clients[c_id].test_DataLoader:
                    image, label, batchkeys = torch.cat([batch['image'][0], batch['image'][1]], dim=0).to(self.device), batch['label'].to(self.device), batch['id']
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys=[]
                    valid_keys.extend([key for key in batchkeys[0] if key in self.sc_clients[c_id].test_activation_mappings])
                    valid_keys.extend([key for key in batchkeys[1] if key in self.sc_clients[c_id].test_activation_mappings])
                    activations_list = [self.sc_clients[c_id].test_activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys[0])):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[0][i]]=local_middle_activations[i]
                        self.clients[c_id].activation_mappings[batchkeys[1][i]]=local_middle_activations[i]
                print(f"Validation Set Key Value Store Created for Client {c_id}")
                print("Validation Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys())))) 

    def fit(self,):
        """
        - trains and tests the models for given num. of epochs
        """

        if self.pooling_mode:
            print('\n\nPOOLING MODE: ENABLED!')

        self._create_save_dir()
        # disabled freeing GPU mem since key-value store needs to be refreshed
        # print('freeing some GPU...')
        # self.remove_frozen_models()

        # if key value store refresh rate = 0, it is disabled
        if self.args.kv_refresh_rate == 0:
            self.populate_key_value_store()
            self.clear_cache()
        
        
        print(f'{"-"*25}\n\ncommence training...\n\n')

        for epoch in tqdm(range(self.args.epochs)):
            
            if self.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

            wandb.log({'epoch':epoch})

            # load the best models before starting of local epoch training among clients
            if epoch>0:
                self.load_best_models()

            for c_id in self.client_ids:
                self.clients[c_id].back_model.train()
                self.sc_clients[c_id].center_back_model.train()

            self.train_one_epoch(epoch)
            self.clear_cache()
        
            for c_id in self.client_ids:
                self.clients[c_id].back_model.eval()
                self.sc_clients[c_id].center_back_model.eval()
            
            should_save = self.test_one_epoch(epoch)
            
            print(should_save)
            if should_save:
                self.early_stop = False
                print("Model improved and saved.")
                
                self.save_models(epoch) 

                self.load_best_models()
                
                self.inference()
                #self.save_kv() # for personalisation phase
            else:
                if self.early_stop_counter == 5:
                    self.early_stop = True
    
            self.clear_cache()
        
        # final metrics of generalization phase  
        self.load_best_models()

        print(f'\n\n\n{"::"*10}BEST TEST PERFORMANCE{"::"*10}')
        self.inference()

        for run in range(100):
            if run>0:
                self.sc_clients[c_id].discriminator.load_state_dict(torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_discriminator.pth', weights_only=True, map_location=self.device))
            for c_id in self.client_ids:
                self.sc_clients[c_id].center_back_model.eval()
                self.sc_clients[c_id].discriminator.train()
            self.train_one_epoch_discriminator(run)
            for c_id in self.client_ids:
                self.sc_clients[c_id].discriminator.eval()
            r = self.test_one_epoch_disc(run)
            if r==True:
                for c_id in self.client_ids:
                    disc_state_dict = copy.deepcopy(self.sc_clients[c_id].discriminator.state_dict())
                    torch.save(disc_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_discriminator.pth')
            else:
                if self.early_stop_counter_disc == 5:
                    print(f"early stopping discriminator training at epoch-{run}")
                    break
        
        print("Batchwise_Discriminator_loss:")
        print(self.disc_batchwise_losses)
        print("clients thresholds:",self.clients_threshold)
        
        # --------------- for mean
        # for c_id in self.client_ids:
        #     self.clients_threshold[c_id]=sum(self.disc_batchwise_losses[c_id])/len(self.disc_batchwise_losses[c_id])
        # --------------- for median
        # for c_id in self.client_ids:
        #     self.clients_threshold[c_id]=statistics.median(self.disc_batchwise_losses[c_id])
        # --------------- for mean-k(std)
        for c_id in self.client_ids:
            m = statistics.mean(self.disc_batchwise_losses[c_id])
            # m = statistics.median(self.disc_batchwise_losses[c_id])
            std = statistics.pstdev(self.disc_batchwise_losses[c_id])
            k=1
            self.clients_threshold[c_id]=m-k*(std)
            print(c_id, m, std, self.clients_threshold[c_id])
        # --------------- using percentiles and iqr
        # for c_id in self.client_ids:
        #     q3, q1 = np.percentile(self.disc_batchwise_losses[c_id], [75, 25])
        #     iqr=q3-q1
        #     k=0.5
        #     self.clients_threshold[c_id]=q1+k*(iqr)
        #     print(c_id, q1, q3, iqr, self.clients_threshold[c_id])
        print("after clients thresholds:",self.clients_threshold)
        
        if self.personalize:
            self.personalization_mode = True
            print("Personalization Started")
            self.save_kv_contrastive()
            self.personalize(epoch)
            for epoch in tqdm(range(epoch,self.args.epochs)):
                
                count=0
                for c_id in self.client_ids:
                    if self.client_per_early_stop[c_id]==True:
                        count+=1
                if count==len(self.client_ids):
                    print("PERSONALISATION PHASE IS OVER.")
                    break
                    
                for c_id in self.client_ids:
                    self.clients[c_id].back_model.train()
                    self.sc_clients[c_id].center_back_model.eval()
                self.train_one_epoch_personalise(epoch)
                self.clear_cache()
                for c_id in self.client_ids:
                    self.clients[c_id].back_model.eval()
                self.test_one_epoch_personalise(epoch)
                self.clear_cache()
                
                for c_id in self.client_ids:
                    if self.client_per_early_stop_counter[c_id]==0:
                        self.save_per_models(epoch, c_id)
                        self.load_best_per_models()
                        self.inference_personalisation(c_id)
                    else:
                        if self.client_per_early_stop_counter[c_id]==5:
                            self.client_per_early_stop[c_id]=True
                
                print(f"Active personalisation clients-{[c_id for c_id in self.client_per_early_stop.keys() if self.client_per_early_stop[c_id]==False]} ; Stopped Clients-{[c_id for c_id in self.client_per_early_stop.keys() if self.client_per_early_stop[c_id]==True]}")
                print(f"Clients Patience values: {self.client_per_early_stop_counter}")
                print(f"Best Personalisation models train accs: {self.best_client_per_train_accs}")
                print(f"Best Personalisation models val accs: {self.best_client_per_val_accs}")
                print(f"Best Personalisation models test accs: {self.best_client_per_test_accs}")
                print(f"Best Personalisation models avg train acc: {sum(self.best_client_per_train_accs.values())/len(self.best_client_per_train_accs)}")
                print(f"Best Personalisation models avg val acc: {sum(self.best_client_per_val_accs.values())/len(self.best_client_per_val_accs)}")
                print(f"Best Personalisation models avg test acc: {sum(self.best_client_per_test_accs.values())/len(self.best_client_per_test_accs)}")
            
            print(f'\n\n\n{"::"*10}BEST PERSONALISATION TEST PERFORMANCE{"::"*10}')
            self.load_best_models()
            self.inference()

        print("**SAMPLEWISE HYBRID INFERENCE**")
        self.inference_new_contrastive_samplewise()
        print("**BATCHWISE HYBRID INFERENCE**")
        self.inference_new_contrastive_batchwise()
        print("**BATCHWISE HYBRID INFERENCE USING GRAD NORM**")
        self.inference_hybrid_batchwise(num_classes=10)
        # self.inference_new_contrastive_batchwise()
        
    def __init__(self,args):
        """
        implementation of PFSL training & testing simulation on ISIC-2019 dataset
            - ISIC-2019: Dermoscopy Image Classification
            - model: ResNet

        initialize everything:
            - data
            - wandb logging
            - metrics
            - device
            - batch sizes
            - flags
            - models
            - pooling mode: train all samples together
        """

        self.args = args
        self.log_wandb = self.args.wandb

        self.import_module = f"ImageClassification_Task.models.{self.args.model}_contrastive_split{self.args.split}"

        self.pooling_mode = self.args.pool

        # refresh key-value store every N epochs
        self.kv_refresh_rate = self.args.kv_refresh_rate

        wandb.login(key=WANDB_KEY)
        self.run = wandb.init(
            project='med-fsl_isic2019',
            config=vars(self.args),
            job_type='train',
            mode='online' if self.log_wandb else 'disabled'
        )

        self.seed()

        #self.isic = ISICDataBuilder()
        # self.cifar_builder = CIFAR10DataBuilder()
        self.cifar_contrastive_builder = CIFAR10_ContrastiveDataBuilder()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.overall_f1 = {
            'train': [],
            'test': []
        }
        
        self.overall_acc = {
            'train': [],
            'test': []
        }
        
        self.overall_f1_personalize = {
            'train': [],
            'test': []
        }
        
        self.overall_acc_personalize = {
            'train': [],
            'test': []
        }

        self.max_f1 = {
            'f1': 0,
            'epoch': -1
        }
        
        self.max_acc = {
            'acc': 0,
            'epoch': -1
        }
        
        self.patience = 5
        #self.best_val_acc = 0
        self.early_stop_counter = 0
        self.early_stop_counter_disc = 0
        self.early_stop = False
        self.best_acc = 0
        self.best_disc_loss = float('inf')
        self.best_epoch = 0
        self.best_disc_epoch = 0

        # ----------------------- debugging
        self.initial_hashes_sb=dict()
        self.initial_hashes_cb=dict()
        
        self.train_batch_size = self.args.batch_size
        self.test_batch_size = self.args.test_batch_size

        self.personalization_mode = False
        print(1, np.random.randint(0, 100, 10))
        self.init_clients_with_data()
        np.random.seed(self.args.seed)
        print(2, np.random.randint(0, 100, 10))
        self.init_client_models_optims()

        self.init_clients_server_copy()



if __name__ == '__main__':
    args = parse_arguments()
    trainer = ISICTrainer(args)
    trainer.fit()
    trainer.inference()