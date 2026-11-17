import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#---------------------------------------------------------------------

def setting2_dirch_val(train_full_dataset, test_full_dataset, num_users):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []

    # Create DataFrames
    df = pd.DataFrame(list(zip(train_full_dataset.data, train_full_dataset.targets)), columns=['images', 'labels'])
    df_test = pd.DataFrame(list(zip(test_full_dataset.data, test_full_dataset.targets)), columns=['images', 'labels'])
    num_of_classes = df['labels'].nunique()

    # Initialize class-wise indices
    dict_classwise = {i: df[df['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}
    dict_classwise_test = {i: df_test[df_test['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}

    # Sample sizes per client
    total_train_samples_per_client = 160
    total_test_samples_per_client = 100
    total_val_samples_per_client = 20

    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes)*0.9)
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

        for j in range(num_of_classes):
            # Sampling training data
            train_indices = sample_data(dict_classwise, j, num_samples_train[j])
            dict_users[i].extend(train_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))

            # Sampling test data
            test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], replace=True)
            dict_users_test[i].extend(test_indices)

            # Sampling validation data
            val_indices = sample_data(dict_classwise, j, num_samples_val[j])
            dict_users_val[i].extend(val_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))

    return dict_users, dict_users_test, dict_users_val



def setting2_dirch_val2(train_full_dataset, test_full_dataset, num_users, alpha):
    np.random.seed(42)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []

    # Create DataFrames
    df = pd.DataFrame(list(zip(train_full_dataset.data, train_full_dataset.targets)), columns=['images', 'labels'])
    df_test = pd.DataFrame(list(zip(test_full_dataset.data, test_full_dataset.targets)), columns=['images', 'labels'])
    num_of_classes = df['labels'].nunique()

    # Initialize class-wise indices
    dict_classwise = {i: df[df['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}
    dict_classwise_test = {i: df_test[df_test['labels'] == i].index.values.astype(int) for i in range(num_of_classes)}

    # Sample sizes per client
    total_train_samples_per_client = 160
    total_test_samples_per_client = 100
    total_val_samples_per_client = 20

    for i in range(num_users):
        dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes)*alpha)
        num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
        num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
        num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

        for j in range(num_of_classes):
            # Sampling training data
            train_indices = sample_data(dict_classwise, j, num_samples_train[j])
            dict_users[i].extend(train_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))

            # Sampling test data
            test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], replace=True)
            dict_users_test[i].extend(test_indices)

            # Sampling validation data
            val_indices = sample_data(dict_classwise, j, num_samples_val[j])
            dict_users_val[i].extend(val_indices)
            dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))

    return dict_users, dict_users_test, dict_users_val

def sample_data(data_dict, class_index, num_samples, seed, replace=False):
    np.random.seed(seed)
    population_size = len(data_dict[class_index])
    if num_samples > population_size:
        num_samples = population_size
    return list(np.random.choice(data_dict[class_index], num_samples, replace=replace))

def check_ofchome(val_dist, train_dist):
    train_idx=np.argmax(train_dist)
    val_idx=np.argmax(val_dist)
    if val_idx == train_idx:
        # print('val_idx==train_idx')
        return False
    for i in range(len(val_dist)):
        if val_dist[i] > train_dist[i]:
            # print(val_dist[i],'>', train_dist[i])
            # print('False')
            return False
    return True

def get_val(train_samples, total_val_samples_per_client, alpha):
    # alpha=0.3
    idxs=[]
    train_samples_new=[]
    for i in range(len(train_samples)):
        if train_samples[i] > 0:
            idxs.append(i)
            train_samples_new.append(train_samples[i])
    
    val_dirichlet_dist = np.random.dirichlet(np.ones(len(train_samples_new)) * alpha)
    num_samples_val = np.round(val_dirichlet_dist * total_val_samples_per_client).astype(int)
    print(1, num_samples_val)
    while not check_ofchome(num_samples_val, train_samples_new):
        val_dirichlet_dist = np.random.dirichlet(np.ones(len(train_samples_new)) * alpha)
        num_samples_val = np.round(val_dirichlet_dist * total_val_samples_per_client).astype(int)
    
    val_samples=[0]*len(train_samples)
    for i in range(len(idxs)):
        val_samples[idxs[i]]=num_samples_val[i]
    
    return val_samples

def setting2_dirch_val_ofchome(train_full_dataset, test_full_dataset, num_users, mode, alpha, seed):
    # np.random.seed(seed)  # Set the seed for reproducibility
    dict_users, dict_users_test, dict_users_val = {}, {}, {}
    for i in range(num_users):
        dict_users[i] = []
        dict_users_test[i] = []
        dict_users_val[i] = []
        
    # num_of_classes = df['labels'].nunique()
    train_labels = np.array(train_full_dataset.targets)
    test_labels = np.array(test_full_dataset.targets)
    num_of_classes = len(np.unique(train_labels))
    # print(2)
    # Initialize class-wise indices
    print(f"=========== CLASSES: {num_of_classes}=======")
    print(f"=========== CLASSES: {np.unique(train_labels)}=======")
    dict_classwise = {i: np.where(train_labels == i)[0].tolist() for i in range(num_of_classes)}
    dict_classwise_test = {i: np.where(test_labels == i)[0].tolist() for i in range(num_of_classes)}
    print('train class freq:', {i: len(np.where(train_labels == i)[0].tolist()) for i in range(num_of_classes)})
    print('test class freq:', {i: len(np.where(test_labels == i)[0].tolist()) for i in range(num_of_classes)})
    #----------dataset split---------
    # cifar, fmnist -> 625, 1000, 125
    # office home -> 625, 300, 125
    total_train_samples_per_client = 160
    total_test_samples_per_client = 100
    total_val_samples_per_client = 20
    alpha=alpha
    print("**********************************", mode)
    
    if mode<2:
        if mode==0:
            b=0
            print("CLIENT's get IID TEST")
        elif mode==1:
            b=1
            print("CLIENT's get OOD TEST")
        else:
            Exception("Invalid mode")
        print("++++++++++++++++++", num_of_classes)
        for i in range(num_users):
            np.random.seed(seed+i)
            dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes)*alpha)
            num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
            # print(num_samples_train)
            # print("-"*50)
            num_samples_test = np.round(dirichlet_dist * total_test_samples_per_client).astype(int)
            num_samples_val = get_val(num_samples_train, total_val_samples_per_client, alpha)
            num_samples_train-=num_samples_val
            
            for j in range(num_of_classes):
                # Sampling training data
                train_indices = sample_data(dict_classwise, j, num_samples_train[j], seed)
                dict_users[i].extend(train_indices)
                dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))
                
                # Sampling validation data
                val_indices = sample_data(dict_classwise, j, num_samples_val[j], seed)
                dict_users_val[i].extend(val_indices)
                dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))
                
                # Sampling test data
                test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], seed)
                dict_users_test[(i+b)%num_users].extend(test_indices)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) - set(test_indices))
        
        for i in range(num_users):
            print('train:', len(dict_users[i]), 'val:', len(dict_users_val[i]), 'test:', len(dict_users_test[i]))
        
        return dict_users, dict_users_val, dict_users_test
    else:
        if mode==2:
            print('-----------------------------TEST SAMPLES - 20% IID, 80% OOD-----------------------------')
            iid_per=0.2
            ood_per=0.8
        elif mode==3:
            print('-----------------------------TEST SAMPLES - 50% IID, 50% OOD-----------------------------')
            iid_per=0.5
            ood_per=0.5
        elif mode==4:
            print('-----------------------------TEST SAMPLES - 80% IID, 20% OOD-----------------------------')
            iid_per=0.8
            ood_per=0.2
        else:
            raise ValueError("Invalid mode")

        clients_train_dist={}
        clients_val_dist={}
        clients_train_samples={}
        clients_val_samples={}
        clients_test_samples={}
        
        # creating train samples
        for i in range(num_users):
            np.random.seed(seed+i)
            dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes)*alpha)
            num_samples_train = np.round(dirichlet_dist * total_train_samples_per_client).astype(int)
            clients_train_dist[i]=dirichlet_dist
            clients_train_samples[i]=num_samples_train
        
        # # creating validation samples
        # for i in range(num_users):
            # val_dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes) * alpha)
            # num_samples_val = np.round(val_dirichlet_dist * total_val_samples_per_client).astype(int)
            num_samples_val = get_val(num_samples_train, total_val_samples_per_client, alpha)
            # num_samples_val = np.round(dirichlet_dist * total_val_samples_per_client).astype(int)

            # while not check(num_samples_val, clients_train_samples[i]):
            #     val_dirichlet_dist = np.random.dirichlet(np.ones(num_of_classes) * alpha)
            #     num_samples_val = np.round(val_dirichlet_dist * total_val_samples_per_client).astype(int)
            
            # clients_val_dist[i]=val_dirichlet_dist
            clients_val_samples[i]=num_samples_val
            clients_train_samples[i]-=num_samples_val
            
        # creating test samples
        for i in range(num_users):
            num_samples_test_iid = np.round(clients_train_dist[i] * int(total_test_samples_per_client*iid_per)).astype(int)
            num_samples_test_ood = np.round(clients_train_dist[(i+1)%num_users] * int(total_test_samples_per_client*ood_per)).astype(int)
            print(num_samples_test_iid, num_samples_test_ood)
            num_samples_test=[i+j for i,j in zip(num_samples_test_iid, num_samples_test_ood)]
            clients_test_samples[i]=num_samples_test
        print("done test sampling")
        for i in range(num_users):
            num_samples_train=clients_train_samples[i]
            num_samples_val=clients_val_samples[i]
            num_samples_test=clients_test_samples[i]
            for j in range(num_of_classes):
                # Sampling training data
                train_indices = sample_data(dict_classwise, j, num_samples_train[j], seed)
                dict_users[i].extend(train_indices)
                dict_classwise[j] = list(set(dict_classwise[j]) - set(train_indices))
                
                # Sampling validation data
                val_indices = sample_data(dict_classwise, j, num_samples_val[j], seed)
                dict_users_val[i].extend(val_indices)
                dict_classwise[j] = list(set(dict_classwise[j]) - set(val_indices))
                
                # Sampling test data
                test_indices = sample_data(dict_classwise_test, j, num_samples_test[j], seed)
                dict_users_test[i].extend(test_indices)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) - set(test_indices))
        
        return dict_users, dict_users_val, dict_users_test
    
class PACSDataset(Dataset):
    def __init__(self, img_paths, labels, client_id, dataset_type, tfms):
        self.img_paths = img_paths
        self.labels = labels
        self.client_id = client_id
        self.dataset_type = dataset_type  # 'train=0, 'val=1', or 'test=2'
        self.tfms = tfms

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        unique_id = f"{self.client_id}-{self.dataset_type}-{idx}"  # Append dataset_type to the unique ID

        if self.tfms:
            image = self.tfms(image)
        label = torch.tensor(label).long()
        return {
            'image': image,
            'label': label,
            'id': unique_id
        }

    def __len__(self):
        return len(self.img_paths)  # Return the length of the images (number of samples)

class PACSDataBuilder:
    def __init__(self, img_size=32, num_clients=10):
        self.img_size = img_size
        self.num_clients = num_clients

    def get_paths(self, test, client_id, seed):
        random.seed(seed)
        data_dir='/data/PACS/'
        basic_transform = transforms.Compose([transforms.ToTensor()])
        if test == 'caltech101':
            class_map = {0: 'LabelMe', 1: 'SUN09', 2: 'VOC2007'}
            train_dir = os.path.join(data_dir, class_map[client_id])
            test_dir = os.path.join(data_dir, 'Caltech101')
        elif test == 'labelme':
            class_map = {0: 'Caltech101', 1: 'SUN09', 2: 'VOC2007'}
            train_dir = os.path.join(data_dir, class_map[client_id])
            test_dir = os.path.join(data_dir, 'LabelMe')
        elif test == 'sun09':
            class_map = {0: 'Caltech101', 1: 'LabelMe', 2: 'VOC2007'}
            train_dir = os.path.join(data_dir, class_map[client_id])
            test_dir = os.path.join(data_dir, 'SUN09')
        elif test == 'voc2007':
            class_map = {0: 'Caltech101', 1: 'LabelMe', 2: 'SUN09'}
            train_dir = os.path.join(data_dir, class_map[client_id])
            test_dir = os.path.join(data_dir, 'VOC2007')
        else:
            raise ValueError("INCORRECT VALUE FOR --ood parameter, set either caltech101/labelme/sun09/voc2007")
        
        full_train_dataset = datasets.ImageFolder(train_dir, transform=basic_transform)
        full_test_dataset = datasets.ImageFolder(test_dir, transform=basic_transform)
        
        #---collecting train and val image paths
        train_indices = list(range(len(full_train_dataset)))
        random.shuffle(train_indices)
        trainset_indices=train_indices[0:int(0.8*len(full_train_dataset))]
        valset_indices=train_indices[int(0.8*len(full_train_dataset)):]
        train_img_paths = [full_train_dataset.samples[i][0] for i in trainset_indices]
        val_img_paths = [full_train_dataset.samples[i][0] for i in valset_indices]
        
        train_labels = np.array(full_train_dataset.targets)[trainset_indices]
        val_labels = np.array(full_train_dataset.targets)[valset_indices]
        
        #----collecting test image paths
        test_indices = list(range(len(full_test_dataset)))
        random.shuffle(test_indices)
        size=len(full_test_dataset)//3
        testset_indices=test_indices[client_id*size:(client_id+1)*size]
        test_img_paths = [full_test_dataset.samples[i][0] for i in testset_indices]
        
        test_labels = np.array(full_test_dataset.targets)[testset_indices]
        
        return train_img_paths, train_labels, val_img_paths, val_labels, test_img_paths, test_labels

    def get_dataset(self):
        transform=transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(root="/data/PACS_split/train/", transform=transform)
        test_dataset = datasets.ImageFolder(root="/data/PACS_split/test/", transform=transform)
        return train_dataset, test_dataset
    
    def get_default_transforms(self):
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        
        transform_test =  transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        
        return transform_train, transform_test

    def get_datasets(self, client_id, alpha, num_clients, ood, seed, transform_train=None, transform_test=None, pool=False):
        # train_img_paths, train_labels, val_img_paths, val_labels, test_img_paths, test_labels=self.get_paths(test=ood, client_id=client_id, seed=seed)
        train_dataset, test_dataset = self.get_dataset()
        if transform_train is None:
            transform_train, transform_test = self.get_default_transforms()

        dict_users, dict_users_val, dict_users_test = setting2_dirch_val_ofchome(train_dataset, test_dataset, num_clients, ood, alpha, seed)
        
        train_indices = dict_users[client_id]
        val_indices = dict_users_val[client_id]
        test_indices = dict_users_test[client_id]
        
        train_img_paths = [train_dataset.samples[idx][0] for idx in train_indices]
        train_labels = [train_dataset.samples[idx][1] for idx in train_indices]

        val_img_paths = [train_dataset.samples[idx][0] for idx in val_indices]
        val_labels = [train_dataset.samples[idx][1] for idx in val_indices]

        test_img_paths = [test_dataset.samples[idx][0] for idx in test_indices]
        test_labels = [test_dataset.samples[idx][1] for idx in test_indices]
        
        print("Training")
        train_ds = PACSDataset(train_img_paths, train_labels, client_id, 0, transform_train)
        print("Validation")
        val_ds = PACSDataset(val_img_paths, val_labels, client_id, 1,  transform_test)
        print("Testing")
        test_ds = PACSDataset(test_img_paths, test_labels, client_id, 2, transform_test)

        return train_ds, val_ds, test_ds

if __name__ == '__main__':
    cifar_builder = PACSDataBuilder()
    client_id = 0
    train_ds, val_ds, test_ds = cifar_builder.get_datasets(client_id)
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Example of fetching a batch
    item = next(iter(train_loader))
    im, lb = item['image'], item['label']
    print(f"Batch image shape: {im.shape}, Batch label shape: {lb.shape}")
