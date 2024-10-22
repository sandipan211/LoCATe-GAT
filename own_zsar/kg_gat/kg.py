from baseline import semantic, dataset, transform
import numpy as np
from scipy import spatial
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import networkx as nx
import matplotlib.pyplot as plt 

def normalize_adj(W):
    """Row-normalize sparse matrix"""
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D 


def normalize_features(a):
    """Row-normalize matrix"""
    row_sums = a.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] == 0:
            row_sums[i] = 1

    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

def knowledge_graph(opt, train_classes, test_classes, val_classes=[], n=5):
    if opt.dataset == 'kinetics':
        ''' 
        during validation loss calculation, it needs to take out the weights corresponding to the validation classes. 
        For other datasets, we did not need to keep nodes for these classes separately as they were same as testing classes.
        But since for kinetics splits val and test classes are different, we need to separately consider these val classes.
        Moreover, the extra nodes for Kinetics-400 need not be added separately. 
        
        '''

        # classes = np.append(train_classes, test_classes, val_classes) => error: cannot append more than two arrays like this
        # print(test_classes.shape, val_classes.shape, train_classes.shape)
        classes = np.concatenate((train_classes, test_classes, val_classes))
        embd = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes, opt.vit_backbone)
        embd = np.asarray(embd)

    else:
        
        classes = np.append(train_classes, test_classes)
        _, _, kinetics_classes, _ = dataset.get_kinetics400()

        embd = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes, opt.vit_backbone)
        embd = np.asarray(embd)
        
        embd_kinetics = semantic.semantic_embeddings(opt.semantic, 'kinetics', kinetics_classes, opt.vit_backbone)
        embd_kinetics = np.asarray(embd_kinetics)

        embd = np.append(embd, embd_kinetics, axis=0)

    # exit(0)

    embd = normalize_features(embd)

    n_classes = embd.shape[0]
    adj = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        dist = np.zeros(n_classes)
        for j in range(n_classes):
            dist[j] = spatial.distance.cosine(embd[i], embd[j])
        neighbors = dist.argsort()[:n]
        for idx in neighbors:
            # if idx < len(classes):
                # print(classes[i], classes[idx])
            adj[i][idx] = 1 #dist[idx]

    # build symmetric adjacency matrix
    adj = adj + np.multiply(adj.T, (adj.T > adj)) - np.multiply(adj, (adj.T > adj))
    adj = normalize_adj(adj)
    return adj, embd

class VideoDataset(Dataset):
    def __init__(self, fnames, labels, classes, name, seen_classes,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False):
        self.data = fnames
        self.labels = labels
        self.class_name = classes
        self.name = name
        self.seen_classes = seen_classes

        self.clip_len = clip_len
        self.n_clips = n_clips

        self.crop_size = crop_size
        self.is_validation = is_validation

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index,
                            label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array(
            [self.label2index[label] for label in labels], dtype=int)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label_array[idx]
        buffer = dataset.load_clips(sample, self.clip_len,
                            self.n_clips, self.is_validation)
        if len(buffer) == 0:
            print("Video not loaded")
            buffer = np.random.rand(
                self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1

        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], dim=0)

        tr = transform.get_transform(self.is_validation, self.crop_size)
        buffer = tr(buffer)
        buffer = buffer.reshape(
            3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        seen = True
        if self.labels[idx] not in self.seen_classes:
            seen = False
        return buffer, label, seen

    def __len__(self):
        return len(self.data)

def split_kg(dataset_name, classes):
    # Split as in paper.
    if dataset_name == 'ucf':
        test_classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'BalanceBeam', 'Billiards',
            'FieldHockeyPenalty', 'FrontCrawl', 'Hammering', 'HandstandWalking', 'JumpingJack',
            'Mixing', 'Nunchucks', 'ParallelBars', 'PlayingDaf', 'PlayingDhol', 'PlayingSitar',
            'PlayingTabla', 'PommelHorse', 'Rafting', 'StillRings', 'TableTennisShot', 'Typing', 
            'UnevenBars', 'YoYo']
        train_classes = [cl for cl in classes if cl not in test_classes]
        return train_classes, test_classes
    elif dataset_name == 'hmdb':
        test_classes = ['chew', 'climb_stairs', 'fall_floor', 'handstand', 'pour', 'shoot_gun', 
            'sit', 'smile', 'stand', 'talk', 'turn', 'wave']
        train_classes = [cl for cl in classes if cl not in test_classes]
        return train_classes, test_classes

def get_kg_datasets(opt):

    if opt.dataset == 'kinetics':
        # get training data from kinetics-400. In ZSL, the entire training set of Kinetics-400 is considered as training data.
        print('ZSL Split statistics of ICCV 2021 paper (as per available data) \n----------------------------------------------------------')
        fnames1, labels1, classes1, folder_train = dataset.get_kinetics400_train()
        print(f"Training classes (from kinetics-400): {len(classes1)}")
        print(f"Training videos: {len(fnames1)}")

        ucf1 = VideoDataset(fnames1, labels1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)        

        # get validation and test sets separately as defined in the ICCV 2021 paper (https://github.com/DeLightCMU/ElaborativeRehearsal)
        fnames3, labels3, classes3, _, fnames2, labels2, classes2, _ = dataset.get_kinetics600_val_test(opt.split_index)
        print(f"Validation classes (from kinetics-600): {len(classes3)}")
        print(f"Validation videos: {len(fnames3)}")        
        print(f"Testing classes (from kinetics-600): {len(classes2)}")
        print(f"Testing videos: {len(fnames2)}") 

        ucf3 = VideoDataset(fnames3, labels3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        ucf2 = VideoDataset(fnames2, labels2, classes2, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)



    elif opt.action in ['train', 'test']:
        fnames, labels, classes, _ = dataset.get_test_data(opt.dataset)
        subset, _ = dataset.get_split(opt.dataset, opt.split_index, classes)
        fnames1, labels1, classes1, fnames2, labels2, classes2 = dataset.subset_classes(subset,
            fnames, labels, classes)
        
        # Training
        ucf1 = VideoDataset(fnames1, labels1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Testing
        ucf2 = VideoDataset(fnames2, labels2, classes2, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        # Validation
        size_val = len(fnames2) // 10
        fnames3, labels3, classes3 = dataset.filter_samples(
            size_val, fnames2, labels2, classes2)
        ucf3 = VideoDataset(fnames3, labels3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
    elif opt.action in ['gzsl_test']:
        # for all datasets except Kinetics
        fnames, labels, classes, _ = dataset.get_test_data(opt.dataset)
        subset, _ = dataset.get_split(opt.dataset, opt.split_index, classes)
        fnames1, labels1, classes1, fnames2, labels2, classes2 = dataset.subset_classes(subset,
            fnames, labels, classes)
        
        # Training
        ucf1 = VideoDataset(fnames1, labels1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Validation
        size_val = len(fnames2) // 10
        fnames3, labels3, classes3 = dataset.filter_samples(
            size_val, fnames2, labels2, classes2)
        ucf3 = VideoDataset(fnames3, labels3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        # Testing
        # 20% of seen
        size_seen = (len(fnames1) // 5)
        fnames4, labels4, classes4 = dataset.filter_samples(
            size_seen, fnames1, labels1, classes1)
        # 100% of unseen
        fnames4 = np.append(fnames4, fnames2)
        labels4 = np.append(labels4, labels2)
        classes4 = np.unique(labels4)
        ucf2 = VideoDataset(fnames4, labels4, classes4, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        classes2 = classes4
    

    # training
    ucf_dl1 = torch.utils.data.DataLoader(ucf1,
                                             batch_size=opt.batch_size // 2 if opt.dataset != 'kinetics' else opt.batch_size,
                                             num_workers=16, pin_memory=False,
                                             shuffle=True, drop_last=False)

    # Testing
    ucf_dl2 = torch.utils.data.DataLoader(ucf2,
                                             batch_size=opt.batch_size if opt.dataset != 'kinetics' else 22,
                                             num_workers=1, pin_memory=True,
                                             shuffle=False, drop_last=False)

    # Validation
    ucf_dl3 = torch.utils.data.DataLoader(ucf3,
                                             batch_size=opt.batch_size if opt.dataset != 'kinetics' else 22,
                                             num_workers=1, pin_memory=True,
                                             shuffle=False, drop_last=False)

    print("Building knowledge graph")
    if opt.dataset == 'kinetics':
        adj, features = knowledge_graph(opt, classes1, classes2, classes3)
    else:
        adj, features = knowledge_graph(opt, classes1, classes2)
    adj = torch.from_numpy(adj)
    features = torch.from_numpy(features)
    return adj, features, ucf_dl1, ucf_dl2, ucf_dl3, len(classes1), len(classes2)
