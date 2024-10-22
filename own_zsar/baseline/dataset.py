import os
import numpy as np
import pdb
import cv2
import torch
from torch.utils.data import Dataset
from baseline import semantic
import pandas as pd
from baseline import transform
from scipy.io import loadmat
import random


def filter_samples(num_samples, fnames, labels, classes):
    # Select a subset of samples
    fnames, labels = np.array(fnames), np.array(labels)
    if num_samples != -1:
        sel = np.linspace(0, len(fnames)-1,
                          min(num_samples, len(fnames))).astype(int)
        fnames, labels = fnames[sel], labels[sel]
    return np.array(fnames), np.array(labels), np.array(classes)


def subset_classes(subset, fnames, labels, classes):
    # Split classes into two subsets
    fnames1, classes1, labels1 = [], [], []
    fnames2, classes2, labels2 = [], [], []
    for i in range(len(fnames)):
        if labels[i] in subset:
            fnames1.append(fnames[i])
            labels1.append(labels[i])
        else:
            fnames2.append(fnames[i])
            labels2.append(labels[i])
    classes1 = np.unique(labels1)
    classes2 = np.unique(labels2)
    return fnames1, labels1, classes1, fnames2, labels2, classes2

def erroneous_trims(fname):
    # for kinetics dataset, the downloaded videos were trimmed videos - hence sometimes the videos contanied erroneous frames due to bad trimming.
    # So I have to check the files first for proper loading before sending them to dataloaders.

    # Initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print(f'Loading error for video {fname}')
        return True 
    else:
        return False


# Function to load video clips
def load_clips(fname, clip_len=16, n_clips=1, is_validation=False):
    np.random.seed(0)

    if not os.path.exists(fname):
        print('Missing: ' + fname)
        return []

    # Initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(fname)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count == 0 or frame_width == 0 or frame_height == 0:
        print('loading error, switching video ...')
        print(fname)
        return []

    total_frames = frame_count
    sampling_period = max(total_frames // n_clips, 1)
    n_snippets = min(n_clips, total_frames // sampling_period)
    if not is_validation:
        starts = np.random.randint(
            0, max(1, sampling_period - clip_len), n_snippets)
    else:
        starts = np.zeros(n_snippets)
    offsets = np.arange(0, total_frames, sampling_period)
    selection = np.concatenate([np.arange(of+s, of+s+clip_len)
                               for of, s in zip(offsets, starts)])

    frames = []
    count = ret_count = 0
    while count < selection[-1]+clip_len:
        retained, frame = capture.read()
        if count not in selection:
            count += 1
            continue

        if not retained:
            if len(frames) > 0:
                frame = np.copy(frames[-1])
            else:
                frame = (255*np.random.rand(frame_height,
                         frame_width, 3)).astype('uint8')
            frames.append(frame)
            ret_count += 1
            count += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1

    capture.release()
    frames = np.stack(frames)

    total = n_clips * clip_len
    while frames.shape[0] < total:
        frames = np.concatenate([frames, frames[:(total - frames.shape[0])]])
    frames = frames.reshape([n_clips, clip_len, frame_height, frame_width, 3])
    return frames


class VideoDataset(Dataset):
    def __init__(self, fnames, labels, class_embd, classes, name, seen_classes,
                 clip_len=8, n_clips=1, crop_size=112, is_validation=False):
        self.data = fnames
        self.labels = labels
        self.class_embd = class_embd
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
        buffer = load_clips(sample, self.clip_len,
                            self.n_clips, self.is_validation)
        if len(buffer) == 0:
            print("Video not loaded")
            buffer = np.random.rand(
                self.n_clips, 3, self.clip_len, 112, 112).astype('float32')
            buffer = torch.from_numpy(buffer)
            return buffer, -1, self.class_embd[0], -1

        s = buffer.shape
        buffer = buffer.reshape(s[0] * s[1], s[2], s[3], s[4])
        buffer = torch.stack([torch.from_numpy(im) for im in buffer], dim=0)

        tr = transform.get_transform(self.is_validation, self.crop_size)
        buffer = tr(buffer)
        buffer = buffer.reshape(
            3, s[0], s[1], self.crop_size, self.crop_size).transpose(0, 1)
        seen = True
        # print(self.labels[idx], self.seen_classes)
        if self.labels[idx] not in self.seen_classes:
            seen = False
        return buffer, label, self.class_embd[label], idx, seen

    def __len__(self):
        return len(self.data)


def get_split(dataset_name, index, classes):
    if index == 777:
        print("TruZe protocol activated!")
        return get_truze_splits(dataset_name, classes)
    elif dataset_name == 'ucf':
        return get_ucf_split(index, classes)
    elif dataset_name == 'hmdb':
        return get_hmdb_split(index, classes)
    elif dataset_name == 'olympics':
        return get_olympics_split(index, classes)
    elif dataset_name == 'activitynet':
        return get_anet_split(index, classes)
    elif dataset_name == 'test':
        return ['ApplyEyeMakeup', 'Basketball', 'BenchPress'], ['Biking', 'Bowling']

# new change made on 01-08-2023: TruZe splits for IEEE TSCVT
def process_ucf_classnames(classes):
    return [c.replace(" ", "") for c in classes]

def process_hmdb_classnames(classes):
    return [c.lower().replace(" ", "_") for c in classes]

def get_truze_splits(dataset_name, classes):
    ucf = {'dataset_name': 'UCF101',
           
           'train' : ['Apply Eye Makeup', 'Archery', 'Baby Crawling', 'Band Marching',  'Baseball Pitch', 'Basketball', 'Basketball Dunk', 'Bench Press', 'Biking', 'Blowing Candles', 'Cutting In Kitchen', 'Body Weight Squats', 'Bowling',' Boxing Punching Bag', 'Boxing Speed Bag', 'Breast Stroke', 'Brushing Teeth', 'Clean And Jerk', 'Cliff Diving', 'Cricket Bowling', 'Cricket Shot', 'Diving', 'Drumming', 'Floor Gymnastics', 'Frisbee Catch',  'Golf Swing', 'Haircut', 'Hammer Throw', 'Head Massage', 'High Jump', 'Horse Riding', 'Hula Hoop', 'Javelin Throw', 'Juggling Balls', 'Jump Rope', 'Kayaking', 'Knitting', 'Long Jump', 'Lunges', 'Mopping Floor', 'Playing Cello', 'Playing Flute', 'Playing Guitar', 'Playing Piano', 'Playing Violin', 'Pole Vault', 'Punch', 'Pull Ups', 'Push Ups', 'Rock Climbing Indoor', 'Rope Climbing', 'Salsa Spin', 'Shaving Beard', 'Shotput', 'Skate Boarding', 'Skiing', 'Skijet', 'Sky Diving', 'Soccer Juggling', 'Soccer Penalty', 'Surfing', 'Swing', 'TaiChi', 'Tennis Swing', 'Throw Discus', 'Trampoline Jumping', 'Typing', 'Volleyball Spiking', 'Walking With Dog', 'Writing On Board'],
           
           'test' : ['Apply Lipstick', 'Balance Beam', 'Billiards', 'Blow Dry Hair', 'Fencing', 'Field Hockey Penalty',  'Front Crawl',  'Hammering', 'Handstand Pushups', 'Handstand Walking', 'Horse Race', 'Ice Dancing', 'Jumping Jack', 'Military Parade', 'Mixing', 'Nunchucks', 'Parallel Bars', 'Pizza Tossing', 'Playing Daf', 'Playing Dhol',  'Playing Sitar', 'Playing Tabla', 'Pommel Horse', 'Rafting', 'Rowing', 'Still Rings', 'Sumo Wrestling', 'Table Tennis Shot', 'Uneven Bars', 'Wall Pushups', 'YoYo']
           }

    hmdb = {'dataset_name': 'HMDB51',
            
            'train' : ['Brush Hair', 'Cartwheel', 'Catch', 'Clap', 'Climb', 'Dive', 'Dribble', 'Drink', 'Eat', 'Golf', 'Hug', 'Kick Ball', 'Kiss', 'Laugh', 'Pullup', 'Punch', 'Push', 'Pushup', 'Ride Bike', 'Ride Horse', 'Shoot Ball', 'Shake Hands', 'Shoot Bow', 'Situp', 'Somersault', 'Swing Baseball', 'Smoke', 'Sword', 'Throw'],
            
            'test' : ['Chew', 'Climb Stairs', 'Draw Sword', 'Fall Floor', 'Fencing', 'Flic Flac', 'Handstand', 'Hit', 'Jump',  'Kick', 'Pick', 'Pour', 'Run', 'Sit', 'Shoot Gun', 'Smile', 'Stand', 'Sword Exercise', 'Talk', 'Turn', 'Walk', 'Wave']
            }

    assert len(ucf['train']) == 70
    assert len(ucf['test']) == 31
    assert len(hmdb['train']) == 29
    assert len(hmdb['test']) == 22

    all_datasets = [ucf, hmdb]
    for dataset in all_datasets:
         if dataset['dataset_name'] == 'UCF101':
             dataset['train'] = process_ucf_classnames(dataset['train'])
             dataset['test'] = process_ucf_classnames(dataset['test'])

            
         elif dataset['dataset_name'] == 'HMDB51':
             dataset['train'] = process_hmdb_classnames(dataset['train'])
             dataset['test'] = process_hmdb_classnames(dataset['test'])

    if dataset_name == 'ucf':
        # debugging
        # s = sorted(ucf['train'] + ucf['test'])
        # for c in s:
        #     if c not in classes:
        #         print(c)
        assert sorted(ucf['train'] + ucf['test']) == sorted(classes)
        return ucf['train'], ucf['test']
    
    elif dataset_name == 'hmdb':
        assert sorted(hmdb['train'] + hmdb['test']) == sorted(classes)
        return hmdb['train'], hmdb['test']




def get_ucf_split(index, classes):
    # Splits are in form of indices
    splits = loadmat('../../datasets/Split.mat')
    split = splits['Split'][0][0][1]
    train_split = split[0][0][0][index]
    test_split = split[0][0][1][index]
    # print(train_split, len(train_split))
    # print(test_split, len(test_split))
    # pdb.set_trace()
    classes.sort()
    train_split = [classes[ele - 1] for ele in train_split]
    test_split = [classes[ele - 1] for ele in test_split]

    # print("Post change")
    # print(train_split, len(train_split))
    # print(test_split, len(test_split))
    # pdb.set_trace()

    print()
    return train_split, test_split


def get_anet_split(index, classes):
    # Splits are stored in a .npy file
    with open('../../datasets/ActivityNet_v_1_3/anet_splits.npy', 'rb') as f:
        train_split = np.load(f, allow_pickle=True)
        test_split = np.load(f, allow_pickle=True)
        classes.sort()
        # indices of classes are assumed in the alphabetical order of class names
        train_split = [classes[ele] for ele in train_split[index]]
        test_split = [classes[ele] for ele in test_split[index]]

        return train_split, test_split        

def get_hmdb_split(index, classes):
    splits = loadmat('../../datasets/Split.mat')
    split = splits['Split'][0][0][0]
    train_split = split[0][0][0][index]
    test_split = split[0][0][1][index]
    classes.sort()
    train_split = [classes[ele - 1] for ele in train_split]
    test_split = [classes[ele - 1] for ele in test_split]
    return train_split, test_split


def get_olympics_split(index, classes):
    splits = loadmat('../../datasets/Split.mat')
    split = splits['Split'][0][0][3]
    train_split = split[0][0][0][index]
    test_split = split[0][0][1][index]
    classes.sort()
    train_split = [classes[ele - 1] for ele in train_split]
    test_split = [classes[ele - 1] for ele in test_split]
    return train_split, test_split


def get_test_data(dataset_name):
    if dataset_name == 'ucf':
        return get_ucf()
    elif dataset_name == 'hmdb':
        return get_hmdb()
    elif dataset_name == 'olympics':
        return get_olympics()
    elif dataset_name == 'activitynet':
        return get_anet()
    elif dataset_name == 'test':
        return get_test()

def get_ucf():
    folder = '../../datasets/ucf/UCF-101/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_anet():
    folder = '../../datasets/ActivityNet_v_1_3/Anet_videos_15fps_short256/'
    fnames, labels = [], []
    # for label in sorted(os.listdir(str(folder))):
    #     for fname in os.listdir(os.path.join(str(folder), label)):
    #         fnames.append(os.path.join(str(folder), label, fname))
    #         labels.append(label)

    with open('../../datasets/ActivityNet_v_1_3/anet_classwise_videos.npy', 'rb') as f:
        labels_to_videos = np.load(f, allow_pickle=True).item()
        classes = sorted(labels_to_videos.keys())
        for label in classes:
            for vid in labels_to_videos[label]:
                fnames.append(os.path.join(str(folder), vid))
                labels.append(label)

        print(f'Videos in dataset: {len(fnames)}')
        fnames, labels = np.array(fnames), np.array(labels)
        return fnames, labels, classes, folder    

def get_hmdb():
    folder = '../../datasets/hmdb/hmdb51_org/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        for fname in sorted(os.listdir(dir)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_olympics():
    folder = '../../datasets/olympic_sports_video/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        dir = os.path.join(str(folder), label)
        for fname in sorted(os.listdir(dir)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_test():
    folder = '../../datasets/test_data/'
    fnames, labels = [], []
    for label in sorted(os.listdir(str(folder))):
        for fname in os.listdir(os.path.join(str(folder), label)):
            fnames.append(os.path.join(str(folder), label, fname))
            labels.append(label)

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder


def get_kinetics700():
    folder = '../../datasets/kinetics/kinetics700_2020/'
    n_classes = 700
    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    fnames, labels = [], []
    for i in len(train):
        labels.append(train['label'][i])
        fnames.append(train['youtube_id'][i])

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_kinetics400():
    folder = '../../datasets/kinetics400/'
    files = ['train.csv', 'validate.csv', 'test.csv']
    fnames, labels = [], []
    for file in files:
        df = pd.read_csv(folder + file)
        for i in range(len(df)):
            labels.append(df['label'][i])
            fnames.append(df['youtube_id'][i])

    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_kinetics400_train():
    # change SS: 12-05-24
    # this is the function for obtaining the training data from kinetics-400. I wrote this while experimenting on the kinetics-400 dataset fpr IEEE TETCI
    folder = '../../datasets/kinetics-400/train_256/'
    train_csv = '../../datasets/kinetics-400/zsar_kinetics_400/zsar_train_k400.csv'
    fnames, labels = [], []
    corrupt_file = '../../datasets/kinetics-400/zsar_kinetics_400/k400_train_corrupt_clips.npy'
    df = pd.read_csv(train_csv)
    f = open(corrupt_file, 'rb')
    corrupt_fnames = np.load(f, allow_pickle=True)
    for i in range(len(df)):
        video_name, class_name = df['path'][i].split('/')[-1], df['path'][i].split('/')[-2]
        filepath = os.path.join(folder, class_name, video_name)
        if os.path.exists(filepath) and filepath not in corrupt_fnames:
            labels.append(class_name)
            fnames.append(filepath)

    f.close()
    fnames, labels = np.array(fnames), np.array(labels)
    classes = np.unique(labels)
    return fnames, labels, classes, folder

def get_kinetics600_val_test(split_index=0):
    # change SS: 12-05-24
    # this is the function for obtaining the training data from kinetics-400. I wrote this while experimenting on the kinetics-400 dataset fpr IEEE TETCI
    folder = '../../datasets/kinetics-600/'
    files = [f'zsar_kinetics_600/zsar_val_k600_s{split_index}.csv', f'zsar_kinetics_600/zsar_tst_k600_s{split_index}.csv']    
    fnames = {'validate': [], 'test': []}
    labels = {'validate': [], 'test': []}
    corrupt_files = [f'zsar_kinetics_600/k600_validate_s{split_index}_corrupt_clips.npy', f'zsar_kinetics_600/k600_test_s{split_index}_corrupt_clips.npy']

    for idx, file in enumerate(files):
        df = pd.read_csv(folder + file)
        datafolder = 'validate' if idx == 0 else 'test'
        f = open(folder + corrupt_files[idx], 'rb')
        corrupt_fnames = np.load(f, allow_pickle=True)
        for i in range(len(df)):
            filepath = os.path.join(folder, datafolder, df['labels'][i] , df['videoname'][i])
            if os.path.exists(filepath) and filepath not in corrupt_fnames:
                labels[datafolder].append(df['labels'][i])
                fnames[datafolder].append(filepath)
        f.close()

    fnames_val, labels_val = np.array(fnames['validate']), np.array(labels['validate'])
    fnames_test, labels_test = np.array(fnames['test']), np.array(labels['test'])

    classes_val, classes_test = np.unique(labels_val), np.unique(labels_test)
    return fnames_val, labels_val, classes_val, (folder + 'validate/'), fnames_test, labels_test, classes_test, (folder + 'test/')


def get_supervised_split_ucf(split_index):
    path = '../../datasets/ucf_supervised_splits/'
    train_path = path + 'trainlist0' + str(split_index) + '.txt'
    test_path = path + 'testlist0' + str(split_index) + '.txt'

    # Train
    ucf_fnames1, ucf_labels1 = [], []
    with open(train_path, 'r') as f:
        for line in f:
            fname, _ = line.split()
            ucf_fnames1.append('../../datasets/ucf/UCF-101/' + fname)
            label, _ = fname.split('/')
            ucf_labels1.append(label)
    ucf_fnames1, ucf_labels1 = np.array(ucf_fnames1), np.array(ucf_labels1)
    ucf_classes1 = np.unique(ucf_labels1)

    # Test
    ucf_fnames2, ucf_labels2 = [], []
    with open(test_path, 'r') as f:
        for line in f:
            fname = line[:-1]
            ucf_fnames2.append('../../datasets/ucf/UCF-101/' + fname)
            label, _ = fname.split('/')
            ucf_labels2.append(label)
    ucf_fnames2, ucf_labels2 = np.array(ucf_fnames2), np.array(ucf_labels2)
    ucf_classes2 = np.unique(ucf_labels2)

    return ucf_fnames1, ucf_labels1, ucf_classes1, ucf_fnames2, ucf_labels2, ucf_classes2

def get_supervised_split_hmdb(split_index):
    path = '../../datasets/hmdb_supervised_splits/'
    fnames1, labels1 = [], []
    fnames2, labels2 = [], []
    for fname in os.listdir(path):
        if split_index == int(fname[-5]):
            ind = fname.find('_test_split')
            label = fname[:ind]
            with open(os.path.join(path, fname), 'r') as f:
                for line in f:
                    name, idx, _ = line.split(" ")
                    if idx == "1":
                        fnames1.append('../../datasets/hmdb/hmdb51_org/' + label + '/' + name)
                        labels1.append(label)
                    elif idx == "2":
                        fnames2.append('../../datasets/hmdb/hmdb51_org/' + label + '/' + name)
                        labels2.append(label)

    fnames1, labels1 = np.array(fnames1), np.array(labels1)
    classes1 = np.unique(labels1)
    fnames2, labels2 = np.array(fnames2), np.array(labels2)
    classes2 = np.unique(labels2)
    return fnames1, labels1, classes1, fnames2, labels2, classes2

def get_supervised_split(dataset, split_index):
    if dataset == 'ucf':
        return get_supervised_split_ucf(split_index)
    elif dataset == 'hmdb':
        return get_supervised_split_hmdb(split_index)


def get_datasets(opt):

    if opt.dataset == 'kinetics':
        # get training data from kinetics-400. In ZSL, the entire training set of Kinetics-400 is considered as training data.
        print('ZSL Split statistics of ICCV 2021 paper (as per available data) \n----------------------------------------------------------')
        fnames1, labels1, classes1, _ = get_kinetics400_train()
        print(f"Training classes (from kinetics-400): {len(classes1)}")
        print(f"Training videos: {len(fnames1)}")
        embeddings1 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes1, opt.vit_backbone)
        ucf1 = VideoDataset(fnames1, labels1, embeddings1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)        

        # get validation and test sets separately as defined in the ICCV 2021 paper (https://github.com/DeLightCMU/ElaborativeRehearsal)
        fnames3, labels3, classes3, _, fnames2, labels2, classes2, _ = get_kinetics600_val_test(opt.split_index)
        print(f"Validation classes (from kinetics-600): {len(classes3)}")
        print(f"Validation videos: {len(fnames3)}")        
        print(f"Testing classes (from kinetics-600): {len(classes2)}")
        print(f"Testing videos: {len(fnames2)}") 

        embeddings3 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes3, opt.vit_backbone)
        ucf3 = VideoDataset(fnames3, labels3, embeddings3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        embeddings2 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes2, opt.vit_backbone)
        ucf2 = VideoDataset(fnames2, labels2, embeddings2, classes2, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        all_seen_classes = classes1
        all_unseen_classes = classes2



    elif opt.action in ['train', 'test']:
        fnames, labels, classes, _ = get_test_data(opt.dataset)
        subset, _ = get_split(opt.dataset, opt.split_index, classes)
        # print(subset, len(subset))
        fnames1, labels1, classes1, fnames2, labels2, classes2 = subset_classes(subset,
            fnames, labels, classes)

        print(f"Training classes: ({len(classes1)}): {classes1}") # training classes
        print(f"Videos for training: {len(fnames1)}")
        print(f"Testing classes: ({len(classes2)}): {classes2}") # training classes
        print(f"Videos for testing: {len(fnames2)}")

        # pdb.set_trace()

        # Training on 1/2 dataset
        embeddings1 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes1, opt.vit_backbone)
        ucf1 = VideoDataset(fnames1, labels1, embeddings1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Validation on 1/8 dataset
        size_val = (len(fnames2) // 4)
        fnames3, labels3, classes3 = filter_samples(
            size_val, fnames2, labels2, classes2)
        embeddings3 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes3, opt.vit_backbone)
        ucf3 = VideoDataset(fnames3, labels3, embeddings3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        # Testing on 1/2 dataset
        embeddings2 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes2, opt.vit_backbone)
        ucf2 = VideoDataset(fnames2, labels2, embeddings2, classes2, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        all_seen_classes = classes1
        all_unseen_classes = classes2

    elif opt.action in ['viz']:
        print("Not done for TruZe protocol yet! May work for ActivityNet...")
        fnames, labels, classes, _ = get_test_data(opt.dataset)
        subset1, subset2 = get_split(opt.dataset, opt.split_index, classes)
        more_classes = random.sample(subset2, 20)
        subset1 = np.append(subset1, more_classes)
        subset1 = np.unique(subset1)
        fnames1, labels1, classes1, fnames2, labels2, classes2 = subset_classes(subset1,
            fnames, labels, classes)

        # Training on 1/2 dataset
        embeddings1 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes1, opt.vit_backbone)
        ucf1 = VideoDataset(fnames1, labels1, embeddings1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Validation on 1/8 dataset
        size_val = (len(fnames2) // 4)
        fnames3, labels3, classes3 = filter_samples(
            size_val, fnames2, labels2, classes2)
        embeddings3 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes3, opt.vit_backbone)
        ucf3 = VideoDataset(fnames3, labels3, embeddings3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        # 30 classes * 20 videos
        fnames4, labels4, classes4 = filter_samples(
            600, fnames2, labels2, classes2)

        embeddings4 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes4, opt.vit_backbone)
        ucf2 = VideoDataset(fnames4, labels4, embeddings4, classes4, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        all_seen_classes = classes1
        all_unseen_classes = classes2
    
    # GZSL - for all datasets except Kinetics
    elif opt.action in ['gzsl_test']:
        fnames, labels, classes, _ = get_test_data(opt.dataset)
        subset, _ = get_split(opt.dataset, opt.split_index, classes)
        fnames1, labels1, classes1, fnames2, labels2, classes2 = subset_classes(subset,
            fnames, labels, classes)

        # Training on 1/2 dataset
        embeddings1 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes1, opt.vit_backbone)
        ucf1 = VideoDataset(fnames1, labels1, embeddings1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Validation on 1/8 dataset
        size_val = (len(fnames2) // 10)
        fnames3, labels3, classes3 = filter_samples(
            size_val, fnames2, labels2, classes2)
        embeddings3 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes3, opt.vit_backbone)
        ucf3 = VideoDataset(fnames3, labels3, embeddings3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        # Testing
        # 20% of seen
        size_seen = (len(fnames1) // 5)
        fnames4, labels4, classes4 = filter_samples(
            size_seen, fnames1, labels1, classes1)
        # print(classes1)
        # 100% of unseen
        fnames4 = np.append(fnames4, fnames2)
        labels4 = np.append(labels4, labels2)
        classes4 = np.unique(labels4)  # this guy sorts
        # print(classes4)
        embeddings4 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes4, opt.vit_backbone)
        ucf2 = VideoDataset(fnames4, labels4, embeddings4, classes4, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        
        all_seen_classes = classes1
        all_unseen_classes = classes4


    
    # Supervised
    elif opt.action in ['sup_train', 'sup_test']:
        fnames1, labels1, classes1, fnames2, labels2, classes2 = get_supervised_split(opt.dataset, opt.split_index)
        print(classes1)
        print(classes2)
        # exit()

        # Training on 1/2 dataset
        embeddings1 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes1, opt.vit_backbone)
        ucf1 = VideoDataset(fnames1, labels1, embeddings1, classes1, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_train, crop_size=opt.image_size, is_validation=False)

        # Validation on 1/8 dataset
        size_val = (len(fnames2) // 8)
        fnames3, labels3, classes3 = filter_samples(
            size_val, fnames2, labels2, classes2)
        embeddings3 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes3, opt.vit_backbone)
        ucf3 = VideoDataset(fnames3, labels3, embeddings3, classes3, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)

        # Testing on 1/2 dataset
        embeddings2 = semantic.semantic_embeddings(opt.semantic, opt.dataset, classes2, opt.vit_backbone)
        ucf2 = VideoDataset(fnames2, labels2, embeddings2, classes2, opt.dataset, classes1,
                            clip_len=opt.clip_len, n_clips=opt.n_clips_test, crop_size=opt.image_size, is_validation=True)
        all_seen_classes = classes1
        all_unseen_classes = classes2

    return {'training': [ucf1], 'validation': [ucf3], 'testing': [ucf2]}, all_seen_classes, all_unseen_classes


def load_datasets(opt):

    datasets, all_seen_classes, all_unseen_classes = get_datasets(opt)

    dataloaders = {}
    for key, datasets in datasets.items():
        dataloader = []
        if key == 'training':
            num_workers = opt.num_workers
        else:
            num_workers = 1
        for dataset in datasets:
            if opt.dataset == 'kinetics':
                # val and test batch size will be kept te same as in other experiments, i.e. 22. For training, increasing batch size to 32
                dl = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size if (
                                                 not dataset.is_validation) else 22,
                                             num_workers=num_workers,
                                             shuffle=not dataset.is_validation, 
                                             drop_last=False,
                                             pin_memory=dataset.is_validation
                                             )
            else:
                dl = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size // 2 if (
                                                 not dataset.is_validation) else opt.batch_size,
                                             num_workers=num_workers,
                                             shuffle=not dataset.is_validation, 
                                             drop_last=False,
                                             pin_memory=dataset.is_validation
                                             )
            dataloader.append(dl)
        dataloaders[key] = dataloader

    # exit(0)

            
    return dataloaders, all_seen_classes, all_unseen_classes
