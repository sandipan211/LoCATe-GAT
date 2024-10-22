import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import json
import os
import pandas as pd
import cv2
import jsonlines
import argparse

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
    
def find_corrupt_kinetics400_train():
    # change SS: 12-05-24
    # this is the function for obtaining the training data from kinetics-400. I wrote this while experimenting on the kinetics-400 dataset fpr IEEE TETCI
    folder = '../../datasets/kinetics-400/train_256/'
    train_csv = '../../datasets/kinetics-400/zsar_kinetics_400/zsar_train_k400.csv'
    fnames, labels = [], []
    problematic_clips = []
    clip_counter = 0
    df = pd.read_csv(train_csv)
    for i in range(len(df)):
        video_name, class_name = df['path'][i].split('/')[-1], df['path'][i].split('/')[-2]
        filepath = os.path.join(folder, class_name, video_name)
        clip_counter += 1
        if clip_counter % 1000 == 0:
            print(f'Verifying clip number {clip_counter}')
        if os.path.exists(filepath):
            if erroneous_trims(filepath):
                problematic_clips.append(filepath)
            else:
                labels.append(class_name)
                fnames.append(filepath)

    print('Dataset: Kinetics-400')
    print(f'Total training videos: {len(problematic_clips) + len(fnames)}')
    print(f'Rejected training videos from Kinetics-400 training set due to error in loading video: {len(problematic_clips)}')
    save_path = '../../datasets/kinetics-400/zsar_kinetics_400/k400_train_corrupt_clips.npy'
    with open(save_path, 'wb') as f:
        np.save(f, problematic_clips)


def find_corrupt_kinetics600_val_test(split_index=0):
    # change SS: 12-05-24
    # this is the function for obtaining the training data from kinetics-400. I wrote this while experimenting on the kinetics-400 dataset fpr IEEE TETCI
    folder = '../../datasets/kinetics-600/'
    files = [f'zsar_kinetics_600/zsar_val_k600_s{split_index}.csv', f'zsar_kinetics_600/zsar_tst_k600_s{split_index}.csv']    
    fnames = {'validate': [], 'test': []}
    labels = {'validate': [], 'test': []}
    problematic_clips = {'validate': [], 'test': []}

    for idx, file in enumerate(files):
        df = pd.read_csv(folder + file)
        datafolder = 'validate' if idx == 0 else 'test'
        print(f'Going through {datafolder}......')
        clip_counter = 0
        for i in range(len(df)):
            filepath = os.path.join(folder, datafolder, df['labels'][i] , df['videoname'][i])
            clip_counter += 1
            if clip_counter % 1000 == 0:
                print(f'Verifying clip number {clip_counter}')
            if os.path.exists(filepath):
                if erroneous_trims(filepath):
                    problematic_clips[datafolder].append(filepath)
                else:                       
                    labels[datafolder].append(df['labels'][i])
                    fnames[datafolder].append(filepath)

    print('Dataset: Kinetics-600')
    print(f'Split number: {split_index}')

    print('Total validate videos: ', len(problematic_clips['validate']) + len(fnames['validate']))
    print('Rejected videos from Kinetics-600 validation set due to error in loading video: ', len(problematic_clips['validate']))
    save_path = f'../../datasets/kinetics-600/zsar_kinetics_600/k600_validate_s{split_index}_corrupt_clips.npy'
    with open(save_path, 'wb') as f:
        np.save(f, problematic_clips['validate'])

    print('Total test videos: ', len(problematic_clips['test']) + len(fnames['test']))  
    print('Rejected videos from Kinetics-600 testing set due to error in loading video: ', len(problematic_clips['test']))
    save_path = f'../../datasets/kinetics-600/zsar_kinetics_600/k600_test_s{split_index}_corrupt_clips.npy'
    with open(save_path, 'wb') as f:
        np.save(f, problematic_clips['test'])    


def prepare_k400(split_folder):
    train_csv = '../../datasets/kinetics-400/zsar_kinetics_400/train.csv'
    # this is the original train.csv file containing all 234,619 video file names

    # prepare train df from k400
    train_df = pd.read_csv(train_csv, sep=' ', header=None)
    train_df.columns = ['path', 'label']
    print(f'Training videos from K-400: {len(train_df)}')

    ER_folder = split_folder
    k400_file = 'trn_video_names.json'

    with open(os.path.join(ER_folder, k400_file), 'r') as f:
        tr_data = json.load(f) # returns a list
        print(f'Training videos from K-400 in ER split: {len(tr_data)}')
        rows_to_drop = []
        for i in range(len(train_df)):
            video_name = train_df['path'][i].split('/')[-1]
            if i%1000 == 0:
                print(i)
            if video_name not in tr_data:
                rows_to_drop.append(i)
                
        zsar_train_df = train_df.drop(labels=rows_to_drop, axis=0)
        print(f'Training videos from K-400 in ER split available with us: {len(zsar_train_df)}')

        return zsar_train_df
        
def prepare_k600(split_folder, data='val'):

    ER_folder = split_folder
    k600_files = ['val_video_names.json', 'tst_video_names.json']

    if data == 'val':
        # preparing validation
        val_df = pd.read_csv('../../datasets/kinetics-600/validate.csv')
        print(f'Validation videos from K-600: {len(val_df)}')

        with open(os.path.join(ER_folder, k600_files[0]), 'r') as f:
            val_data = json.load(f) # returns a list
            
            # extracting youtube ids
            val_data = [val_data[i][:11] for i in range(len(val_data))] # youtube id always has length 11
            print(f'Validation videos from K-600 in ER split: {len(val_data)}')
            rows_to_drop = []
            for i in range(len(val_df)):
                video_name = val_df['youtube_id'][i]
                if i%1000 == 0:
                    print(i)
                if video_name not in val_data:
                    rows_to_drop.append(i)

            zsar_val_df = val_df.drop(labels=rows_to_drop, axis=0)
            print(f'Validation videos from K-600 in ER split available with us: {len(zsar_val_df)}')

            return zsar_val_df
    
    else:
        # preparing testing
        test_df = pd.read_csv('./kinetics600/test.csv')
        print(f'Testing videos from K-600: {len(test_df)}')
        with open(os.path.join(ER_folder, k600_files[1]), 'r') as f:
            test_data = json.load(f) # returns a list
            
            # extracting youtube ids
            test_data = [test_data[i][:11] for i in range(len(test_data))] # youtube id always has length 11
            print(f'Testing videos from K-600 in ER split: {len(test_data)}')
            rows_to_drop = []
            for i in range(len(test_df)):
                video_name = test_df['youtube_id'][i]
                if i%1000 == 0:
                    print(i)
                if video_name not in test_data:
                    rows_to_drop.append(i)

            zsar_test_df = test_df.drop(labels=rows_to_drop, axis=0)
            print(f'Testing videos from K-600 in ER split available with us: {len(zsar_test_df)}')
            zsar_test_df.to_csv('../../datasets/kinetics-600/zsar_kinetics_600/zsar_test_k600_overall.csv')

            return zsar_test_df


def make_zsar_split(split_num, split_folder, data):

    ED_list_620 = json.load(open(os.path.join(split_folder, 'classes620_label_defn.json')))

    if data == 'train':
        zsar_train_df = prepare_k400(split_folder)
        print(zsar_train_df.head())
        zsar_train_df.to_csv('../../datasets/kinetics-400/zsar_kinetics_400/zsar_train_k400.csv')

    elif data == 'val':
        zsar_val_df = prepare_k600(split_folder, data)
        print(zsar_val_df.head())
        zsar_val_df.to_csv('../../datasets/kinetics-600/zsar_kinetics_600/zsar_val_k600_overall.csv')

        val_class_idxs = json.load(open(os.path.join(split_folder, 'val_class_idxs.json')))[split_num]
        val_video_names = json.load(open(os.path.join(split_folder, 'val_video_names.json')))
        val_meta = os.path.join(split_folder, 'val_video_metas.jsonl')
        val_class_names = [ED_list_620[idx]['word'] for idx in val_class_idxs]
        val_videos = []
        val_labels = []
        # print(len(val_class_names))
        with jsonlines.open(val_meta, 'r') as f:
            for item in f:
                if item['label'] in val_class_idxs:
                    val_videos.append(item['videoname'])
                    val_labels.append(ED_list_620[item['label']]['word'])

        val_videos, val_labels = np.array(val_videos), np.array(val_labels)            
        zsar_val_df = pd.DataFrame({'videoname': val_videos, 'labels': val_labels})
        print(f'Val videos: {len(zsar_val_df)}') # verified true - matching paper's
        zsar_val_df.to_csv(f'../../datasets/kinetics-600/zsar_kinetics_600/zsar_val_k600_s{split_num}.csv')


    else:
        zsar_test_df = prepare_k600(split_folder, data)
        print(zsar_test_df.head())
        zsar_test_df.to_csv('../../datasets/kinetics-600/zsar_kinetics_600/zsar_test_k600_overall.csv')

        tst_class_idxs = json.load(open(os.path.join(split_folder, 'tst_class_idxs.json')))[split_num]
        tst_video_names = json.load(open(os.path.join(split_folder, 'tst_video_names.json')))
        tst_meta = os.path.join(split_folder, 'tst_video_metas.jsonl')
        # tst_class_names = [ED_list_620[idx]['word'] for idx in tst_class_idxs]
        tst_videos = []
        tst_labels = []
        with jsonlines.open(tst_meta, 'r') as f:
            for item in f:
                if item['label'] in tst_class_idxs:
                    tst_videos.append(item['videoname'])
                    tst_labels.append(ED_list_620[item['label']]['word'])

        tst_videos, tst_labels = np.array(tst_videos), np.array(tst_labels)            
        zsar_tst_df = pd.DataFrame({'videoname': tst_videos, 'labels': tst_labels})
        print(f'Test videos: {len(zsar_tst_df)}') # verified true - matching paper's
        zsar_tst_df.to_csv(f'../../datasets/kinetics-600/zsar_kinetics_600/zsar_tst_k600_s{split_num}.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True, type=str, help='Action: [find_corrupt, zsar_split]')
    parser.add_argument('--split_index', default=0, type=int, help='Split index for ZSAR split')    
    parser.add_argument('--dataset', default='k400', type=str, help='Dataset: [k400, k600]')
    parser.add_argument('--split_folder', default='../../datasets/ER-ZSAR/zsl220/', type=str, help='Path where actual split files in json format are stored (ICCV 2021 paper)')
    parser.add_argument('--data', default='val', type=str, help='Set: [train, val, test]')


    opt = parser.parse_args()

    if opt.dataset == 'k400' and opt.action == 'find_corrupt':
        find_corrupt_kinetics400_train()

    elif opt.dataset == 'k600' and opt.action == 'find_corrupt':
        find_corrupt_kinetics600_val_test(opt.split_index)

    elif opt.action == 'zsar_split':
        make_zsar_split(opt.split_index, opt.split_folder, opt.data)


