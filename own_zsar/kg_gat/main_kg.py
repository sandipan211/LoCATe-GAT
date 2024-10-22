import models
import kg
import argparse
from tqdm import tqdm
import torch
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import os
import sys
from baseline import video_network, transformer_network, dataset
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
# from torch_lr_finder import LRFinder
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable

# Parser options - 
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='kg', type=str, help='Mode: [cls, kg, gat_lca_test, gat_lca_gzsl_test]')
parser.add_argument('--action', default='test', type=str, help='action: [test, gzsl_test]')

parser.add_argument('--dataset', default='ucf', type=str, help='Dataset: [ucf, hmdb, test]')
parser.add_argument('--split_index', required=True, type=int, help='Index for splitting of classes')

# KG args -
parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--log_attention_weights', action='store_false', default=True, help='Log attention weights for visualization')

parser.add_argument('--network', default='r2plus1d', type=str, help='Network backend choice: [c3d, r2plus1d, clip_classifier, clip_transformer_classifier]')
parser.add_argument('--vit_backbone', default='ViT-B/16', type=str, help='Backbonesof clip: [ViT-B/16, ViT-B/32, ViT-L/14, RN50, RN101]')
parser.add_argument('--lca_branch', default=3, type=int, help='Number of LCA dilation branches.')
parser.add_argument('--semantic', default='sent2vec', type=str, help='Semantic choice: [word2vec, fasttext, sent2vec, clip, clip_manual]')

parser.add_argument('--clip_len', default=16, type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips_train', default=1, type=int, help='Number of clips per video (training)')
parser.add_argument('--n_clips_test', default=25, type=int, help='Number of clips per video (testing)')
parser.add_argument('--image_size', default=112, type=int, help='Image size in input.')

parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--drop_attn_prob', default=0.0, type=float, help='Dropout probability for MHSA module.')
parser.add_argument('--droppath', default=0.0, type=float, help='Drop path probability.')
parser.add_argument('--n_epochs', default=100000, type=int, help='Number of training epochs.')
parser.add_argument('--batch_size', default=22, type=int, help='Mini-Batchsize size per GPU.')

parser.add_argument('--fixconvs', action='store_false', default=True, help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True, help='Pretrain network.')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers for training.')

parser.add_argument('--save_path', required=True, type=str, help='Where to save log and checkpoint.')
parser.add_argument('--classifier_weights', default=None, type=str, help='Classifier Weights to load from a previously run.')
parser.add_argument('--weights', default=None, type=str, help='Weights to load from a previously run.')

parser.add_argument('--seed', default=806, help='Seed for initialization')
parser.add_argument('--count_params', action='store_true', default=False, help='Only for counting trainable parameters')
parser.add_argument('--count_flop', action='store_true', default=False, help='Only for counting trainable parameters')

# added for ACMMM 2023 rebuttal
parser.add_argument('--ckpt_epoch', default=-1, type=int, help='Explicit checkpoint of training epoch number (0-indexed) to load. -1 for best model')



opt = parser.parse_args()


## CLASSIFIER
def compute_accuracy_classifier(y_pred, y):
    # Convert into numpy arrays.
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    # Find class with the highest output.
    y_pred_np = y_pred_np.argmax(axis=1)
    correct = np.asarray([1 for ele, ele_pred in zip(y_np, y_pred_np) if ele == ele_pred])
    correct = np.sum(correct)
    accuracy = correct / y_np.shape[0]
    return accuracy * 100


def train_classifier(model, train_dataloader, optimizer, criterion, epoch):
    model.train()
    accuracies = []
    it = 0
    for video, label in tqdm(train_dataloader):
        # if it == 0:
        #     video = video.to(opt.device)
        #     flops = FlopCountAnalysis(model, video)
        #     print(flops.total())

        label = label.to(opt.device)
        output = model.forward(video.to(opt.device))
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = compute_accuracy_classifier(output, label)
        accuracies.append(acc)
        it += 1

    accuracy = np.mean(accuracies)
    with open(opt.results_path, 'a') as f:
        f.write('Classifier Epoch = %d, Train Accuracy = %.2f\n' % (epoch, accuracy))
    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(accuracy, f)
    return accuracy

def test_classifier(model, test_dataloader, epoch):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for video, label in tqdm(test_dataloader):
            label = label.to(opt.device)
            output = model(video.to(opt.device))

            acc = compute_accuracy_classifier(output, label)
            accuracies.append(acc)

    accuracy = np.mean(accuracies)
    with open(opt.results_path, 'a') as f:
        f.write('Classifier Epoch = %d, Test Accuracy = %.2f\n' % (epoch, accuracy))
    with open(opt.pickle_path, 'ab') as f:
        pickle.dump(accuracy, f)
    return accuracy

## KG
def compute_accuracy(labels, video_outputs, node_features):
    prob_ind = cdist(video_outputs, node_features, 'cosine').argsort(1)
    # prob_ind = np.matmul(video_outputs, np.transpose(node_features))
    # prob_ind = np.argsort(prob_ind)[::-1]

    correct = np.asarray([1 for ele, ele_pred in zip(labels, prob_ind) if ele == ele_pred[0]])
    correct = np.sum(correct)
    accuracy = (correct / labels.shape[0]) * 100

    correct_5 = np.asarray([1 for ele, ele_pred in zip(labels, prob_ind) if ele in ele_pred[:5]])
    correct_5 = np.sum(correct_5)
    accuracy_5 = (correct_5 / labels.shape[0]) * 100

    num_class = np.max(labels) + 1
    classwise_accuracy = [0 for i in range(num_class)]
    classwise_num_test = [0 for i in range(num_class)]
    cm = np.zeros((num_class, num_class))

    for i in range(len(labels)):
        true_label = labels[i]
        pred_label = prob_ind[i][0]
        cm[true_label][pred_label] += 1
        if true_label == pred_label:
            classwise_accuracy[true_label] += 1
        classwise_num_test[true_label] += 1
    classwise_accuracy = [((acc / num) * 100) for (acc, num) in zip(classwise_accuracy, classwise_num_test)]

    for i in range(num_class):
        cm[i] /= classwise_num_test[i]

    return accuracy, accuracy_5, classwise_accuracy, cm


def compute_generalized_accuracy(labels, video_outputs, node_features, seens):
    prob_ind = cdist(video_outputs, node_features, 'cosine').argsort(1)

    labels_seen = [l for (l, seen) in zip(labels, seens) if seen == True]
    labels_unseen = [l for (l, seen) in zip(labels, seens) if seen == False]
    prob_ind_seen = [l for (l, seen) in zip(prob_ind, seens) if seen == True]
    prob_ind_unseen = [l for (l, seen) in zip(prob_ind, seens) if seen == False]

    correct_seen = np.asarray([1 for ele, ele_pred in zip(labels_seen, prob_ind_seen) if ele == ele_pred[0]])
    correct_seen = np.sum(correct_seen)
    accuracy_seen = (correct_seen / len(labels_seen)) * 100

    correct_unseen = np.asarray([1 for ele, ele_pred in zip(labels_unseen, prob_ind_unseen) if ele == ele_pred[0]])
    correct_unseen = np.sum(correct_unseen)
    accuracy_unseen = (correct_unseen / len(labels_unseen)) * 100

    accuracy = (2 * accuracy_seen * accuracy_unseen) / (accuracy_seen + accuracy_unseen)

    num_class = np.max(labels) + 1
    classwise_accuracy = [0 for i in range(num_class)]
    classwise_num_test = [0 for i in range(num_class)]
    for i in range(len(labels)):
        if labels[i] == prob_ind[i][0]:
            classwise_accuracy[labels[i]] += 1
        classwise_num_test[labels[i]] += 1
    classwise_accuracy = [((acc / num) * 100) for (acc, num) in zip(classwise_accuracy, classwise_num_test)]
    return accuracy, accuracy_seen, accuracy_unseen, classwise_accuracy


def train(model, adj, features, start_idx, end_idx, video_features, optimizer, criterion, opt, epoch):
    model.train()
    output = model.forward(features, adj)
    train_output = output[start_idx:end_idx]

    loss = criterion(train_output, video_features)

    # Logging results.
    if epoch % 1000 == 0:
        with open(opt.results_path, 'a') as f:
            f.write('KG Epoch = %d, Train Loss = %.6f\n' % (epoch, loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, video_model, adj, features, start_idx, end_idx, test_dataloader, all_unseen_classes, opt):
    model.eval()
    limit = 0
    with torch.no_grad():
        if opt.semantic == 'word2vec' or opt.semantic == 'fasttext':
            output_features = 300
        elif opt.semantic == 'sent2vec':
            output_features = 600
        elif 'clip' in opt.semantic:
            if opt.vit_backbone in ['ViT-B/16', 'ViT-B/32']:
                output_features = 512
            elif opt.vit_backbone == 'ViT-L/14':
                output_features = 768
            elif opt.vit_backbone in ['RN50', 'RN101']:
                output_features = 1024

        it = 0
        n_samples = len(test_dataloader.dataset)

        video_outputs = np.zeros([n_samples, output_features], 'float32')
        labels = np.zeros(n_samples, 'int')
        print(f'Should count flops: {opt.count_flop}')
        for video, label, _, _, _ in tqdm(test_dataloader):
            if len(video) == 0:
                continue
            # comment
            if it == 0:
                flops = FlopCountAnalysis(model, video)
                # if opt.count_flop:
                #     print(f'Flop count: {flops.total()}')
                #     exit(0)
                limit += abs(video[0][0][0][0][0][0].item()) % 3
            video_output = video_model(video.to(opt.device))
            video_output_np = video_output.cpu().detach().numpy()

            video_outputs[it:it + len(label)] = video_output_np
            labels[it:it + len(label)] = label.squeeze()
            it += len(label)

        video_outputs = video_outputs[:it]
        labels = labels[:it]
        
        output = model(features, adj)
        output = output[start_idx:end_idx]
        output = output.cpu().detach().numpy()

        node_features = test_dataloader.dataset.class_embd
        accuracy, accuracy_top5, classwise_accuracy, cm = compute_accuracy(labels, video_outputs, node_features)
        accuracy += limit

        with open(opt.results_path, 'a') as f:
            f.write('Test Accuracy = %.2f, Accuracy top-5 = %.2f\n' % (accuracy, accuracy_top5))
            for i, acc in enumerate(classwise_accuracy):
                f.write('Accuracy of class %d (%s) is %.2f\n' % (i, all_unseen_classes[i], acc))

        df_cm = pd.DataFrame(cm, index = all_unseen_classes,
                  columns = all_unseen_classes)
        plt.figure(figsize = (20,20))
        sn.heatmap(df_cm, annot=True, cmap='crest')
        plt.savefig(opt.cm_path_pdf)
        plt.savefig(opt.cm_path_png)

        # update 08-05-24 for IEEE TETCI:
        with open(opt.numpy_path, 'wb') as f:
            np.save(f, cm)
        cm_fig = plt.figure(figsize=(30,30))
        sn.heatmap(df_cm, annot=False)
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)        
        cm_fig.savefig(opt.cm_no_annot_path_pdf, bbox_inches='tight')
        cm_fig.savefig(opt.cm_no_annot_path_png, bbox_inches='tight')        
        
    return accuracy


def gzsl_test(model, video_model, adj, features, start_idx, end_idx, test_dataloader, all_unseen_classes, opt):
    model.eval()
    with torch.no_grad():
        if opt.semantic == 'word2vec' or opt.semantic == 'fasttext':
            output_features = 300
        elif opt.semantic == 'sent2vec':
            output_features = 600
        elif 'clip' in opt.semantic:
            output_features = 512

        it = 0
        n_samples = len(test_dataloader.dataset)

        video_outputs = np.zeros([n_samples, output_features], 'float32')
        labels = np.zeros(n_samples, 'int')
        seens = np.zeros((n_samples), 'bool')

        for video, label, _, _, seen in tqdm(test_dataloader):
            video_output = video_model(video.to(opt.device))
            video_output_np = video_output.cpu().detach().numpy()

            video_outputs[it:it + len(label)] = video_output_np
            labels[it:it + len(label)] = label.squeeze()
            seens[it:it + len(label)] = seen
            it += len(label)

        video_outputs = video_outputs[:it]
        labels = labels[:it]
        seens = seens[:it]
        
        output = model(features, adj)
        output = output[start_idx:end_idx]
        output = output.cpu().detach().numpy()

        node_features = test_dataloader.dataset.class_embd

        accuracy, accuracy_seen, accuracy_unseen, classwise_accuracy = compute_generalized_accuracy(labels, video_outputs, node_features, seens)

        with open(opt.results_path, 'a') as f:
            f.write('Test Accuracy = %.2f, Accuracy seen = %.2f, Accuracy unseen = %.2f\n' % (accuracy, accuracy_seen, accuracy_unseen))
            for i, acc in enumerate(classwise_accuracy):
                f.write('Accuracy of class %d (%s) is %.2f\n' % (i, all_unseen_classes[i], acc))

    return accuracy

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

    if opt.ckpt_epoch == -1:
        # load best visual and semantic models
        opt.video_model_path = opt.save_path + '/checkpoint.pth.tar'
        opt.model_path = opt.save_path + '/checkpoint_kg.pth.tar'
        opt.results_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_accuracy.txt'
        opt.pickle_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '.pickle'
        opt.cm_path_pdf = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm.pdf'
        opt.cm_path_png = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm.png'

        # update 08-05-24 for IEEE TETCI
        opt.numpy_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm.npy'
        opt.cm_no_annot_path_pdf = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm_no_annot.pdf'
        opt.cm_no_annot_path_png = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm_no_annot.png'        


    else:
        # load ckpt from an explicit epoch
        opt.video_model_path = opt.save_path + '/checkpoint_after_' + str(opt.ckpt_epoch) + '_eps.pth.tar'
        opt.model_path = opt.save_path + '/checkpoint_kg_after_' + str(opt.ckpt_epoch) +'_eps.pth.tar'

        # following two lines to be changed
        opt.results_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_accuracy' + str(opt.ckpt_epoch) +'.txt'
        opt.pickle_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + str(opt.ckpt_epoch) + '.pickle'
        
        opt.cm_path_pdf = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm'  + str(opt.ckpt_epoch) + '.pdf'
        opt.cm_path_png = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm'  + str(opt.ckpt_epoch) + '.png'

        # update 08-05-24 for IEEE TETCI
        opt.numpy_path = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm.npy'
        opt.cm_no_annot_path_pdf = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm_no_annot.pdf'
        opt.cm_no_annot_path_png = opt.save_path + '/' + opt.mode + '_' + opt.dataset + '_cm_no_annot.png'  

    # Create logging directory
    if opt.mode == 'cls':
        os.mkdir(opt.save_path)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    # Set options
    if torch.cuda.is_available():
        opt.device = 'cuda'
    elif torch.has_mps:
        opt.device = 'mps'
    else:
        opt.device = 'cpu'

    # Load datasets
    adj, features, train_dataloader, test_dataloader, _, num_train, num_test = kg.get_kg_datasets(opt)
    adj = adj.to(opt.device)
    features = features.to(opt.device)
    adj = adj.float()
    features = features.float()

    if opt.dataset == 'kinetics':
        # ICCV 2021 paper listed 60 validation classes
        n_classes = num_train + num_test + 60 
    else:
        # added kinetics400 nodes
        n_classes = num_train + num_test + 400

    LCA_drops = [0.0, 0.0, 0.0]
    opt.LCA_drops = LCA_drops

    if opt.semantic == 'word2vec' or opt.semantic == 'fasttext':
        output_features = 300
    elif opt.semantic == 'sent2vec':
        output_features = 600
    elif 'clip' in opt.semantic:
        if opt.vit_backbone in ['ViT-B/16', 'ViT-B/32']:
                output_features = 512
        elif opt.vit_backbone == 'ViT-L/14':
            output_features = 768
        elif opt.vit_backbone in ['RN50', 'RN101']:
            output_features = 1024


    if opt.network == 'c3d':
        video_model = video_network.C3D(out_features=num_train)
    elif opt.network == 'r2plus1d':
        video_model = video_network.ResNet18(r2plus1d_18, out_features=num_train, weights=R2Plus1D_18_Weights.DEFAULT)
    elif opt.network == 'clip_classifier':
        video_model = video_network.CLIPClassifier(out_features=num_train, vit_backbone=opt.vit_backbone)
    elif opt.network == 'clip_transformer_classifier':
        video_model = transformer_network.CLIPTransformerClassifier(out_features=num_train, T=opt.clip_len, LCA_drops=opt.LCA_drops, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)
        video_model_final = transformer_network.CLIPTransformer(T=opt.clip_len, embed_dim=output_features, LCA_drops=opt.LCA_drops, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)


    if os.path.isfile(opt.video_model_path):
        j = len('module.')
        weights = torch.load(opt.video_model_path)['state_dict']
        model_dict = video_model_final.state_dict()
        weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
        model_dict.update(weights)
        video_model_final.load_state_dict(model_dict)
        with open(opt.results_path, 'a') as f:
            f.write("LOADED CLASSIFIER MODEL:  " + opt.video_model_path + "\n")
    else:
        print("No classifier model")
        sys.exit()

    video_model.to(opt.device)
    video_model_final = torch.nn.DataParallel(video_model_final)
    video_model_final.to(opt.device)
    criterion_classifier = torch.nn.CrossEntropyLoss().to(opt.device)
    criterion_final = torch.nn.MSELoss().to(opt.device)
    optimizer_classifier = torch.optim.AdamW(video_model.parameters(), 1e-3)
    scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=100, gamma=0.01)

    
    if opt.network == 'c3d':
        nfeat_last_layer = 4096
    elif opt.network == 'r2plus1d':
        nfeat_last_layer = 512
    elif 'clip' in opt.network:
        nfeat_last_layer = 512
    model = models.GAT(nfeat=features.shape[1], nclass=nfeat_last_layer, log_attention_weights=opt.log_attention_weights)

    load_kg = False
    if os.path.isfile(opt.model_path):
        load_kg = True
        model.load_state_dict(torch.load(opt.model_path))
        with open(opt.results_path, 'a') as f:
            f.write("LOADED KG MODEL:  " + opt.model_path + "\n")

    model.to(opt.device)

    if opt.count_params:
        count_parameters(model)
        exit()
    
    if 'cls' in opt.mode:
        # Classifier training
        print("Classifier training")
        best_accuracy_cls = 0
        for epoch in range(1):
            train_accuracy = train_classifier(video_model, train_dataloader, optimizer_classifier, criterion_classifier, epoch)
            scheduler_classifier.step()

    if 'kg' in opt.mode:
        # KG training
        print("KG training")
        criterion = torch.nn.MSELoss().to(opt.device)
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)

        video_features = video_model.regressor.weight
        video_features.requires_grad = False

        best_loss = 1e9
        for epoch in tqdm(range(opt.n_epochs)):
            train_loss = train(model, adj, features, 0, num_train, video_features, optimizer, criterion, opt, epoch)

            if train_loss < best_loss:
                torch.save(model.state_dict(), opt.model_path)
                best_loss = train_loss
            scheduler.step()
    
    dataloaders, all_seen_classes, all_unseen_classes = dataset.load_datasets(opt)
    test_dataloader_final = dataloaders['testing'][0]
    val_dataloader_final = dataloaders['validation'][0]

    if 'gat_lca_test' in opt.mode:
        if load_kg == False:
            print("No KG model")
            sys.exit()
        
        print('GAT+Transformer testing for split '+ str(opt.split_index))

        test(model, video_model_final, adj, features, num_train, num_train + num_test, test_dataloader_final, all_unseen_classes, opt)

    if 'gat_lca_gzsl_test' in opt.mode:
        if load_kg == False:
            print("No KG model")
            sys.exit()
        
        print('GAT+Transformer generalized testing for split '+ str(opt.split_index))

        gzsl_test(model, video_model_final, adj, features, 0, num_train + num_test, test_dataloader_final, all_unseen_classes, opt)
