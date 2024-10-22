import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from baseline.transformer_network import CLIPVideo, CLIPClassifier, CLIPTransformer, CLIPTransformerClassifier

def get_network(opt):
    """
    Selection function for available networks.
    """
    if opt.semantic == 'word2vec' or opt.semantic == 'fasttext':
        output_features = 300
    elif opt.semantic == 'sent2vec':
        output_features = 600
    elif 'clip' in opt.semantic:
        if opt.vit_backbone in ['ViT-B/16', 'ViT-B/32', 'RN101']:
            output_features = 512
        elif opt.vit_backbone == 'ViT-L/14':
            output_features = 768
        elif opt.vit_backbone in ['RN50']:
            output_features = 1024

    if opt.network == 'c3d':
        return C3D(out_features=output_features, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained)
    elif opt.network == 'r2plus1d':
        return ResNet18(r2plus1d_18, out_features=output_features, fixconvs=opt.fixconvs, nopretrained=opt.nopretrained, weights=R2Plus1D_18_Weights.DEFAULT)
    elif opt.network == 'clip':
        return CLIPVideo(vit_backbone=opt.vit_backbone)
    elif opt.network == 'clip_classifier':
        return CLIPClassifier(out_features=output_features, vit_backbone=opt.vit_backbone)
    elif opt.network == 'clip_transformer':
        return CLIPTransformer(T=opt.clip_len, LCA_drops=opt.LCA_drops, embed_dim=output_features, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)
    elif opt.network == 'clip_transformer_classifier':
        return CLIPTransformerClassifier(out_features=output_features, T=opt.clip_len, LCA_drops=opt.LCA_drops, drop_attn=opt.drop_attn_prob, droppath=opt.droppath, vit_backbone=opt.vit_backbone, lca_branch=opt.lca_branch)
    else:
        raise Exception('Network {} not available!'.format(opt.network))


class ResNet18(nn.Module):
    def __init__(self, network, out_features, fixconvs=True, nopretrained=True, weights=None):
        super(ResNet18, self).__init__()
        if nopretrained == True:
            self.model = network(weights=weights)
        if fixconvs:
            for param in self.model.parameters():
                param.requires_grad = False

        self.regressor = nn.Linear(self.model.fc.in_features, out_features)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        x = self.regressor(x)
        x = F.normalize(x)
        return x

    def output(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)
        x = self.model(x)
        x = x.view(bs*nc, -1)
        x = x.reshape(bs, nc, -1)
        x = torch.mean(x, 1)
        return x

class C3D(nn.Module):
    def __init__(self, out_features=300, fixconvs=True, nopretrained=True):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.10)

        self.relu = nn.ReLU()

        if nopretrained:
            self.load_state_dict(torch.load('../../datasets/c3d.pickle'))

        self.regressor = nn.Linear(4096, out_features)

        if fixconvs:
            for model in [self.conv1, self.conv2,
                          self.conv3a, self.conv3b,
                          self.conv4a, self.conv4b,
                          self.conv5a, self.conv5b,
                          self.fc6]:
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)

        h = h.reshape(bs, nc, -1)
        h = torch.mean(h, 1)
        h = h.reshape(bs, -1)

        h = self.regressor(h)
        h = torch.nn.functional.normalize(h, dim=-1)
        return h

    def output(self, x):
        bs, nc, ch, l, h, w = x.shape
        x = x.reshape(bs*nc, ch, l, h, w)

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))

        h = h.reshape(bs, nc, -1)
        h = torch.mean(h, 1)
        h = h.reshape(bs, -1)
        return h