import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphConvolution


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout=0.6, alpha=0.2, log_attention_weights=True):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions1 = [GraphAttentionLayer(nfeat, 256, dropout=dropout, alpha=alpha, concat=True, log_attention_weights=log_attention_weights) for _ in range(4)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayer(1024, 256, dropout=dropout, alpha=alpha, concat=True, log_attention_weights=log_attention_weights) for _ in range(4)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)

        self.out_att = [GraphAttentionLayer(1024, nclass, dropout=dropout, alpha=alpha, concat=False, log_attention_weights=log_attention_weights) for _ in range(6)]
        for i, attention in enumerate(self.out_att):
            self.add_module('attention3_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=1)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.stack([att(x, adj) for att in self.out_att], axis=-1)
        x = torch.mean(x, -1)
        x = F.elu(x)
        x = F.normalize(x)
        
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.gc4 = GraphConvolution(1024, 1024)
        self.gc5 = GraphConvolution(1024, 1024)
        self.gc6 = GraphConvolution(1024, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc6(x, adj)
        x = F.normalize(x)
        return x