# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Local
import nn_utils

nn_utils.seed_everything(42)

class BBTT_DNN(nn.Module):
    def __init__(self, nIn=5, nHidden=32):
        super().__init__()
        self.hidden_0 = nn.Linear(nIn, nHidden)
        self.hidden_1 = nn.Linear(nHidden, nHidden)
        self.hidden_2 = nn.Linear(nHidden, nHidden)
        self.hidden_3 = nn.Linear(nHidden, 2)
    
    def forward(self, x):
        x = torch.relu(self.hidden_0(x))
        x = torch.relu(self.hidden_1(x))
        x = torch.relu(self.hidden_2(x))
        x = F.log_softmax(self.hidden_3(x), dim=1)
        return x

class GNNLayer(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.projection_0 = nn.Linear(nIn, nOut)
        self.projection_1 = nn.Linear(nOut, nOut)
        self.projection_2 = nn.Linear(nOut, nOut)
    
    def forward(self, node, adj):
        """
        node -> node feature vector, shape is [batch_size, n_nodes, n_in]
        adj -> adjacency matrices, same for all events for now, shape is [n_nodes, n_nodes]

        NOTE for both, the output shape is [b, n_nodes, n_out]
        bmm -> (b, n_nodes, n_nodes) x (b, n_nodes, n_out)  # 3D x 3D
        matmul -> (b, n_nodes, n_nodes) x (n_nodes, n_out)  # boardcasting
        """
        nNeighbours = adj.sum(dim=-1, keepdim=True)
        node = F.leaky_relu(self.projection_0(node), 0.01)
        node = F.leaky_relu(self.projection_1(node), 0.01)
        node = F.leaky_relu(self.projection_2(node), 0.01)
        node = torch.matmul(adj, node)
        node = node / nNeighbours
        return node


class GAttentionLayer(nn.Module):
    def __init__(self, nIn, nOut, nHeads=1, bConcat=True, fAlpha=0.2):
        super().__init__()
        self.nHeads = nHeads
        self.bConcat = bConcat
        if bConcat:
            assert nOut % nHeads == 0
            nOut = nOut // nHeads
        
        self.projection_0 = nn.Linear(nIn, nOut)
        self.projection_1 = nn.Linear(nOut, nOut)
        self.projection_2 = nn.Linear(nOut, nOut)
        self.a = nn.Parameter(torch.Tensor(nHeads, 2*nOut))
        self.leakyrelu = nn.LeakyReLU(fAlpha)

        nn.init.xavier_uniform_(self.projection_0.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection_1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection_2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
    
    def forward(self, node, adj, debug=False):
        nBatch, nNode = node.size()[0], node.size()[1]

        node = self.projection_0(node)
        node = self.projection_1(node)
        node = self.projection_2(node)
        node = node.view(nBatch, nNode, self.nHeads, -1)

        adj = torch.ones(nBatch, 1).mul(adj.view(1, -1)).view(-1, 6, 6)
        edges = torch.nonzero(adj)
        nodeFlat = node.view(nBatch * nNode, self.nHeads, -1)
        edgeIndicesRow = edges[:, 0] * nNode + edges[:, 1]
        edgeIndicesCol = edges[:, 0] * nNode + edges[:, 2]
        
        aIn = torch.cat([
            torch.index_select(nodeFlat, 0, edgeIndicesRow),
            torch.index_select(nodeFlat, 0, edgeIndicesCol)
        ], dim=-1)

        attnLogits = torch.einsum('bhc,hc->bh', aIn, self.a)
        attnLogits = self.leakyrelu(attnLogits)

        attnMatrix = attnLogits.new_zeros(adj.shape+(self.nHeads,)).fill_(-9e15)
        attnMatrix[adj[...,None].repeat(1,1,1,self.nHeads) == 1] = attnLogits.reshape(-1)

        attnProbs = F.softmax(attnMatrix, dim=2)
        if debug:
            print("Attention Probs\n", attnProbs.permute(0, 3, 1, 2))
        node = torch.einsum('bijh, bjhc->bihc', attnProbs, node)

        if self.bConcat:
            node = node.reshape(nBatch, nNode, -1)
        else:
            node = node.mean(dim=2)
        
        return node


class BBTT_GNN(nn.Module):
    def __init__(self, nIn=3, nHidden=8):
        super().__init__()

        # #    H  H  t  t  b  b  MET
        # self.adj = torch.Tensor([
        #     [1, 1, 1, 1, 0, 0, 0],  # H
        #     [1, 1, 0, 0, 1, 1, 0],  # H
        #     [1, 0, 1, 0, 0, 0, 1],  # t
        #     [1, 0, 0, 1, 0, 0, 1],  # t
        #     [0, 1, 0, 0, 1, 0, 1],  # b
        #     [0, 1, 0, 0, 0, 1, 1],  # b
        #     [0, 0, 1, 1, 1, 1, 1],  # MET
        # ])

        #    H  H  t  t  b  b 
        self.adj = torch.Tensor([
            [1, 1, 1, 1, 0, 0],  # H
            [1, 1, 0, 0, 1, 1],  # H
            [1, 0, 1, 1, 0, 0],  # t
            [1, 0, 1, 1, 0, 0],  # t
            [0, 1, 0, 0, 1, 1],  # b
            [0, 1, 0, 0, 1, 1],  # b
        ])

        self.hl_0 = nn.Linear(5, nHidden)
        self.hl_1 = nn.Linear(nHidden, nHidden)
        self.hl_2 = nn.Linear(nHidden, nHidden)
        self.hl_3 = nn.Linear(nHidden, nHidden)

        self.gnn_0 = GNNLayer(nIn, nHidden)
        self.gnn_1 = GNNLayer(nHidden, nHidden)
        self.gnn_2 = GNNLayer(nHidden, nHidden)
        self.gnn_3 = GNNLayer(nHidden, nHidden)
        self.gnn_final = nn.Linear(nHidden*self.adj.size(0), nHidden)

        self.fc_0 = nn.Linear(nHidden*2, nHidden*2)
        self.fc_1 = nn.Linear(nHidden*2, 2)

    def forward(self, x, o):
        o = F.leaky_relu(self.hl_0(o), 0.01)
        o = F.leaky_relu(self.hl_1(o), 0.01)
        o = F.leaky_relu(self.hl_2(o), 0.01)
        o = F.leaky_relu(self.hl_3(o), 0.01)
        
        x = F.leaky_relu(self.gnn_0(x, self.adj), 0.01)
        x = F.leaky_relu(self.gnn_1(x, self.adj), 0.01)
        x = F.leaky_relu(self.gnn_2(x, self.adj), 0.01)
        x = F.leaky_relu(self.gnn_3(x, self.adj), 0.01)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.gnn_final(x), 0.01)

        x = F.leaky_relu(self.fc_0(torch.cat([x,o], dim=1)), 0.01)
        x = F.log_softmax(self.fc_1(x), dim=1)

        return x


class BBTT_GATNN(nn.Module):
    def __init__(self, nIn=3, nHidden=8):
        super().__init__()

        #    H  H  t  t  b  b 
        self.adj = torch.Tensor([
            [1, 1, 1, 1, 0, 0],  # H
            [1, 1, 0, 0, 1, 1],  # H
            [1, 0, 1, 1, 0, 0],  # t
            [1, 0, 1, 1, 0, 0],  # t
            [0, 1, 0, 0, 1, 1],  # b
            [0, 1, 0, 0, 1, 1],  # b
        ])

        self.gattn_0 = GAttentionLayer(nIn, nHidden, 1)
        self.gattn_1 = GAttentionLayer(nHidden, nHidden, 1)
        self.gattn_2 = GAttentionLayer(nHidden, nHidden, 1)
        self.gattn_3 = GAttentionLayer(nHidden, nHidden, 1)
        self.fc_0 = nn.Linear(nHidden*self.adj.size(0), nHidden*self.adj.size(0))
        self.fc_1 = nn.Linear(nHidden*self.adj.size(0), 2)

    def forward(self, x):
        x = F.leaky_relu(self.gattn_0(x, self.adj), 0.01)
        x = F.leaky_relu(self.gattn_1(x, self.adj), 0.01)
        x = F.leaky_relu(self.gattn_2(x, self.adj), 0.01)
        x = F.leaky_relu(self.gattn_3(x, self.adj), 0.01)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc_0(x), 0.01)
        x = F.log_softmax(self.fc_1(x), dim=1)
        return x


def unit_test():
    dnn = BBTT_DNN()
    gnn = BBTT_GNN()
    gatnn = BBTT_GATNN()

    nParamDNN = sum(p.numel() for p in dnn.parameters() if p.requires_grad)
    nParamGNN = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    nParamGATNN = sum(p.numel() for p in gatnn.parameters() if p.requires_grad)

    nBatchSize = 10
    x1 = torch.randn(nBatchSize, 5)
    x2 = torch.randn(nBatchSize, 6, 3)
    x3 = torch.randn(nBatchSize, 6, 3)

    y1 = dnn(x1)
    y2 = gnn(x2)
    y3 = gatnn(x3)

    print(y1)
    print(y2)
    print(y3)

    print("DNN", nParamDNN, "GNN", nParamGNN, "GATNN", nParamGATNN)

# unit_test()
