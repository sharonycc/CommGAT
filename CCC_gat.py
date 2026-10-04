# Written By 
# Fatema Tuz Zohora

from scipy import sparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax #Linear, 
from torch_geometric.data import Data, DataLoader
import gzip, pickle
from epoch_eval_utils import evaluate_pseudo_edge_scores

from GATv2Conv_CellNEST import GATv2Conv

def get_graph(training_data, expression_matrix_path=''):
    """Add Statement of Purpose
    Args:
        training_data: Path to the input graph    
    Returns:
        List of torch_geometric.data.Data type: Loaded input graph
        Integer: Dimension of node embedding
    """
    
    f = gzip.open(training_data, 'rb')
    row_col, edge_weight, lig_rec, num_cell, w1, w2, w3 = pickle.load(f)

    datapoint_size = num_cell
    #print(edge_weight)
    if expression_matrix_path == '':
        # one hot vector used as node feature vector
        X = np.eye(datapoint_size, datapoint_size)
        np.random.shuffle(X)

    else:
        f = gzip.open(expression_matrix_path, 'rb')
        X = pickle.load(f)

    X_data = X # node feature vector
    num_feature = X_data.shape[1]#基因
    
    print('Node feature matrix: X has dimension ', X_data.shape)
    print("Total number of edges in the input graph is %d"%len(row_col))
    

    ###########

    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = []
    for k in range(len(row_col)):
        ew = edge_weight[k]

        if np.isscalar(ew):
            ew = [float(ew)]
        else:
            ew = list(np.asarray(ew).ravel())

        ew_extended = ew + [float(w1[k]), float(w2[k]), float(w3[k])]
        edge_attr.append(ew_extended)

    edge_attr = torch.tensor(np.asarray(edge_attr), dtype=torch.float)
    print("edge_attr shape:", edge_attr.shape)

    graph_bags = []
    graph = Data(x=torch.tensor(X_data, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)

    print('Input graph generation done')

    data_loader = DataLoader(graph_bags, batch_size=1) # moved to get_graph
    
    return data_loader, num_feature

class ExpressionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


# class EdgeWeightScorer(nn.Module):
#     """
#     用于学习配受体介导细胞通信边的权重组合关系。
#
#     输入：
#         z_i, z_j
#         edge_cont = [cell_ij边权重, LR共表达, w1, w2, w3]
#
#     不直接输入 LR_ID 的连续编号，避免模型学习错误的大小关系。
#     """
#
#     def __init__(self, node_dim, hidden_dim=128, dropout=0.1):
#         super().__init__()
#
#         in_dim = node_dim * 4
#
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#
#     def forward(self, z, edge_index):
#         src = edge_index[0]
#         dst = edge_index[1]
#
#         z_src = z[src]
#         z_dst = z[dst]
#
#         # edge_attr: [cell_weight, LR共表达, LR_ID, w1, w2, w3]
#         # 保留 LR_ID 在 edge_attr 中，但不作为连续变量输入 scorer
#         # edge_cont = edge_attr[:, [0, 1, 3, 4, 5]]
#
#         h = torch.cat([
#             z_src,
#             z_dst,
#             z_src * z_dst,
#             torch.abs(z_src - z_dst)
#         ], dim=1)
#
#         logits = self.net(h).squeeze(-1)
#
#         return logits
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """

        
        super(Encoder, self).__init__()
        print('incoming channel %d'%in_channels)

        heads = heads
        self.expr_encoder = ExpressionEncoder(in_dim=in_channels, hidden_dim=hidden_channels, dropout=dropout)
        self.conv =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=6, heads=heads, concat = False,  dropout=dropout)#)
        self.conv_2 =  GATv2Conv(hidden_channels, hidden_channels, edge_dim=6, heads=heads, concat = False, dropout=dropout)#, dropout=0)

        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'

        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'

        #self.prelu = nn.Tanh(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)

        # self.edge_scorer = EdgeWeightScorer(node_dim=hidden_channels, hidden_dim=128, dropout=dropout)
        # self.edge_scores = None



    def forward(self, data,edge_attr_override=None):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        edge_attr = data.edge_attr if edge_attr_override is None else edge_attr_override

        # layer 1
        x = self.expr_encoder(data.x)
        x, attention_scores, attention_scores_unnormalized = self.conv(x, data.edge_index, edge_attr=edge_attr, return_attention_weights = True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized


        # layer 2
        x, attention_scores, attention_scores_unnormalized  = self.conv_2(x, data.edge_index, edge_attr=edge_attr, return_attention_weights = True)  # <---- ***
        self.attention_scores_mine = attention_scores #self.attention_scores_mine_l1 #attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized #self.attention_scores_mine_unnormalized_l1 #attention_scores_unnormalized

        ###############################
        z = self.prelu(x)
        # edge_logits = self.edge_scorer(z=z, edge_index=data.edge_index)
        # self.edge_scores = torch.sigmoid(edge_logits)

        return z #, attention_scores

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        """Add Statement of Purpose
        Args: [to be]
               
        Returns: [to be]
    
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data, corrupt_ratio=0.3):
    x_neg = data.x
    edge_index_neg = data.edge_index
    edge_attr_neg = data.edge_attr.clone()

    E = edge_attr_neg.size(0)
    num_corrupt = int(E * corrupt_ratio)

    corrupt_edges = torch.randperm(E, device=edge_attr_neg.device)[:num_corrupt]
    corrupt_mask = torch.zeros(E, dtype=torch.bool, device=edge_attr_neg.device)
    corrupt_mask[corrupt_edges] = True

    # edge_attr:
    # 0: cell_ij边权重
    # 1: LR共表达
    # 2: LR_ID
    # 3: w1
    # 4: w2
    # 5: w3
    lr_ids = edge_attr_neg[:, 2].long()
    cols_to_shuffle = [1, 3, 4, 5]

    for lr in torch.unique(lr_ids):
        idx = torch.where((lr_ids == lr) & corrupt_mask)[0]

        if idx.numel() <= 1:
            continue

        for k in cols_to_shuffle:
            perm = idx[torch.randperm(idx.numel(), device=edge_attr_neg.device)]
            edge_attr_neg[idx, k] = edge_attr_neg[perm, k]

    return my_data(x_neg, edge_index_neg, edge_attr_neg)

def attention_contrastive_loss(real_att, fake_att, margin=0.05):
    """
    目标：
        原始 edge_attr 下的 attention
        >
        同 LR_ID 内打乱 edge_attr 后的 attention

    不使用伪标签。
    """

    if isinstance(real_att, tuple):
        real_att = real_att[1]
    if isinstance(fake_att, tuple):
        fake_att = fake_att[1]

    if real_att.dim() > 1:
        real_att = real_att.mean(dim=1)
    if fake_att.dim() > 1:
        fake_att = fake_att.mean(dim=1)

    real_att = torch.sigmoid(real_att)
    fake_att = torch.sigmoid(fake_att)

    target = torch.ones_like(real_att)

    return F.margin_ranking_loss(
        real_att,
        fake_att,
        target,
        margin=margin
    )

def make_hard_fake_edge_attr(edge_attr, corrupt_ratio=0.5):
    """
    无监督 hard negative:
    在同一个 LR_ID 内部打乱 LR共表达、w1、w2、w3。
    不使用伪标签乘积。
    """

    fake = edge_attr.clone()

    E = fake.size(0)
    num_corrupt = int(E * corrupt_ratio)

    corrupt_edges = torch.randperm(E, device=fake.device)[:num_corrupt]
    corrupt_mask = torch.zeros(E, dtype=torch.bool, device=fake.device)
    corrupt_mask[corrupt_edges] = True

    lr_ids = fake[:, 2].long()
    cols_to_shuffle = [1, 3, 4, 5]

    for lr in torch.unique(lr_ids):
        idx = torch.where((lr_ids == lr) & corrupt_mask)[0]

        if idx.numel() <= 1:
            continue

        for col in cols_to_shuffle:
            perm = idx[torch.randperm(idx.numel(), device=fake.device)]
            fake[idx, col] = fake[perm, col]

    return fake

# class TwoLossWeights(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.raw_weights = nn.Parameter(torch.zeros(2))
#
#     def forward(self, dgi_loss, edge_loss):
#         weights = torch.softmax(self.raw_weights, dim=0)
#
#         # total_loss = (
#         #     weights[0] * dgi_loss
#         #     + weights[1] * edge_loss
#         # )
#         total_loss = (
#                 0.5 * dgi_loss
#                 + 0.5 * edge_loss
#         )

        # return total_loss, weights.detach()
class BoundedSumToOneLossWeight(nn.Module):
    def __init__(self, dgi_min=0.45, dgi_max=0.55):
        super().__init__()

        self.dgi_min = dgi_min

        self.dgi_max = dgi_max

        self.raw_dgi = nn.Parameter(torch.tensor(0.0))

    def forward(self, dgi_loss, att_loss):
        lambda_dgi = self.dgi_min + (self.dgi_max - self.dgi_min) * torch.sigmoid(self.raw_dgi)
        lambda_att = 1 - lambda_dgi

        total_loss = (
            lambda_dgi * dgi_loss
            + lambda_att * att_loss
        )

        return total_loss, lambda_dgi.detach(), lambda_att.detach()

# class EdgePriorGate(nn.Module):
#     def __init__(self, edge_dim=6, hidden_dim=32):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(5, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def forward(self, edge_attr):
#         # edge_attr: [cell_weight, LR共表达, LR_ID, w1, w2, w3]
#         edge_cont = edge_attr[:, [0, 1, 3, 4, 5]]
#
#         # log1p 更适合乘性权重关系
#         edge_cont = torch.log1p(torch.clamp(edge_cont, min=0.0))
#
#         return self.net(edge_cont).squeeze(-1)


def train_CellNEST(args, data_loader, in_channels, pseudo_label_path=''):

    """Add Statement of Purpose
    Args: [to be]
           
    Returns: [to be]

    """
    loss_curve = np.zeros((args.num_epoch//100+1))
    loss_curve_counter = 0
################################
    metrics_records = []
    best_loss_metrics = None

    pseudo_edge_labels = None
    if pseudo_label_path is not None and pseudo_label_path != '':
        f = gzip.open(pseudo_label_path, 'rb')
        pseudo_edge_labels = pickle.load(f)
        pseudo_edge_labels = np.asarray(pseudo_edge_labels)
        print("Loaded pseudo edge labels from:", pseudo_label_path)
        print("Num labeled edges:", np.sum(pseudo_edge_labels >= 0))
#################################
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    DGI_model = DeepGraphInfomax(
        hidden_channels=args.hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=args.hidden, heads=args.heads, dropout = args.dropout),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),#所有节点求均值，表示“整张真实图的全局表示”
        corruption=corruption).to(device)
    loss_weighter = BoundedSumToOneLossWeight().to(device)
    #print('initialized DGI model')
    #DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=0.005, weight_decay=5e-4)
    # DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=args.lr_rate) #1e-5)#5 #6 #DGI_optimizer = torch.optim.RMSprop(DGI_model.parameters(), lr=1e-5)
    DGI_optimizer = torch.optim.Adam(list(DGI_model.parameters()) + list(loss_weighter.parameters()),lr=args.lr_rate)
    DGI_filename = args.model_path+'DGI_'+ args.model_name  +'.pth.tar'

    if args.load == 1:
        print('loading model')
        checkpoint = torch.load(DGI_filename)
        DGI_model.load_state_dict(checkpoint['model_state_dict'])
        DGI_model.to(device)
        DGI_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_weighter.load_state_dict(checkpoint['loss_weighter_state_dict'])
        epoch_start = checkpoint['epoch']
        min_loss = checkpoint['loss']
        '''
        for state in DGI_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        ''' 
        print('min_loss was %g'%min_loss)
    else:
        print('Saving init model state ...')
        torch.save({
            'epoch': 0,
            'model_state_dict': DGI_model.state_dict(),
            'optimizer_state_dict': DGI_optimizer.state_dict(),
            'loss_weighter_state_dict': loss_weighter.state_dict(),
            #'loss': loss,
            }, args.model_path+'DGI_init_model_optimizer_'+ args.model_name  + '.pth.tar')
        min_loss = 10000
        epoch_start = 0
        
    import datetime
    start_time = datetime.datetime.now()

    #print('training starts ...')
    for epoch in range(epoch_start, args.num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        DGI_all_loss = []

        for data in data_loader:
            data = data.to(device)
            _ = DGI_model.encoder(data)
            real_att = DGI_model.encoder.attention_scores_mine_unnormalized.detach().clone()
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            # real_att = DGI_model.encoder.attention_scores_mine_unnormalized
            fake_edge_attr = make_hard_fake_edge_attr(edge_attr=data.edge_attr, corrupt_ratio=0.5)
            _ = DGI_model.encoder(data,edge_attr_override=fake_edge_attr.detach())
            fake_att = DGI_model.encoder.attention_scores_mine_unnormalized
            att_loss = attention_contrastive_loss(real_att=real_att,fake_att=fake_att,margin=0.05 )
            _ = DGI_model.encoder(data, edge_attr_override=None)
            loss, lambda_dgi, lambda_att = loss_weighter( dgi_loss=DGI_loss,att_loss=att_loss)

            loss.backward()
            DGI_all_loss.append(loss.item())
            DGI_optimizer.step()


        if ((epoch)%100) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))
            loss_curve[loss_curve_counter] = np.mean(DGI_all_loss)
            loss_curve_counter = loss_curve_counter + 1

            """
            伪标签评估指标
            """
            ######
            current_loss = np.mean(DGI_all_loss)
            with torch.no_grad():
                edge_score_tensor = torch.sigmoid(DGI_model.encoder.attention_scores_mine_unnormalized)

                if edge_score_tensor.dim() > 1:
                    edge_score_tensor = edge_score_tensor.mean(dim=1)

                edge_score_mean = float(edge_score_tensor.mean().detach().cpu())
                edge_score_std = float(edge_score_tensor.std().detach().cpu())

            # 默认空指标
            att_metrics = {
                "ATT AUROC": np.nan,
                "ATT AP": np.nan,
                "ATT AUPRC": np.nan,
                "ATT ACC": np.nan,
                "ATT Precision": np.nan,
                "ATT Recall": np.nan,
                "ATT F1": np.nan,
            }

            # ===== 每轮评估 =====
            if pseudo_edge_labels is not None:
                # -------- 1) attention 分数 --------
                # current_att_scores = DGI_model.encoder.attention_scores_mine[1]
                current_att_scores = DGI_model.encoder.attention_scores_mine_unnormalized
                current_att_scores = torch.sigmoid(current_att_scores)
                current_att_scores = current_att_scores.detach().cpu().numpy()

                # shape [num_edges, 1] 或 [num_edges, heads] -> 压成 [num_edges]
                if len(current_att_scores.shape) > 1:
                    current_att_scores = current_att_scores.reshape(current_att_scores.shape[0], -1)
                    current_att_scores = current_att_scores.mean(axis=1)

                att_metrics, att_threshold = evaluate_pseudo_edge_scores(
                    scores=current_att_scores,
                    labels=pseudo_edge_labels,
                    threshold=None,
                    prefix="ATT"
                )

                # -------- 2) decoder 分数 --------
                # pos_z 是当前 epoch 最后一个 batch 的正图节点 embedding
                # current_dec_scores = edge_decoder_scores(pos_z, data.edge_index)
                # current_dec_scores = current_dec_scores.detach().cpu().numpy()
                #
                # current_dec_scores = DGI_model.encoder.edge_scores
                # current_dec_scores = current_dec_scores.detach().cpu().numpy()
                #
                # dec_metrics, dec_threshold = evaluate_pseudo_edge_scores(
                #     scores=current_dec_scores,
                #     labels=pseudo_edge_labels,
                #     threshold=None,
                #     prefix="DEC"
                # )


            # ===== 记录每轮结果 =====
            metrics_records.append({
                "Epoch": epoch + 1,
                "Loss": float(current_loss),
                "DGI_loss": float(DGI_loss.item()),
                "Att_loss": float(att_loss.item()),
                "Weight_DGI": float(lambda_dgi),
                "Weight_Att": float(lambda_att.item()),
                "EdgeScoreMean": edge_score_mean,
                "EdgeScoreStd": edge_score_std,
                **att_metrics
            })

            # 每轮都写一次 csv
            metrics_df = pd.DataFrame(metrics_records)
            metrics_df.to_csv(
                args.model_path + 'DGI_' + args.model_name + '_pseudo_metrics.csv',
                index=False
            )

            # ===== 最优 loss 时保存模型 + embedding + attention =====
            if current_loss < min_loss:
                min_loss = current_loss

                best_loss_metrics = {
                    "Epoch": "BestLoss",
                    "Loss": float(current_loss),
                    **att_metrics
                }


                ######## save the current model state ########los最小时候的模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DGI_model.state_dict(),
                    'optimizer_state_dict': DGI_optimizer.state_dict(),
                    'loss_weighter_state_dict': loss_weighter.state_dict(),
                    'loss': min_loss,
                    }, DGI_filename)

                ##################################################
                # save the node embedding
                X_embedding = pos_z #embeding
                X_embedding = X_embedding.cpu().detach().numpy()
                X_embedding_filename =  args.embedding_path + args.model_name + '_Embed_X' #.npy
                with gzip.open(X_embedding_filename, 'wb') as fp:  
                    pickle.dump(X_embedding, fp)
                                    
                # save the attention scores
                X_attention_index = DGI_model.encoder.attention_scores_mine[0]
                X_attention_index = X_attention_index.cpu().detach().numpy()

                # layer 1
                X_attention_score_normalized_l1 = DGI_model.encoder.attention_scores_mine_l1[1]
                X_attention_score_normalized_l1 = X_attention_score_normalized_l1.cpu().detach().numpy()
                # layer 1 unnormalized
                X_attention_score_unnormalized_l1 = DGI_model.encoder.attention_scores_mine_unnormalized_l1
                X_attention_score_unnormalized_l1 = X_attention_score_unnormalized_l1.cpu().detach().numpy()

                # layer 2
                X_attention_score_normalized = DGI_model.encoder.attention_scores_mine[1]
                X_attention_score_normalized = X_attention_score_normalized.cpu().detach().numpy()
                # layer 2 unnormalized
                X_attention_score_unnormalized = DGI_model.encoder.attention_scores_mine_unnormalized
                X_attention_score_unnormalized = X_attention_score_unnormalized.cpu().detach().numpy()

                print('making the bundle to save')
                X_attention_bundle = [X_attention_index, X_attention_score_normalized_l1, X_attention_score_unnormalized, X_attention_score_unnormalized_l1, X_attention_score_normalized]
                X_attention_filename =  args.embedding_path + args.model_name + '_attention' #.npy
                # np.save(X_attention_filename, X_attention_bundle) # this is deprecated
                with gzip.open(X_attention_filename, 'wb') as fp:  
                    pickle.dump(X_attention_bundle, fp)

                logfile=open(args.model_path+'DGI_'+ args.model_name+'_loss_curve.csv', 'wb')
                np.savetxt(logfile,loss_curve, delimiter=',')
                logfile.close()

                #print(DGI_model.encoder.attention_scores_mine_unnormalized_l1[0:10])

#            if ((epoch)%60000) == 0:
#                DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)  #5 #6

    end_time = datetime.datetime.now()

    print('Training time in seconds: ', (end_time-start_time).seconds)

    checkpoint = torch.load(DGI_filename)
    DGI_model.load_state_dict(checkpoint['model_state_dict'])
    DGI_model.to(device)
    DGI_model.eval()
    print("debug loss")
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            print("Best model recomputed DGI loss:", DGI_loss.item())
    DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
    print("debug loss latest tupple %g"%DGI_loss.item())

    # ===== 训练结束后，把最优 loss 对应指标追加到最后 =====
    if len(metrics_records) > 0:
        metrics_df = pd.DataFrame(metrics_records)

        if best_loss_metrics is not None:
            best_row_df = pd.DataFrame([best_loss_metrics])
            metrics_df = pd.concat([metrics_df, best_row_df], ignore_index=True)

        metrics_df.to_csv(
            args.model_path + 'DGI_' + args.model_name + '_pseudo_metrics.csv',
            index=False
        )
        print("Saved epoch-wise pseudo metrics with BestLoss row.")

    return DGI_model

