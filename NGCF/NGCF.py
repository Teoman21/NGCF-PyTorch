'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})
        
        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
        
    def create_abpr_loss(self, users, pos_items, neg_items):
        """
        Adaptive Bayesian Personalized Ranking (ABPR) Loss
        """
        # Compute scores for positive and negative items
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        # Calculate the ranking difference
        score_diff = pos_scores - neg_scores

        # Compute weights dynamically based on difficulty (sigmoid as weight)
        weights = torch.sigmoid(neg_scores - pos_scores)

        # Compute the weighted BPR loss
        bpr_loss = -torch.mean(weights * torch.log(torch.sigmoid(score_diff)))

        # Regularization
        regularizer = torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2
        reg_loss = self.decay * regularizer / self.batch_size

        return bpr_loss + reg_loss, bpr_loss, reg_loss

    def create_hinge_loss(self, users, pos_items, neg_items, margin=1.0):
        """
        Pairwise Hinge Loss:
        L = max(0, margin - (s_u,i - s_u,j)) + regularization
        
        Args:
        - users: User embeddings
        - pos_items: Positive item embeddings
        - neg_items: Negative item embeddings
        - margin: Margin for hinge loss
        
        Returns:
        - Total loss
        - Hinge loss
        - Regularization loss
        """
        # Calculate scores
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        # Hinge loss
        hinge_loss = torch.mean(torch.relu(margin - (pos_scores - neg_scores)))

        # Regularization
        regularizer = torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2
        reg_loss = self.decay * regularizer / self.batch_size

        return hinge_loss + reg_loss, hinge_loss, reg_loss

    def drop_edge(self, adj, drop_rate=0.2):
        """
        Apply DropEdge by randomly removing a fraction of edges from the adjacency matrix.
        
        Args:
        - adj (torch.sparse.FloatTensor): The sparse adjacency matrix.
        - drop_rate (float): The fraction of edges to drop.
        
        Returns:
        - torch.sparse.FloatTensor: The modified adjacency matrix with edges dropped.
        """
        assert 0.0 <= drop_rate < 1.0, "Drop rate must be between 0 and 1."
        
        # Get the indices and values of the adjacency matrix
        indices = adj._indices()
        values = adj._values()

        # Compute the number of edges to keep
        num_edges = values.size(0)
        keep_prob = 1 - drop_rate
        keep_mask = torch.rand(num_edges, device=adj.device) < keep_prob

        # Filter the edges to keep
        new_indices = indices[:, keep_mask]
        new_values = values[keep_mask]

        # Create the new sparse tensor with dropped edges
        return torch.sparse.FloatTensor(new_indices, new_values, adj.size()).to(adj.device)

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    
    def forward(self, users, pos_items, neg_items, drop_flag=True):
        """
        Forward pass with DropEdge regularization.

        Args:
        - users: Batch of user indices.
        - pos_items: Batch of positive item indices.
        - neg_items: Batch of negative item indices.
        - drop_flag: Whether to apply DropEdge during training.

        Returns:
        - u_g_embeddings: User embeddings.
        - pos_i_g_embeddings: Positive item embeddings.
        - neg_i_g_embeddings: Negative item embeddings.
        """
        if drop_flag:
            A_hat = self.drop_edge(self.sparse_norm_adj, drop_rate=self.node_dropout)
        else:
            A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                            + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                            + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
