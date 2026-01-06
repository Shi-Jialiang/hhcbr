import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import json
from collections import defaultdict
import dcor


def cal_bpr_loss(pred):
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)

    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss

def cal_healthy_bpr_loss_plus_fsa(
    pred, 
    positive_ids, 
    negative_ids, 
    fsa_path="/home/kazu/bundle/CrossCBR-master/datasets/MealRec+H/healthiness/bundle_fsa.txt"
):

    with open(fsa_path, "r") as f:
        fsa_list = [float(line.strip()) for line in f.readlines()]
    fsa_scores = torch.tensor(fsa_list, device=pred.device)   

    if pred.shape[1] > 2:
        negs = pred[:, 1:]                               
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)    
    else:
        negs = pred[:, 1].unsqueeze(1)                   
        pos = pred[:, 0].unsqueeze(1)                   

    pos_fsa = fsa_scores[positive_ids].unsqueeze(1).expand_as(negs)  
    neg_fsa = fsa_scores[negative_ids]                               

    sign_c = torch.where(pos_fsa < neg_fsa, 1.0, -1.0)               
    coef = torch.abs(pos_fsa - neg_fsa) / 12                       
    sign_weighted = sign_c * coef                                     

    health_adjusted_diff = sign_weighted * (pos - negs)

    loss = -torch.log(torch.sigmoid(health_adjusted_diff) + 1e-8)
    loss = torch.mean(loss)

    return loss

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class HHCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_ingredients = conf["num_ingredients"]  
        self.dataset_name = conf["dataset"]
        
        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph, self.b_ingredient_graph, self.u_ingredient_graph, self.i_ingredient_graph = raw_graph
        
        # generate the agg graph
        self.get_bundle_agg_graph_ori()
        self.get_bundle_agg_graph()

        # generate original graph 
        self.get_user_item_graph()
        self.get_user_item_graph_ori()

        self.get_user_bundle_graph()
        self.get_user_bundle_graph_ori()

        self.get_bundle_item_graph()
        self.get_bundle_item_graph_ori()

        # generate ehancement graph 
        self.get_bundle_level_ingredient_graph()
        self.get_bundle_level_ingredient_graph_ori()

        self.get_bundle_level_item_graph_ori()
        self.get_bundle_level_item_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

    
    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)
        self.user_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        if self.dataset_name == "MealRec+L" :
            self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature)
            self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
            nn.init.xavier_normal_(self.bundles_feature)
            self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
            nn.init.xavier_normal_(self.items_feature)
        elif self.dataset_name == "MealRec+H":
            self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature)
            self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
            nn.init.xavier_normal_(self.bundles_feature)
            self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
            nn.init.xavier_normal_(self.items_feature)

    ## original graph ##
    ### user-item graph ###
    def get_user_item_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]]) # sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.user_item_graph = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_user_item_graph_ori(self):
        ui_graph = self.ui_graph 
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]]) # sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))
        self.user_item_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)
    ### user-item graph ###

    ### bundle-item graph ###
    def get_bundle_item_graph(self):
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_item_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundle_item_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bundle_item_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    ### bundle-item graph ###

    ### user-bundle graph ###
    def get_user_bundle_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.user_bundle_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_user_bundle_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.user_bundle_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    ### user-bundle graph ###

    ### ehancement bundle graph ###
    ## ingredient ##
    def get_bundle_level_ingredient_graph(self):
        device = self.device
        modification_ratio = self.conf.get("bundle_level_ratio", 0.0)
        ub_graph = self.ub_graph

        ratio = 1.0 
        temp_graph = self.u_ingredient_graph.dot(self.b_ingredient_graph.T)  
        ub_graph = ub_graph + ratio * temp_graph             

        bundle_level_graph = sp.bmat([
            [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
            [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]
        ])

        if modification_ratio != 0:
            if self.conf.get("aug_type") == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix(
                    (values, (graph.row, graph.col)),
                    shape=graph.shape
                ).tocsr()

        self.bundle_ingredient_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundle_level_ingredient_graph_ori(self):
        device = self.device
        ub_graph = self.ub_graph

        ratio = 1.0
        temp_graph = self.u_ingredient_graph.dot(self.b_ingredient_graph.T) 
        ub_graph = ub_graph + ratio * temp_graph             


        bundle_level_graph = sp.bmat([
            [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
            [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]
        ])

        self.bundle_ingredient_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    ## ingredient ##

    ## item ##
    def get_bundle_level_item_graph(self):
        modification_ratio = self.conf.get("bundle_level_ratio", 0.0)
        device = self.device
        ub_graph = self.ub_graph
        ratio = 1.0  
        temp_graph = self.ui_graph.dot(self.bi_graph.T) 
        ub_graph = ub_graph + ratio * temp_graph             

        bundle_level_graph = sp.bmat([
            [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
            [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]
        ])

        if modification_ratio != 0:
            if self.conf.get("aug_type") == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix(
                    (values, (graph.row, graph.col)),
                    shape=graph.shape
                ).tocsr()

        self.bundle_item_user_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundle_level_item_graph_ori(self):
        device = self.device
        ub_graph = self.ub_graph
        ratio = 1.0
        temp_graph = self.ui_graph.dot(self.bi_graph.T)  
        ub_graph = ub_graph + ratio * temp_graph             

        bundle_level_graph = sp.bmat([
            [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
            [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]
        ])
        self.bundle_item_user_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    ## item ##
    ### ehancement bundle graph ###

    ### agg_graph ###
    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph 
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph 
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)
    ### agg_graph ###

    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: 
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature
    
    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature

    def propagate(self, test=False):
        #  ======================================== Main =====================================================
        #  ============================= user--item level propagation  =======================================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.user_item_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.user_item_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)


        #  ============================ user--bundle level propagation ========================================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.user_bundle_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.user_bundle_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)      

        #  ============================= user--ingredient--bundle propagation =================================
        if test:
            BL_users_ingredient_feature, BL_bundles_ingredient_feature = self.one_propagate(self.bundle_ingredient_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_ingredient_feature, BL_bundles_ingredient_feature = self.one_propagate(self.bundle_ingredient_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)    

        #  ============================= user--item--bundle propagation propagation ===========================
        if test:
            BL_users_item_feature, BL_bundles_item_feature = self.one_propagate(self.bundle_item_user_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_item_feature, BL_bundles_item_feature = self.one_propagate(self.bundle_item_user_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)                

        BL_bundles_feature = (BL_bundles_feature + BL_bundles_ingredient_feature + BL_bundles_item_feature)/3 
        BL_users_feature = (BL_users_feature + BL_users_ingredient_feature + BL_users_item_feature)/3 


        bundles_feature = [IL_bundles_feature, 
                           BL_bundles_feature,
                           BL_bundles_item_feature, 
                           BL_bundles_ingredient_feature
                           ]
                           
        users_feature = [IL_users_feature, 
                         BL_users_feature, 
                         BL_users_item_feature,
                         BL_users_ingredient_feature
                         ] 

        return users_feature, bundles_feature

    def cal_c_loss(self, pos, aug):

        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) 

        pos_score = torch.exp(pos_score / self.c_temp) 
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) 

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_loss(self, users_feature, bundles_feature, positive_ids, negative_ids):

        (IL_bundles_feature, 
         BL_bundles_feature, 
         BL_bundles_item_feature, 
         BL_bundles_ingredient_feature,
         ) = bundles_feature 
                
        (IL_users_feature, 
         BL_users_feature, 
         BL_users_item_feature,
         BL_users_ingredient_feature,
         ) = users_feature

        pred = (
        torch.sum(IL_users_feature * IL_bundles_feature, 2) 
        + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        + torch.sum(BL_users_item_feature * BL_bundles_item_feature, 2) 
        + torch.sum(BL_users_ingredient_feature * BL_bundles_ingredient_feature, 2) 
        ) 
        bpr_loss = cal_bpr_loss(pred)

        ### IL ###
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        ### BU-Ehancement ###
        item_u_cross_view_cl= self.cal_c_loss(BL_users_item_feature, BL_users_feature)
        item_b_cross_view_cl = self.cal_c_loss(BL_bundles_item_feature, BL_bundles_feature) 

        ingredient_u_cross_view_cl = self.cal_c_loss(BL_users_ingredient_feature, BL_users_feature)
        ingredient_b_cross_view_cl = self.cal_c_loss(BL_bundles_ingredient_feature, BL_bundles_feature) 

        c_losses = [u_cross_view_cl, b_cross_view_cl, item_u_cross_view_cl, item_b_cross_view_cl, 
                    ingredient_u_cross_view_cl, ingredient_b_cross_view_cl]
        
        c_loss = sum(c_losses) / len(c_losses) 

        #fsa
        if self.dataset_name == "MealRec+L":
            # 0.3 2.0
            healthy_loss = cal_healthy_bpr_loss_plus_fsa(pred, positive_ids, negative_ids, fsa_path="/home/kazu/bundle/CrossCBR-master/datasets/MealRec+L/healthiness/bundle_fsa.txt")
            bpr_loss = 0.3*bpr_loss + 2.0*healthy_loss
            c_loss = 1.1*c_loss

        elif self.dataset_name == "MealRec+H":
            healthy_loss = cal_healthy_bpr_loss_plus_fsa(pred, positive_ids, negative_ids, fsa_path="/home/kazu/bundle/CrossCBR-master/datasets/MealRec+H/healthiness/bundle_fsa.txt")
            # 1.9 3.0
            bpr_loss = 1.9*bpr_loss + 3.0*healthy_loss 
            c_loss = 1.4*c_loss

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_user_item_graph()    
            self.get_bundle_item_graph()
            self.get_user_bundle_graph()

            self.get_bundle_level_ingredient_graph()
            self.get_bundle_level_item_graph()

            self.get_bundle_agg_graph()


        users, bundles = batch
        users_feature, bundles_feature, = self.propagate()
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        positive_ids, negative_ids = bundles[:, 0], bundles[:, 1:]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, positive_ids, negative_ids)

        return bpr_loss, c_loss 


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result

        (bundles_feature_atom,
         bundles_feature_non_atom, 
         BL_bundles_item_feature, 
         BL_bundles_ingredient_feature,
         ) = bundles_feature 
        
        (users_feature_atom, 
         users_feature_non_atom, 
         BL_users_item_feature, 
         BL_users_ingredient_feature,
         ) = [i[users] for i in users_feature] 
              
        scores = (
        torch.mm(users_feature_atom, bundles_feature_atom.t()) 
        + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        + torch.mm(BL_users_item_feature, BL_bundles_item_feature.t()) 
        + torch.mm(BL_users_ingredient_feature, BL_bundles_ingredient_feature.t())
        )

        return scores 
