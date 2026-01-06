#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


import torch
from torch.utils.data import Dataset, DataLoader
from healthiness_eval import NutritionData, NutritionMetric

def NutritionData(data_path):
    with open(data_path + '/course_who.txt', 'r') as f:
        course_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/course_fsa.txt', 'r') as f:
        course_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/meal_who.txt', 'r') as f:
        meal_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/meal_fsa.txt', 'r') as f:
        meal_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_who.txt', 'r') as f:
        user_who = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    with open(data_path + '/user_fsa.txt', 'r') as f:
        user_fsa = np.array(list(map(lambda s: tuple(float(i) for i in s[:-1].split('\t')), f.readlines())))
    return course_who, course_fsa, meal_who, meal_fsa, user_who, user_fsa

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items, self.num_ingredients = self.get_data_size()
        self.num_categories = 3

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()
        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        b_ingredient_graph = self.get_b_ingredient()
        u_ingredient_graph = self.get_u_ingredient()
        i_ingredient_graph = self.get_i_ingredient()

        # u_b_metapath_pairs, u_b_metapath_graph = self.get_ub_metapath()


        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph,
                      b_ingredient_graph, u_ingredient_graph, i_ingredient_graph
        ]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)

    
    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:4]


    def get_aux_graph(self, u_i_graph, b_i_graph, conf):
        u_b_from_i = u_i_graph @ b_i_graph.T
        u_b_from_i = u_b_from_i.todense()
        bn1_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:, bn1_window[0]:bn1_window[1]]

        b_b_from_i = b_i_graph @ b_i_graph.T
        b_b_from_i = b_b_from_i.todense()
        bn2_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        b_b_for_neg_sample = np.argsort(b_b_from_i, axis=1)[:, bn2_window[0]:bn2_window[1]]

        return u_b_for_neg_sample, b_b_for_neg_sample


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_graph

    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph

    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph

    def get_b_ingredient(self):
        with open(os.path.join(self.path, self.name, 'meta_data/bundle_ingredient.txt'), 'r') as f:
            b_ingredient_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_ingredient_pairs, dtype=np.int32)
        values = np.ones(len(b_ingredient_pairs), dtype=np.float32)
        b_ingredient_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_ingredients)).tocsr()

        print_statistics(b_ingredient_graph, 'B-Ingredient statistics')

        return b_ingredient_graph

    def get_i_ingredient(self):
        with open(os.path.join(self.path, self.name, 'meta_data/course_ingredient.txt'), 'r') as f:
            i_ingredient_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(i_ingredient_pairs, dtype=np.int32)
        values = np.ones(len(i_ingredient_pairs), dtype=np.float32)
        i_ingredient_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_items, self.num_ingredients)).tocsr()

        print_statistics(i_ingredient_graph, 'Item-Ingredient statistics')

        return i_ingredient_graph
    
    def get_u_ingredient(self):
        with open(os.path.join(self.path, self.name, 'meta_data/user_ingredient.txt'), 'r') as f:
            u_ingredient_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_ingredient_pairs, dtype=np.int32)
        values = np.ones(len(u_ingredient_pairs), dtype=np.float32)
        u_ingredient_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_ingredients)).tocsr()

        print_statistics(u_ingredient_graph, 'User-Ingredient statistics')

        return u_ingredient_graph

