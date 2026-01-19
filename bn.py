import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

import daft

from cpt_func import *

#Code som genrerer BN, CPTs og visualiserer nettet. 
#Kan kalles av en simulering og returnerer et BN objekt.

def generate_bn(arc_table: np.array):
    net = BBN()

    define_cpt_files(arc_table)

    #Generate nodes and edges based on arc_table
    edgelist = get_edgelist(arc_table)
    net.add_edges_from(edgelist)

    #Genereate map with parent node for each child node.
    #Parent dict structure: {child_node: [num_parents, [parent1, parent2, ...]]}
    node_parent_dict = get_node_parent_dict(arc_table)
    leaf_node_cpts(node_parent_dict)
    
    #define cpt for each node based on parent dict
    for node in node_parent_dict.keys():
        num_parents = node_parent_dict[node][0]
        parent_list = node_parent_dict[node][1]
        evidence = [num_states_parent] * num_parents
        node_lower = node.lower()
        cpt = get_cpt(node)
        node_cpt_list = []

        if num_parents != 0:  # node is NOT a leafnode
            cpd = TabularCPD(
                variable=node,
                variable_card=3,
                values=cpt,
                evidence=parent_list,
                evidence_card=evidence
            )
        else:  # node is leafnode
            cpd = TabularCPD(
                variable=node,
                variable_card=3,
                values=cpt
        )
        net.add_cpds(cpd)
    
    #net.add_cpds(node_cpt_list)

    net.check_model()
    leafnode_state_dict = {}
    for node in node_parent_dict.keys(): 
        if node_parent_dict[node][0] == 0:  # leafnode
            leafnode_state_dict[node] = 2  # initializes all nodes to best state
            #leafnode_state_dict[node] = 0  # initializes all nodes to worst state

    return net, leafnode_state_dict

def get_node_parent_dict(arc_table):
    node_keys = np.unique(arc_table[:,0:2])
    node_keys = node_keys.tolist()
    node_parent_dict = {k: [0, []] for k in node_keys}
    for arc in arc_table: 
        if arc[1] in node_parent_dict: 
            node_parent_dict[arc[1]][0] +=1
            node_parent_dict[arc[1]][1].append(str(arc[0]))

    return node_parent_dict

def get_edgelist(arc_table):
    edgelist = []
    for arc in arc_table: 
        edgelist.append((str(arc[0]), str(arc[1])))
    return edgelist