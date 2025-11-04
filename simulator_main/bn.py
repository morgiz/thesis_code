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

def generate_bn():
    net = BBN()

    high = 3
    low = 1

    ES = 'ES'           #EngineStatus
    FSF = 'FSF'         #FuelSystemFailure
    tau = 'tau'         #TurbochargerStatus
    FC = 'FC'           #FuelControl
    OP = 'OP'           #OilPressure
    LSF = 'LSF'         #LubricationSystemFailure
    WT = 'WT'           #WaterTemperature
    CSF = 'CSF'         #CoolingSystemFailure
    AMF = 'AMF'         #AirManagementFailure
    IESF = 'IESF'       #IntakeEngineSystemFailure
    IMAP = 'IMAP'       #IntakeManifoldAirPressure
    TSS = 'TSS'         #TurbochargerStatusSensor
    MEF = 'MEF'         #MainEngineFailure
    EF = 'EF'           #EngineFailure
    AP = 'AP'           #AvailablePower
    T = 'T'             #Traffic
    SitCom = 'SitCom'   #SituationalComplexity
    COMMS = 'COMMS'     #Communications
    Wi = 'Wi'           #WindConditions
    Wa = 'Wa'           #WeatherConditions
    WC = 'WC'           #WeatherComplexity
    MD = 'MD'           #MeteorologicalData
    NP = 'PN'           #NecessaryPower
    NEAP = 'NEAP'       #NecessaryEngineAvailablePower
    EM = 'EM'           #EngineMode

    # parent_node, child_node, vekt
    #Tilsvare tabell 3.4 i rapporten
    #Ønsker å bruke denne tabellen som utgangspunkt for cptene, samt å sette opp BN
    arc_table = np.array([
        [ES, FSF, low],
        [tau,FSF, high],
        [FC, FSF, high],
        [OP, LSF, high],
        [WT, CSF, high],
        [AMF, IESF, high],
        [IMAP, IESF, low],
        [TSS, MEF, high],
        [FSF, MEF, high],
        [LSF, MEF, high],
        [CSF, MEF, high],
        [IESF, MEF, low],
        [EM, MEF, high],
        [MEF, AP, high],
        [EF, AP, high],
        [AP, NEAP, high],
        [T, SitCom, high],
        [COMMS, SitCom, low],
        [Wi, WC, high],
        [Wa, WC, low],
        [SitCom, NP, high],
        [WC, NP, low],
        [MD, NP, low],
        [NP, NEAP, high],
    ])
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
        print(node, parent_list, evidence)

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

    return net, leafnode_state_dict


def print_bn(net: BBN): 
    # net.fit() sjekker tilstanden til alle cpt basert på evidence. 
    printable = net.to_daft(node_params={globals()["EM"]:{'shape': 'rectangle'}})
    printable.show()

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