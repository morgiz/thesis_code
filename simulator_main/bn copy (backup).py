import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

from cpt_func import *




#Code som genrerer BN, CPTs og visualiserer nettet. 
#Kan kalles av en simulering og returnerer et BN objekt.

def generate_bn():
    net = BBN()
    """
    NEAP = 'LackOfPower'
    AP = 'AvailablePower'
    NP = 'NecessaryPower' 
    MEF = 'MainEngineFailure'
    SitCom = 'SituationalComplexity'
    EM = 'EngineMode' 
    WC = 'WeatherConditions'

    nodelist = [NEAP, AP, MEF, NP, SitCom, WC, EM] #MÅ ikke egentlig legge til nodene her, for det gjøres når man legger til edges
    net.add_nodes_from(nodelist)
    """
    #edgelist = [(AP,NEAP), (MEF,AP), (EM, AP), (SitCom, NP), (NP, NEAP), (WC,NP)]
    #net.add_edges_from(edgelist)


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


    #Genereate map with parent node for each child node.
    #Parent dict structure: {child_node: [num_parents, [parent1, parent2, ...]]}
    parent_dict = get_node_parent_dict(arc_table)
    
    #define cpt for each node based on parent dict
    for node in parent_dict.keys():
        define_node_cpts(node, parent_dict)

    #Define CPTs
    #Får tilstandene 0,1,2 som tilsvarer lav, middels og høy "severity". 
    #samme CPT for MEF of SitCom
    cpt_mef = TabularCPD(
        variable=MEF,
        variable_card=3,
        values=[[0.03],[0.9],[0.07]]
    )

    cpt_em = TabularCPD(
        variable = EM, 
        variable_card=3, 
        values = [[0.9],[0.08],[0.02]]
    )


    cpt_SitCom = TabularCPD(
        variable=SitCom,
        variable_card=3,
        values=[[0.03],[0.9],[0.07]]
    )

    #Samme CPT for AP og NP
    cpt_ap = TabularCPD(
        variable=AP,
        variable_card=3,
        #tilfeldige verdier under
        values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],  
                [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     
                [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],   

        evidence=[MEF,EM], #Which nodes act as evidence
        evidence_card=[3,3] #number of states in evidence nodes
    )

    cpt_np = TabularCPD(
        variable=NP,
        variable_card=3,
        #Tilfeldige verdier i cpt
        values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],     
                [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     
                [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],     
        evidence=[SitCom,WC],
        evidence_card=[3,3]
    )
    cpt_wc = TabularCPD(
        variable=WC,
        variable_card=3,
        values=[[0.03],[0.9],[0.07]]
    )

    cpt_neap = TabularCPD(
        variable=NEAP,
        variable_card=3, # num of states

        #Fyller inn litt tilfeldige verdier under, bare for å ha noe å gå etter
            #AP  #     L       |       M        |       H      |
            #NP  # L   M    H  |  L    M     H  | L     M    H |
        values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],     #NEAP = low
                [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     #NEAP = medium
                [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],     #NEAP = High
        evidence=[AP, NP],
        evidence_card=[3,3]
    )
    net.add_cpds(cpt_ap, cpt_np, cpt_neap, cpt_mef, cpt_SitCom, cpt_wc, cpt_em)

    net.check_model()
    leafnode_state_dict = {MEF:1, EM: 1 , SitCom:1, WC:0}
    return net


def print_bn(net): 
    # net.fit() sjekker tilstanden til alle cpt basert på evidence. 
    printable = net.to_daft(node_params={globals()["EM"]:{'shape': 'rectangle'}})

def get_node_parent_dict(arc_table):
    node_keys = np.unique(arc_table[:,0:2])
    node_keys = node_keys.tolist()
    node_parent_dict = {k: [0, []] for k in node_keys}
    for node in node_keys:
        for arc in arc_table: 
            if arc[1] in node_parent_dict: 
                node_parent_dict[arc[1]][0] +=1
                node_parent_dict[arc[1]][1].append(str(arc[0]))