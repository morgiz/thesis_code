import numpy as np 
import itertools
import pandas as pd

import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation


# === Templates ===
# template[i][j]: i = parent state (worst→best), j = child state (worst→best)
template_state_high = np.array([
    [0.9, 0.09, 0.01],
    [0.05, 0.9, 0.05],
    [0.01, 0.09, 0.9]
])

template_state_low = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6]
])

num_states_parent = 3
num_states_child = 3
state_labels = ["Worst", "Intermediate", "Best"]


#opprett cpt for alle noder og skriv de til fil.
#Skal kun kalles når cptene må lages eller oppdateres. 
def define_cpt_files(arc_table): 
    high = 3
    low = 1

    sum_weights = np.sum(arc_table[:,2].astype(int)) 
    #print(f"Sum weights: {sum_weights}")
    high_soi = high / sum_weights
    low_soi = low / sum_weights

    #print(f"High SOI: {high_soi}, Low SOI: {low_soi}")
    #high_soi = 0.051724138
    #low_soi = 0.017241379

    #Sett child node til den noden man vil regne ut cpt for. 
    #Parent_map blir konstruert med utgangspunkt i arc'ene definert over. 
    #Finner hver distincte child, og lager cpt'ene for denne. 
    for distinct_child in set(arc_table[:,1]): 
        print(distinct_child)
        #distinct_child = np.reshape(distinct_child, shape=-1)
        #print(distinct_child)
        child_node = distinct_child
        parent_map = {}
        # Works for one or more parents, create parent map: 
        for arc in arc_table:
            if child_node == arc[1]: 
                parent_map[str(arc[0])] = int(arc[2])
        #parent_map = {"A": high_soi} 
        #parent_map = {"A": high_soi, "B": low_soi}
        parent_list = list(parent_map.keys())
        #print(child_node)
        #print(parent_map)

        # === Parent state combinations ===
        parent_state_combos = list(itertools.product(range(num_states_parent), repeat=len(parent_list)))
        num_combinations = num_states_parent ** len(parent_list)

        # === Initialize CPT ===
        child_cpt = np.zeros((num_states_child, num_combinations))

        # === Populate CPT ===
        for combo_index, parent_states in enumerate(parent_state_combos):
            total_influence = np.zeros(num_states_child)

            for p_idx, parent_name in enumerate(parent_list):
                soi = parent_map[parent_name]
                parent_state = parent_states[p_idx]

                template = template_state_high if soi == high_soi else template_state_low

                # IMPORTANT: parent state is the row → use template[parent_state, :]
                total_influence += soi * template[parent_state, :]

            #Normalization
            total_influence /= np.sum(total_influence)
            child_cpt[:, combo_index] = total_influence
        
        filename = "cpts/"+str(distinct_child)+"_cpt.txt"
        df = pd.DataFrame(child_cpt)
        df.to_csv(filename, index = False)

def leaf_node_cpts(p_dict): #Parent dict as input in bn.py
    for node in p_dict: 
        if p_dict[node][0] == 0: 
            filename = "cpts/"+str(node)+"_cpt.txt"
            df = pd.DataFrame([0.3,0.3,0.4]) #Placeholder for leaf node probabilities.
            df.to_csv(filename, index = False)


def get_cpt(child_node): 
    #les inn cpt fra fil for gitt child node. 
    filename = "cpts/"+str(child_node)+"_cpt.txt"
    df = pd.read_csv(filename)
    cpt = df.to_numpy()
    return cpt[:,:]

def define_node_cpts(node, node_parent_dict):
    num_parents = node_parent_dict[node][0]
    parent_list = node_parent_dict[node][1]
    for node in node_parent_dict.keys(): 
        evidence = []
    for i in range(num_parents):
        evidence.append(num_states_parent) 
    node_lower = node.lower()
    cpt = get_cpt(node)

    if num_parents != 0: #node is not leafnode
        exec(f"""cpt_{node_lower} = TabularCPD(
        variable=node,
        variable_card=3,
        values=cpt,   
        evidence=parent_list, 
        evidence_card=evidence 
        )""")

    elif num_parents==0 : #node is leafnode
        exec(f"""cpt_{node_lower} = TabularCPD(
        variable=node,
        variable_card=3,
        values=cpt  
        )""")
    #exec(f"return cpt_{node_lower}")