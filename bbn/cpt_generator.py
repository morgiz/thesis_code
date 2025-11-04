import numpy as np 
import itertools
import pandas as pd

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

def define_cpts(): 
    #opprett cpt for alle noder og skriv de til fil.

    high = 3
    low = 1

    # parent_node, child_node, vekt
    #Tilsvare tabell 3.4 i rapporten
    arc_table = np.array([
        ["ES", "FSF", low],
        ["tau","FSF", high],
        ["FC", "FSF", high],
        ["OP", "LSF", high],
        ["WT", "CSF", high],
        ["AMF", "IESF", high],
        ["IMAP", "IESF", low],
        ["TSS", "MEF", high],
        ["FSF", "MEF", high],
        ["LSF", "MEF", high],
        ["CSF", "MEF", high],
        ["IESF", "MEF", low],
        ["EM", "MEF", high],
        ["MEF", "AP", high],
        ["EF", "AP", high],
        ["AP", "NEAP", high],
        ["T", "SitCom", high],
        ["COMMS", "SitCom", low],
        ["Wi", "WC", high],
        ["Wa", "WC", low],
        ["SitCom", "PN", high],
        ["WC", "PN", low],
        ["MD", "PN", low],
        ["PN", "NEAP", high],
    ])

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


define_cpts()