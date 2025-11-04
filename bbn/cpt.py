import numpy as np
import math as m

high_soi = 0.051724138
low_soi = 0.017241379
#inneholder node, og hvor kraftig påvirkning arc'en har på child noden
parent_map = {"A":high_soi, "B":low_soi} #hhv lav og høy SoI
parent_list = ["A", "B"] #Use a list have a structure to iterate over.
#parent_map = {"A":0.051724138} #hhv lav og høy SoI
#parent_list = ["A"]
num_states_child = 3
num_states_parent = 3
child_cpt = np.zeros((num_states_parent,num_states_child**len(parent_map)))
child_cpt2 = np.zeros((num_states_parent,num_states_child**len(parent_map)))

template_state_high = [[0.9,0.09,0.01],
                       [0.05,0.9,0.05],
                       [0.01,0.09,0.9]]

#template_state_low = [[0.6,0.3,0.1], [0.2,0.6,0.2],[0.1,0.3,0.6]]

template_state_low = [[0,0,0],
                      [0,0,0],
                      [0,0,0]]
x = 0
depth = 0


def loop(child_cpt, depth, child_state): 
    for i in range (num_states_parent** (len(parent_list))):
        if parent_map[parent_list[depth]] == high_soi:
            
            template_state = globals()["template_state_high"]
        else: 
            template_state = globals()["template_state_low"]
        child_cpt[child_state][i] += parent_map[parent_list[depth]] * template_state[child_state][i % (len(parent_list) - depth)]
        if depth < len(globals()["parent_list"])-1 :
        #stuff
            loop(child_cpt, depth+1, child_state)
    return child_cpt

#int(m.floor( m.pow(i, 1/(len(parent_map)-depth)) ))


for i in range (num_states_child): 
    cpt = loop(child_cpt, depth, i)
    #for y in range(num_states_parent ** len(parent_map)): 
    #    for p in range(len(parent_list)): 
    #        if parent_map[parent_list[p]] == high_soi:
    #        
    #            template_state = template_state_high
    #        else: 
    #            template_state = template_state_low
    #        #For hver forelder må jeg legge til template verdien * vekten
    #       child_cpt[p][i][y] += parent_map[parent_list[p]] * template_state[i][y%num_states_parent]    

#For a node with a single parent the resulting 3x3 matrix is on the form: 
#                   child state
#parent state   Worst   Inter   Best
# Worst         x_1      x_2     x_3
# Inter         x_4      x_5     x_6
# Best          x_7      x_8     x_9
#
# I use the Low, normal, and high states in my leaf nodes, and the low state is not necessarily the worst. 
# I can still use the table provided above by just moving the relevant row to the right location. 
#print(child_cpt)
print(cpt)

"""
for i in range (num_states_child): 
    for y in range(num_states_parent ** len(parent_map)): 
        for parent in parent_list: 
            if parent_map[parent] == 0.017241379: 
                template_state = template_state_low
            else: 
                template_state = template_state_high 

            child_cpt[i][y] += parent_map[child] * template_state[i][int(np.sqrt(y))]
"""