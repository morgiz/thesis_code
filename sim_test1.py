import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

from cpt_func import *
from bn import * 

import numpy as np
import pandas as pd

def store_leaf_states(leaf_node_plt_states, leaf_states):
    for leaf in leaf_states:
        leaf_node_plt_states[leaf].append(leaf_states[leaf])

ES = 'ES'           #EngineSpeed
FSF = 'FSF'         #FuelSystemFailure
tau = 'tau'         #Torque
FC = 'FC'           #FuelConsumption
OP = 'OP'           #OilPressure
LSF = 'LSF'         #LubricationSystemFailure
WT = 'WT'           #WaterTemperature
CSF = 'CSF'         #CoolingSystemFailure
AMF = 'AMF'         #AirMassFlow
IESF = 'IESF'       #IntakeAndExhaustSystemFailure
IMAP = 'IMAP'       #IntakeManifoldAirPressure
TSS = 'TSS'         #TimeSinceService
MEF = 'MEF'         #MainEngineFailure
AP = 'AP'           #AvailablePower
Wi = 'Wi'           #WindConditions
Wa = 'Wa'           #Waves
Cu = 'Cu'           #Current
MD = 'MD'           #MissionDemand
ED = 'ED'           #EnvironmentalDemand
PN = 'PN'           #PowerNecessary
NEAP = 'NEAP'       #NotEnoughAvailablePower
EM = 'EM'           #EngineMode
D2L = 'D2L'          #DistanceToLand/Obstacle
SPT = 'SPT'           #SpeedSetPoint
WM = 'WM'           #WantedMode

high = 3
low = 1
# parent_node, child_node, weight
arc_table = np.array([
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [TSS,  MEF,  high],
    [FSF,  MEF,  high],
    [LSF,  MEF,  low],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [MD,   PN,   high],
    [ED,   PN,   low],
    [Wi,   ED,   high],
    [Cu,   ED,   low],
    [D2L, ED, high],
    [SPT, MD, low ],
    [WM, MD, high ]
])

net, leaf_states = generate_bn(arc_table)

#Childs to supervise values for in the simulation
child_map = {"AP": [], "PN":[], "NEAP":[], "MEF":[]}

#Make a dictionary that save leaf node states for each time step. 
leaf_node_plt_states = {k: [] for k in leaf_states.keys()}

#Order list for leaf nodes
#leaf_node_order = list(leaf_states.keys())
leaf_node_order = [ES, tau, FC, OP, WT, IMAP, AMF, TSS, WM, SPT, Wi, Cu, D2L] 
num_leaf_nodes = len(leaf_node_order)

simulation_steps = (num_leaf_nodes + 2)*5  #5 timesteps for each node, plus 5 extra at the beginning and end.
time = 0 
node_i = 0
current_node = leaf_node_order[node_i]
first_switch = 1

init_state = 2
test_state = 0

while time < simulation_steps: 
    store_leaf_states(leaf_node_plt_states, leaf_states)

    #Check BN values and prepare for plot
    output = BeliefPropagation(net)
    for child in child_map.keys(): 
        q = output.query(variables=[child], evidence=leaf_states)
        child_map[child].append(q.values)

    if time % 5 == 0 and time >= 4 and first_switch == 1:
        leaf_states[current_node] = test_state 
        first_switch = 0

    elif time % 5 == 0  and first_switch != 1:
        if node_i != len(leaf_node_order)-1 : 
            leaf_states[current_node] = init_state

            node_i += 1 
            current_node = leaf_node_order[node_i]

            leaf_states[current_node] = test_state

        else: #Last node has five time-steps
            leaf_states[current_node] = init_state

    time += 1

#Plot results
fig1, axs1 = plt.subplots(num_leaf_nodes, 1, figsize=(10, num_leaf_nodes), layout = 'constrained', sharex=True)
for i, leaf in enumerate(leaf_node_order):
    axs1[i].plot(leaf_node_plt_states[leaf], marker='o')
    axs1[i].set_title(f'State of {leaf}')
    axs1[i].set_yticks([0, 1, 2])
    axs1[i].set_yticklabels(['Worst', 'Intermediate', 'Best'])
    axs1[i].grid(True)
fig1.supxlabel('Time step')
fig1.supylabel('State')
fig1.subplots_adjust(hspace = 0.3)

#Plot for child nodes 
fig2, axs = plt.subplots(len(child_map), 1, figsize=(10, 2*len(child_map)), sharex=True)
for i, child in enumerate(child_map.keys()):
    child_states = np.array(child_map[child])
    axs[i].plot(child_states[:, 0], marker='o')#, label='State 0')
    axs[i].plot(child_states[:, 1], marker='o')#, label='State 1')
    axs[i].plot(child_states[:, 2], marker='o')#, label='State 2')
    axs[i].set_title(f'Probabilities of {child} states')
    axs[i].set_ylim(0, 1)
    axs[i].grid(True)
fig2.supxlabel('Time step')
fig2.supylabel('Probability')
fig2.set_label(['Worst', 'Intermediate', 'Best'])
fig2.subplots_adjust(hspace = 0.3)
#fig2.set_layout_engine('tight')

#plt.show()

fig1.savefig('plots/sim_test1/leaf_plots.svg', format='svg')
fig2.savefig('plots/sim_test1/child_plots.svg', format='svg')