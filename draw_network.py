
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

from bn import *
from cpt_func import *

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
WM = 'WM'           #WantedMode
SPT = 'SPT'           #SpeedSetPoint
PNL = 'PNL'

high = 3
low = 1

bn_without_util = np.array([
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [TSS,  MEF,  high],
    [FSF,  MEF,  high],
    [LSF,  MEF,  high],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [PNL,  PN,   high],
    #[Wi, NP, low],
    #[T,NP, high],
    #[DL, NP, high]
])

bn_with_util = np.array([
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [TSS,  MEF,  high],
    [FSF,  MEF,  high],
    [LSF,  MEF,  high],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [MEF, "MefUtil", high],
    [EM, "MefUtil",high],
    [NEAP, "NeapUtil", high],
    [PN, "NpUtil", high],
    [EM, "NpUtil", high], 
    ["MefUtil", "TotUtil", high],
    ["NeapUtil", "TotUtil", high], 
    ["NpUtil", "TotUtil", high],
    [Wi, PN, low],
    ["MissionDemand", PN, high],
    ["EnvironmentDemand", PN, low],
    ["SpeedSetpoint", "MissionDemand", low],
    ["CurrentMode", "MissionDemand", high],
    [Wi, "EnvironmentDemand", low],
    [D2L, "EnvironmentDemand", high],
    #[T,PN, high],
    [D2L, PN, high]
])

mef_arcs = np.array([
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [FSF,  MEF,  high],
    [LSF,  MEF,  high],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [TSS,  MEF,  high],
])

bn_full_no_util = np.array([
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [FSF,  MEF,  high],
    [LSF,  MEF,  high],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [TSS,  MEF,  high],
    [MD, PN, high],
    [ED, PN, low],
    [SPT, MD, low],
    [WM, MD, high],
    [Wi, ED, low],
    [D2L, ED, high],
    [Cu, ED, low],
])

complete_bn = np.array([
    [ES,   FSF,  low],
    [tau,  FSF,  high],
    [FC,   FSF,  high],
    [OP,   LSF,  high],
    [WT,   CSF,  high],
    [AMF,  IESF, high],
    [IMAP, IESF, low],
    [TSS,  MEF,  high],
    [FSF,  MEF,  high],
    [LSF,  MEF,  high],
    [CSF,  MEF,  high],
    [IESF, MEF,  low],
    [MEF,  AP,   high],
    [EM,   AP,   high],
    [AP,   NEAP, high],
    [PN,   NEAP, high],
    [MD,   PN,   high],
    [ED,   PN,   high],
    [Wi,   ED,   low],
    [Cu,   ED,   low],
    [MEF, "MefUtil", high],
    [EM, "MefUtil",high],
    [NEAP, "NeapUtil", high],
    [PN, "PnUtil", high],
    [EM, "PnUtil", high], 
    ["MefUtil", "TotUtil", high],
    ["NeapUtil", "TotUtil", high], 
    ["PnUtil", "TotUtil", high],    
    [D2L, ED, high],
    [WM, MD, high ],
    [SPT, MD, low ]
])

starting_point = np.array([
    [AP,NEAP, high],
    [PN, NEAP, high]
])

theory_example = np.array([
    ["B", "A", high],
    ["C", "A", high]
])

#net, leaf_states = generate_bn(bn_without_util)
#util, leaf_states = generate_bn(bn_with_util)
mef_bn, _  = generate_bn(mef_arcs)
bn_v3, _ = generate_bn(bn_full_no_util)

bn_complete, _ = generate_bn(complete_bn)

bn_starting_point, _ = generate_bn(starting_point)
bn_theory,_ = generate_bn(theory_example)


"""
net = net.to_graphviz()
net.graph_attr.update(rankdir="BT")
mode = net.get_node(EM)
mode.attr["shape"] = "rectangle"
net.layout(prog="dot")
net.draw("bns/without_util.png", format="png")

util = util.to_graphviz()
util.graph_attr.update(rankdir="BT")
mode = util.get_node(EM)
mode.attr["shape"] = "rectangle"
u1 = util.get_node("MefUtil")
u2 = util.get_node("NeapUtil")
u3 = util.get_node("NpUtil")
u4 = util.get_node("TotUtil")
u1.attr["shape"] = "hexagon"
u2.attr["shape"] = "hexagon"
u3.attr["shape"] = "hexagon"
u4.attr["shape"] = "hexagon"
util.layout(prog="dot")
util.draw("bns/with_util2.png", format="png")
"""

mef_bn = mef_bn.to_graphviz()
mef_bn.graph_attr.update(rankdir="BT")
mef_bn.graph_attr.update(ordering = "in")
mef_bn.layout(prog="dot")
mef_bn.get_node(EM).attr["shape"] = "rectangle"
mef_bn.draw("bns/mef_bn.svg", format="svg")




bn_v3 = bn_v3.to_graphviz()
bn_v3.graph_attr.update(rankdir="BT")
bn_v3.graph_attr.update(ordering = "in")
mode = bn_v3.get_node(EM)
mode.attr["shape"] = "rectangle"
bn_v3.layout(prog="dot")
bn_v3.draw("bns/bn_full_no_util.svg", format="svg")

""""""
bn_complete = bn_complete.to_graphviz()
bn_complete.graph_attr.update(rankdir="BT")
mode = bn_complete.get_node(EM)
mode.attr["shape"] = "rectangle"
u1 = bn_complete.get_node("MefUtil")
u2 = bn_complete.get_node("NeapUtil")
u3 = bn_complete.get_node("PnUtil")
u4 = bn_complete.get_node("TotUtil")
u1.attr["shape"] = "hexagon"
u2.attr["shape"] = "hexagon"
u3.attr["shape"] = "hexagon"
u4.attr["shape"] = "hexagon"
bn_complete.layout(prog="dot")
bn_complete.draw("bns/bn_complete.svg", format="svg")

engine_nodes = [ES, tau, FC, OP, WT, AMF, WT, AMF, IMAP, TSS]
pn_nodes = [SPT, WM, Wi, Cu]

for node in engine_nodes: 
    bn_complete.get_node(node).attr["style"] = "filled"
    bn_complete.get_node(node).attr["color"] = "/set23/1"

bn_complete.get_node(D2L).attr["style"] = "filled"
bn_complete.get_node(D2L).attr["color"] = "/set23/2"

for node in pn_nodes: 
    bn_complete.get_node(node).attr["style"] = "filled"
    bn_complete.get_node(node).attr["color"] = "/set23/3"



bn_complete.draw ("bns/bn_complete_colored.svg", format="svg")


bn_starting_point = bn_starting_point.to_graphviz()
bn_starting_point.graph_attr.update(rankdir="BT")
bn_starting_point.layout(prog ="dot") 
bn_starting_point.draw("bns/starting_point.svg", format="svg")

bn_theory = bn_theory.to_graphviz()
bn_theory.graph_attr.update(rankdir="BT")
bn_theory.layout(prog ="dot") 
bn_theory.draw("bns/theory.svg", format="svg")
