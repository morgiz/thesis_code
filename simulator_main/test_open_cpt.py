import matplotlib.pyplot as plt
import matplotlib
import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

from cpt_func import *
from bn import * 

import numpy as np
import pandas as pd
import daft

#a = pd.read_csv("cpts/AP_cpt.txt")
#a = a.to_numpy()
#print(a)

net = generate_bn()

#print_bn(net)
printable = net.to_daft(node_params={'EM':{'shape': 'rectangle'}})
printable.show()
#printable = net.to_graphviz()
#printable.draw('test.png', prog = 'dot')


