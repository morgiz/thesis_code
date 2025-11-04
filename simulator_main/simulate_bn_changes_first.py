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

net, leaf_states = generate_bn()

simulation_steps = 10
time = 0 
time_steps = 1

#Lag en array som lagrer states for hver løvnode for hvert tidssteg. 


while time < simulation_steps: 
    time += time_steps
    
