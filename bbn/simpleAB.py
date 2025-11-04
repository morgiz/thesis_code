import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

net = BBN()
A = "A"
B = "B" 

nodelist = [A, B]
net.add_nodes_from(nodelist)

edgelist = [(A, B)]
net.add_edges_from(edgelist)

# Define CPTs
cpt_a = TabularCPD(
    variable=A,
    variable_card=2,
    values=[[0.6], [0.4]]
)

cpt_b = TabularCPD(
    variable=B,
    variable_card=2,
    values=[[0.7, 0.2],   # P(B=0|A)
            [0.3, 0.8]],  # P(B=1|A)
    evidence=[A],
    evidence_card=[2]
)

net.add_cpds(cpt_a, cpt_b)

# Verify the model
assert net.check_model()

graph = net.to_daft()

graph.show() # This will display the graph if run in an appropriate environment

