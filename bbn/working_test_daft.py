import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD

#Denne koden kjører fra laptopen min, men jeg var nødt til å legge inn graphviz i PATH midlertidig for at den skulle kjøre. 

#Edit: Kjører nå også på skolepc. bruker daft i stedet for det jeg opprinnelig tenkte. 

net = BBN()

nodelist = ['s1', 's2', 's3', 's4']
net.add_nodes_from(nodelist)

edgelist = [('s1','s2'), ('s2','s3'), ('s2','s4'), ('s3', 's4'), ('s1','s4')]
net.add_edges_from(edgelist)


# Turning Our Bayesian Network into a Picture
graph = net.to_daft() #Bruker daft for å tegne diagrammet. 

# Showing Our Picture
graph.render()
graph.show()