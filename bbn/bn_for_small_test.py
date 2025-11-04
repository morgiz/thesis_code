import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd

#Denne koden kjører fra laptopen min, men jeg var nødt til å legge inn graphviz i PATH midlertidig for at den skulle kjøre. 

#Edit: Kjører nå også på skolepc. bruker daft i stedet for det jeg opprinnelig tenkte. 

"""
Notater om dokumentet og moduler:
- Har skrevet litt forskjellige kodesnutter som kan være nyttig. ting er koded inn og ut, og ikke alt er relevant på en gang. 
- Man trenger ikke calibrere() treet, da dette fikses av _query() som kalles av query() og max_query()
- map_query() returnerer mest sannsynlige tilstand for en variabel, mens en query() returnerer en joint tabell by default.
- En joint tabell fra query kan ikke itereres over, men det kan en join=False tabell, da den returnerer en dictionary med de gitte CPTene. 
- Kan ikke kjøre query på en var med bevis for samme var. 
""" 

net = BBN()

LOP = 'LackOfPower'
AP = 'AvailablePower'
NP = 'NecessaryPower' 
MEF = 'MainEngineFailure'
SC = 'SituationalComplexity'
EM = 'EngineMode'

nodelist = [LOP, AP, MEF, NP, SC,EM]
net.add_nodes_from(nodelist)

edgelist = [(AP,LOP), (MEF,AP), (EM,MEF), (EM, AP), (SC, NP), (NP, LOP)]
net.add_edges_from(edgelist)


#Define CPTs
#Får tilstandene 0,1,2 som tilsvarer lav, middels og høy "severity". 
#samme CPT for MEF of SC


cpt_mef = TabularCPD(
    variable=MEF,
    variable_card=3,
    values=[[0.03],[0.9],[0.07]],
    evidence = [EM]
)

cpt_sc = TabularCPD(
    variable=SC,
    variable_card=3,
    values=[[0.03],[0.9],[0.07]]
)

#Samme CPT for AP og NP
#Leser fra CPT fil og setter inn i values. lage en lese-funksjon som returnerer en array elns? 
a = pd.read_csv("cpts/LSF_cpt.txt")
a = a.to_numpy()
print(a)
cpt_ap = TabularCPD(
    variable=AP,
    variable_card=3,
    #  MEF |  L   |  M  |  H  |
    values=[[0.01, 0.20, 0.90],     #LOP = low
            [0.21, 0.68, 0.07],     #LOP = medium
            [0.78, 0.12, 0.03]],     #LOP = High
    evidence=[MEF,EM], #Which nodes act as evidence
    evidence_card=[3] #number of states in evidence nodes
)


cpt_np = TabularCPD(
    variable=NP,
    variable_card=3,
    #   SC |  L   |  M  |  H  | 
    values=[[0.01, 0.20, 0.90],     #LOP = low
            [0.21, 0.68, 0.07],     #LOP = medium
            [0.78, 0.12, 0.03]],     #LOP = High

    evidence=[SC],
    evidence_card=[3]
)

cpt_lop = TabularCPD(
    variable=LOP,
    variable_card=3, # num of states

    #Fyller inn litt tilfeldige verdier under, bare for å ha noe å gå etter
        #AP  #     L       |       M        |       H      |
        #NP  # L   M    H  |  L    M     H  | L     M    H |
    values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],     #LOP = low
            [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     #LOP = medium
            [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],     #LOP = High
    evidence=[AP, NP],
    evidence_card=[3,3]
)
net.add_cpds(cpt_ap,cpt_np,cpt_lop,cpt_mef, cpt_sc)

net.check_model()


#alle_cpt = net.get_cpds()
#for i in alle_cpt: 
#    print(i)

#print(net.get_independencies().independencies[:]) # funksjon fra pgmpy

belief_prop = BeliefPropagation(net) #Gjør om modellen til et junction tree. 
#q = belief_prop.query(variables=[LOP,NP], evidence= {AP: 0}, show_progress=True)
#q_new = belief_prop.query(variables=[LOP], evidence = {AP:0}, show_progress = True, joint = True) #Legger sammen verdiene i query'en sånn
                                                                                                #at vi ser kun de sammenlagdte sannsynlighetene for LOP
#q_np = belief_prop.query(variables=[NP], evidence = {AP:0}, show_progress = True, joint=True) 
#print(q_np)
#state_name = q_np.get_state_names(var=NP,state_no=0)
#print(state_name)
#print(q_np.values[1])
leafnode_state_dict = {MEF:1, SC: 1}

for i in range (10): 
    if i > 5: 
        leafnode_state_dict[MEF] = 2

    ap_query = belief_prop.map_query(variables=[AP], evidence = {MEF:leafnode_state_dict[MEF], SC: leafnode_state_dict[SC]}, show_progress = True)
    np_query = belief_prop.map_query(variables=[NP], evidence = {MEF:leafnode_state_dict[MEF], SC: leafnode_state_dict[SC]}, show_progress = True)

    print("Iterasjon "+str(i))
    print("AP: " + str(ap_query[AP]))
    print("NP: " + str(np_query[NP]))





#print(q_new)

#print("Mest sannsynlige tilstand hhv LOP og NP")
#print(belief_prop.map_query([LOP], {AP:0}))


# Turning Our Bayesian Network into a Picture
graph = net.to_daft(node_params={'EM':{'shape': 'rectangle'}}) #Bruker daft for å tegne diagrammet. 

# Showing Our Picture
#graph.render()
graph.show() #Show inneholder render(), så ikke nødvendig å kjøre begge. 