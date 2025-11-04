from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, StaticObstacle, \
    MachineryMode, MachineryModeParams, MachineryModes, ThrottleControllerGains, \
    EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingControllerGains, \
    SpecificFuelConsumptionWartila6L26, SpecificFuelConsumptionBaudouin6M26Dot3, LosParameters, \
    BaseMachineryModel, RudderConfiguration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
from pgmpy.models import DiscreteBayesianNetwork as BBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

"""
Notater: 
        #Jeg kan ikke introdusere en feil i main engine capacity direkte i ShipModel, da det
        #blir delt på 0 i en annen funksjon. Må finne på noe annet, i stedet. 
        #ship_model.ship_machinery_model.machinery_modes.list_of_modes[1].main_engine_capacity = 0  # Simulate main engine failure by setting its capacity to zero
        #ship_model.ship_machinery_model.update_available_propulsion_power()

        Får koden til å kjøre, og ser at den endrer mål etter 2 accident punkt, men 
        retningen den setter og holder skjønner jeg ikke hvor kommer fra... 
"""

plt.close('all')

main_engine_capacity = 2160e3
diesel_gen_capacity = 510e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

time_step = 0.5
simulation_time_length = 1000
simulation_time = 0

# Configure the simulation
ship_config = ShipConfiguration(
    coefficient_of_deadweight_to_displacement=0.7,
    bunkers=200000,
    ballast=200000,
    length_of_ship=80,
    width_of_ship=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    dead_weight_tonnage=3850000,
    mass_over_linear_friction_coefficient_in_surge=130,
    mass_over_linear_friction_coefficient_in_sway=18,
    mass_over_linear_friction_coefficient_in_yaw=90,
    nonlinear_friction_coefficient__in_surge=2400,
    nonlinear_friction_coefficient__in_sway=4000,
    nonlinear_friction_coefficient__in_yaw=400
)
env_config = EnvironmentConfiguration(
    current_velocity_component_from_north=-2,
    current_velocity_component_from_east=-2,
    wind_speed=0,
    wind_direction=0
)

#Har tatt koden under fra example_ship_basic_simulation.py og lagt til rute following
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator
)
pto_mode = MachineryMode(params=pto_mode_params)

mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline
)
mec_mode = MachineryMode(params=mec_mode_params)

pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2 * diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor
)
pti_mode = MachineryMode(params=pti_mode_params)

mso_modes = MachineryModes(
    [pto_mode,
     mec_mode,
     pti_mode]
)


fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=1, #Mec mode i utgangspunktet
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200000,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=30,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

simulation_setup = SimulationConfiguration(
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=0 * np.pi / 180,
    initial_forward_speed_m_per_s=5,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=time_step,
    simulation_time=simulation_time_length,
)
ship_model = ShipModel(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=simulation_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)
desired_forward_speed_meters_per_second = 7
time_since_last_ship_drawing = 30

# Place obstacles
obstacle_data = np.loadtxt('obstacles.txt')
list_of_obstacles = []
for obstacle in obstacle_data:
    list_of_obstacles.append(StaticObstacle(obstacle[0], obstacle[1], obstacle[2]))

# Set up control systems
throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=throttle_controller_gains,
    max_shaft_speed=ship_model.ship_machinery_model.shaft_speed_max,
    time_step=time_step,
    initial_shaft_speed_integral_error=114
)

heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01) #ki endret fra 0.01, kp fra 4, pd fra 90
los_guidance_parameters = LosParameters(
    radius_of_acceptance=600,
    lookahead_distance=500,
    integral_gain=0.002, #Endret fra 0.002
    integrator_windup_limit=4000 #endret fra 4000
)


route_file = "bn_test_route.txt"
auto_pilot = HeadingByRouteController(
    route_name=route_file,
    heading_controller_gains=heading_controller_gains,
    los_parameters=los_guidance_parameters,
    time_step=time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180
)

integrator_term = []
times = []

rudder_plot = []
rudder_plot_fb = []
rudder_angle = 0

# Bayesian model #######################
#Initialises BN and relevant variables

net = BBN()
LOP = 'LackOfPower'
AP = 'AvailablePower'
NP = 'NecessaryPower' 
MEF = 'MainEngineFailure'
SC = 'SituationalComplexity'
EM = 'EngineMode' 
WC = 'WeatherConditions'

nodelist = [LOP, AP, MEF, NP, SC, WC, EM] #MÅ ikke egentlig legge til nodene her, for det gjøres når man legger til edges
net.add_nodes_from(nodelist)

edgelist = [(AP,LOP), (MEF,AP), (EM, AP), (SC, NP), (NP, LOP), (WC,NP)]
net.add_edges_from(edgelist)


#Define CPTs
#Får tilstandene 0,1,2 som tilsvarer lav, middels og høy "severity". 
#samme CPT for MEF of SC
cpt_mef = TabularCPD(
    variable=MEF,
    variable_card=3,
    values=[[0.03],[0.9],[0.07]]
)

cpt_em = TabularCPD(
    variable = EM, 
    variable_card=3, 
    values = [[0.9],[0.08],[0.02]]
)


cpt_sc = TabularCPD(
    variable=SC,
    variable_card=3,
    values=[[0.03],[0.9],[0.07]]
)

#Samme CPT for AP og NP
cpt_ap = TabularCPD(
    variable=AP,
    variable_card=3,
    #tilfeldige verdier under
    values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],  
            [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     
            [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],   

    evidence=[MEF,EM], #Which nodes act as evidence
    evidence_card=[3,3] #number of states in evidence nodes
)

cpt_np = TabularCPD(
    variable=NP,
    variable_card=3,
    #Tilfeldige verdier i cpt
     values=[[0.90,0.20,0.01, 0.95,0.88,0.12, 0.98,0.92, 0.90],     
            [0.07,0.68,0.21, 0.04,0.10,0.65, 0.011,0.05,0.06],     
            [0.03,0.12,0.78, 0.01,0.02,0.23, 0.009,0.03, 0.04]],     
    evidence=[SC,WC],
    evidence_card=[3,3]
)
cpt_wc = TabularCPD(
    variable=WC,
    variable_card=3,
    values=[[0.03],[0.9],[0.07]]
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
net.add_cpds(cpt_ap, cpt_np, cpt_lop, cpt_mef, cpt_sc, cpt_wc, cpt_em)

net.check_model()
leafnode_state_dict = {MEF:1, EM: 1 , SC:1, WC:0}

# net.fit() sjekker tilstanden til alle cpt basert på evidence. 
printable = net.to_daft(node_params={EM:{'shape': 'rectangle'}})
#printable.show() 

belief_prop = BeliefPropagation(net)

#*****************************************


main_engine_fault = False
fault_time = 100  # Time in seconds when the fault occurs
modes_for_plot = []
accident_times_for_plot = []
state = 0 # state 0 is normal operation, 1 is FBS 

mech = 1
pti = 0
pto = 2

main_engine_mode = mech

"""
Sett opp nett og vurder risiko for hver eneste decision node.  
"""
#Available power and power_necssary will have a value of 0,1 or 2 representing
#low, medium and high, where medium is expected normal operation. 
def evaluate_BN(available, necessary): 
    if available < necessary:
        ship_model.ship_machinery_model.mode_selector(set_mode_to(mech))
        if ship_model.ship_machinery_model.mode == pto:
            print("Båten er i PTO modus. Skift til noe med mer kraft")
    
        
def set_mode_to(mode: int): #Sørger for at main_engine_mode endres globalt, når modusen settes i en annen funksjon. 
    globals()['main_engine_mode'] = mode 
    return mode

###########################################


def set_fbs():
    globals()['state'] = 1




#*********************************************************************

while simulation_time < simulation_time_length:
    simulation_time += time_step
 
    # Measure position and speed
    north_position = ship_model.north
    east_position = ship_model.east
    heading = ship_model.yaw_angle
    speed = ship_model.forward_speed

    if ship_model.int.time >= fault_time and (not main_engine_fault):
        main_engine_fault = True
        #Simulerer en motorfeil som gjør at skipet må gå på pto mode. 
        ship_model.ship_machinery_model.mode_selector(set_mode_to(pto))
        accident_times_for_plot.append(ship_model.int.time)

    if ship_model.int.time >= (fault_time + 50) and state == 0 :
        leafnode_state_dict[SC] = 2 #Setter nødvendig kraft til high etter 500 sekunder med feil. 
        #Dette for å simulere at skipet trenger mer kraft for å nå en havn eller lignende.
        set_fbs()
        accident_times_for_plot.append((ship_model.int.time))

    ap_query = belief_prop.map_query(variables=[AP], evidence = {MEF:leafnode_state_dict[MEF], SC: leafnode_state_dict[SC]}, show_progress = True)
    np_query = belief_prop.map_query(variables=[NP], evidence = {MEF:leafnode_state_dict[MEF], SC: leafnode_state_dict[SC]}, show_progress = True)

    ap_most_prob_state = ap_query[AP]
    np_most_prob_state = np_query[NP]    

    #BN tar en vurdering av nødvendig kraft for hvert tidssteg. 
    evaluate_BN(ap_most_prob_state, np_most_prob_state)

    # Find appropriate rudder angle and engine throttle
    rudder_angle = auto_pilot.rudder_angle_from_route(
        north_position=north_position,
        east_position=east_position,
        heading=heading
    )

    rudder_plot.append(rudder_angle)

    throttle = throttle_controller.throttle(
        speed_set_point=desired_forward_speed_meters_per_second,
        measured_speed=speed,
        measured_shaft_speed=speed
    )

    # Update and integrate differential equations for current time step
    ship_model.store_simulation_data(throttle)
    ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)     
    ship_model.integrate_differentials()

    #integrator_term.append(auto_pilot.navigate.e_ct_int)

    times.append(ship_model.int.time)

#*********************************************************************
    
    modes_for_plot.append(main_engine_mode)

#*********************************************************************

    # Make a drawing of the ship from above every 20 second
    if time_since_last_ship_drawing > 30:
        ship_model.ship_snap_shot()
        time_since_last_ship_drawing = 0
    time_since_last_ship_drawing += ship_model.int.dt
    # Progress time variable to the next time step
    ship_model.int.next_time()


# Store the simulation results in a pandas dataframe
results = pd.DataFrame().from_dict(ship_model.simulation_results)

# Example on how a map-view can be generated
map_fig, map_ax = plt.subplots()
map_ax.plot(results['east position [m]'], results['north position [m]'])
#map_ax.scatter(auto_pilot.navigate.east, auto_pilot.navigate.north, marker='x', color='green')  # Plot the waypoints - Originalen. 

"""
Jeg plotter heller waypoints og fallback points utenom det som er i auto-piloten når simuleringen er ferdig. 
Vil heller kunne plotte alle punktene. 
"""
wp_east, wp_north, fbp_east, fbp_north = [[] for _ in range(4)]
with open('bn_test_route.txt', 'r') as wp: 
    for line in wp.readlines():
        list = line.split(" ")
        wp_east.append(int(list[1]))
        wp_north.append(int(list[0]))

map_ax.scatter(wp_east, wp_north, marker='x', color='green')
map_ax.scatter(fbp_east, fbp_north, marker = 'o', color = 'red')

map_ax.set_title('Map and ship route')
map_ax.set_xlabel('East [m]')
map_ax.set_ylabel('North [m]')
for x, y in zip(ship_model.ship_drawings[1], ship_model.ship_drawings[0]):
    map_ax.plot(x, y, color='black')
#for obstacle in list_of_obstacles:
#    obstacle.plot_obst(ax=map_ax)

#for north,east in zip(accidents_for_plot[0],accidents_for_plot[1]):
for t in accident_times_for_plot: 
    map_ax.plot(results['east position [m]'][int(t/time_step)],
                results['north position [m]'][int(t/time_step)],
                marker = 'o', color = 'orange')

map_ax.set_aspect('equal')

# Example on plotting time series
#fuel_ifg, fuel_ax = plt.subplots()
#results.plot(x='time [s]', y='power [kw]', ax=fuel_ax)

#int_fig, int_ax = plt.subplots()
#int_ax.plot(times, integrator_term)

# Rudder plots
#rud_fig, rud_ax = plt.subplots()
#rud_ax.plot(times, rudder_plot, color = "green")
#rud_ax.plot(times, rudder_plot_fb, color = "red")
#rud_ax.plot(times, ship_model.simulation_results['yaw angle [deg]'], color = 'blue')
#rud_ax.set_title("Rudder angles over time")

"""
mode_fig, mode_ax = plt.subplots()
mode_ax.plot(times , modes_for_plot)
mode_ax.set_title('Machinery modes over time')
mode_ax.set_ylabel('0: PTO, 1: MEC, 2: PTI')
mode_ax.set_xlabel('Time [s]')
"""
plt.show()

