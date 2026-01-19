from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, StaticObstacle, \
    MachineryMode, MachineryModeParams, MachineryModes, ThrottleControllerGains, \
    EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingControllerGains, \
    SpecificFuelConsumptionWartila6L26, SpecificFuelConsumptionBaudouin6M26Dot3, LosParameters
import numpy as np
import matplotlib.pyplot as plt

from pgmpy.inference import BeliefPropagation

from bn import *
from cpt_func import *

"""
This script simulates a ship en-route which passes through two islands. 
D2L is changed according to set values
MEF fault is introduced at 400 by setting all engine leaf nodes to Worst. (except TSS)
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
    current_velocity_component_from_north=0, #originally set to -2
    current_velocity_component_from_east=0, #originally set to -2
    wind_speed=0,
    wind_direction=0
)

#Code below taken from example_ship_basic_simulation.py and added route following
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

#Main Engine Failure mode
mef_mode_params = MachineryModeParams(
    main_engine_capacity=0.75*main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline
)
mef_mode = MachineryMode(params=mef_mode_params)

mso_modes = MachineryModes(
    [pti_mode,
     pto_mode,
     mec_mode,
     mef_mode]
)


fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=2, #Initially mech mode
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
    initial_yaw_angle_rad=60 * np.pi / 180,
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

heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
los_guidance_parameters = LosParameters(
    radius_of_acceptance=600,
    lookahead_distance=500,
    integral_gain=0.002,
    integrator_windup_limit=4000
)


route_file = "routes/obstacle_route_2.txt"
auto_pilot = HeadingByRouteController(
    route_name=route_file,
    heading_controller_gains=heading_controller_gains,
    los_parameters=los_guidance_parameters,
    time_step=time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180
)

####################      Obstacle(s)        ########################
obst1 = StaticObstacle(
    n_pos=2500,
    e_pos=3500, 
    radius=350
)

obst2 = StaticObstacle(
    n_pos=2500,
    e_pos=1500,
    radius=500
)

obst_array = [obst1, obst2]

#              500m : W, <800: I, 
d2l_limits = [500, 800, ]
speed_limits = [10, 6, ] 

times = []

rudder_plot = []
rudder_plot_fb = []
rudder_angle = 0

######################     Bayesian model     #######################

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

mefUtilWeights = [[1, 0, 0], 
                  [1, 0.5, 0.5], 
                  [1, 1, 1]]

neapUtilWeights = [0, 0.5, 1]

npUtilWeights = [[0, 0.75, 1 ],
                [0.5, 0.75, 1 ],
                [1, 0.75, 0.5 ]]

util = [0,0,0]

net, leaf_states = generate_bn(arc_table)
bn_belief = BeliefPropagation(net)

#Child nodes to supervise values for in the simulation
child_map = {"AP": [], "PN":[], "NEAP":[], "MEF":[]}

#Dictionary for storing leaf node state for each time step
leaf_node_plt_states = {k: [] for k in leaf_states.keys()}

#*****************************************

modes_for_plot = []
accident_times_for_plot = []
d2l_for_plot = []
utility_for_plot = [[],[],[]]

mech = 2
pti = 0
pto = 1
mef = 3
wantedMode = mech  #Initial wanted mode

worst = 0
intermediate = 1
best = 2

main_engine_mode = mech
tsmc = 0 # Time since mode change
switch_delay = 30  # Delay time before mode can be switched again
        
def set_mode_to(mode: int): #Make sure to update global variable 
    globals()['main_engine_mode'] = mode 
    return mode

def store_leaf_states(leaf_node_plt_states, leaf_states):
    for leaf in leaf_states:
        leaf_node_plt_states[leaf].append(leaf_states[leaf])

#Structure: time: [(leaf_node, new_state), (...)]
#Engine failure update dict: 
leaf_updates = {400: [(ES, worst), (tau, worst),(OP, worst), (FC, worst), (WT, worst), (AMF, worst), (IMAP, worst),(TSS, worst)]}

######################        Main simulation loop       ##########################
while simulation_time < simulation_time_length:
    #if simulation_time % 50 == 0: 
    #    accident_times_for_plot.append(int(simulation_time))

    simulation_time += time_step
    tsmc += time_step  # Increase time since last mode change
    
    # Measure position and speed
    north_position = ship_model.north
    east_position = ship_model.east
    heading = ship_model.yaw_angle
    speed = ship_model.forward_speed
    
    #Update leaf_states based on predefined updates
    if simulation_time in leaf_updates.keys():
        for update in leaf_updates[simulation_time]:
            leaf_states[update[0]] = update[1]

        #Update engine mode to simulate reduced power
        #Don't use set_mode_to() as the ship still operates with mech
        #Only with reduced power
        ship_model.ship_machinery_model.mode_selector(mef)

        accident_times_for_plot.append(int(simulation_time))
            
    for child in child_map.keys(): 
        q = bn_belief.query(variables=[child], evidence=leaf_states)
        val = q.values
        child_map[child].append(val)

    #Update distance to land/obstacle based on dist-limits defined in obst_limits
    #Finds smallest distance to land and updates DL based on this
    obst_distances = []
    for obst in obst_array:
        obst_distances.append( obst.distance(north_position,east_position) )
    if len(obst_distances) > 0: 
        min_dist = np.min(obst_distances)
    d2l_for_plot.append(min_dist)

    if min_dist <= d2l_limits[0]: 
        leaf_states[D2L] = worst
    elif min_dist <= d2l_limits[1]: 
        leaf_states[D2L] = intermediate
    else: 
        leaf_states[D2L] = best

    #Updates current speed based on speed_limits
    if desired_forward_speed_meters_per_second >= speed_limits[0]: 
        leaf_states[SPT] = worst
    elif desired_forward_speed_meters_per_second >= speed_limits[1]:
        leaf_states[SPT] = intermediate
    else:
        leaf_states[SPT] = best
    
    # Translate engine mode to best-worst semantics wrt demand-estimate in model
    if wantedMode == mech: 
        leaf_states[WM] = worst
    elif wantedMode == pto:
        leaf_states[WM] = intermediate
    else: 
        leaf_states[WM] = best

    #Evaluate risk/utility
    if tsmc >= switch_delay:
        for mode in range(3):
            store_state = leaf_states[EM]  #Save current EM state
            temp_states = leaf_states.copy()
            temp_states[EM] = mode
            mefUtil = 0
            neapUtil = 0
            npUtil = 0

            #Perform query to find prob of each state for MEF, NEAP and PN
            mefQ = bn_belief.query([MEF], temp_states).values
            neapQ = bn_belief.query([NEAP], temp_states).values
            npQ = bn_belief.query([PN], temp_states).values

            #Calculate utility for each mode
            for state in range(3): 
                mefUtil += mefQ[state]*mefUtilWeights[state][mode] 
                neapUtil += neapQ[state]*neapUtilWeights[state] 
                npUtil += npQ[state]*npUtilWeights[state][mode] 

            util[mode] = round(float(mefUtil+neapUtil+npUtil),6)

            leaf_states[EM] = store_state  #Restore EM state
        best_mode = util.index(max(util))

        if best_mode != main_engine_mode:  #Sett ny modus ved behov
            ship_model.ship_machinery_model.mode_selector(set_mode_to(best_mode))
        tsmc = 0 
        
    for mode in range (3):
        utility_for_plot[mode].append(util[mode])

######################        Rudder and throttle control       ##########################
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

    times.append(ship_model.int.time)

######################      Store variables       #########################  
    modes_for_plot.append(main_engine_mode)
    store_leaf_states(leaf_node_plt_states, leaf_states)



######################      Draw Ship       #########################
    # Make a drawing of the ship from above every 20 second
    if time_since_last_ship_drawing > 30:
        ship_model.ship_snap_shot()
        time_since_last_ship_drawing = 0
    time_since_last_ship_drawing += ship_model.int.dt
    # Progress time variable to the next time step
    ship_model.int.next_time()


######################     Save results    ######################
np.save("sim_results/test3v2/results", ship_model.simulation_results)
np.save("sim_results/test3v2/drawings", ship_model.ship_drawings)
np.save("sim_results/test3v2/obst", obst_array)
np.save("sim_results/test3v2/leaves", leaf_node_plt_states)
np.save("sim_results/test3v2/util", utility_for_plot)
np.save("sim_results/test3v2/modes_times_d2l", [modes_for_plot,times,d2l_for_plot])
np.save("sim_results/test3v2/childs", child_map)
np.save("sim_results/test3v2/acc_times", accident_times_for_plot)