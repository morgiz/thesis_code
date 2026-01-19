import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

time_step = 0.5

#for test
d2l_limits = [500, 800, ]
speed_limits = [10, 6, ] 
###########

test_num = "4"
open_folder = f"sim_results/test{test_num}/" #Folder to open results from
save_folder = f"plots/sim_test{test_num}/" #Folder to save plots to

#Leaf list_test 2
leaf_list_test3 = ['D2L']


#leaf_list_test3 = ['ES', 'FC', 'OP', 'WT','SPT', 'D2L']

#Leaf list test 4
#leaf_list_test3 = ['ES', 'tau', 'FC', 'OP', 'WT', 'AMF','IMAP', 'WM', 'SPT', 'Wi', 'Cu', 'D2L'] 


res = np.load(f"{open_folder}results.npy", allow_pickle=True).item()
drawings = np.load(f"{open_folder}drawings.npy", allow_pickle=True).tolist()
obst_array = np.load(f"{open_folder}obst.npy", allow_pickle=True).tolist()
leaf_node_plt_states = np.load(f"{open_folder}leaves.npy", allow_pickle=True).tolist()
utility_for_plot = np.load(f"{open_folder}util.npy", allow_pickle=True).tolist()
m_t_d2l = np.load(f"{open_folder}modes_times_d2l.npy", allow_pickle=True).tolist()
modes_for_plot = m_t_d2l[0]
times = m_t_d2l[1]
d2l_for_plot = m_t_d2l[2]
child_map = np.load(f"{open_folder}childs.npy", allow_pickle=True).item()
accident_times_for_plot = np.load(f"{open_folder}acc_times.npy", allow_pickle=True).tolist()
results = pd.DataFrame().from_dict(res)

# Example on how a map-view can be generated
map_fig, map_ax = plt.subplots()
map_ax.plot(results['east position [m]'], results['north position [m]'])
#map_ax.scatter(auto_pilot.navigate.east, auto_pilot.navigate.north, marker='x', color='orange')  # Plot the waypoints - Originalen. 


wp_east, wp_north, fbp_east, fbp_north = [[] for _ in range(4)]
with open('routes/obstacle_route_2.txt', 'r') as wp: 
    for line in wp.readlines():
        list = line.split(" ")
        wp_east.append(int(list[0]))
        wp_north.append(int(list[1]))
map_ax.scatter(wp_east, wp_north, marker='x',color='green' )


map_ax.set_title('Map and ship route')
map_ax.set_xlabel('East [m]')
map_ax.set_ylabel('North [m]')
for x, y in zip(drawings[1], drawings[0]):
    map_ax.plot(x, y, color='black')
for obst in obst_array: 
    obst.plot_obst(map_ax) #Plot obstacle in same map as ship

for t in accident_times_for_plot: #Bruker denne til å markere hvert 50. sekund på kartet
    map_ax.plot(results['east position [m]'][int(t/time_step)],
                results['north position [m]'][int(t/time_step)],
                marker = 'x', color = 'red')
map_ax.set_aspect('equal')

mode_fig, mode_ax = plt.subplots()
mode_ax.plot(times , modes_for_plot)
mode_ax.set_title('Machinery modes')
mode_ax.set_yticks([0,1,2], ['PTI', 'PTO', 'Mech'])
mode_ax.set_xlabel('Time [s]')

util_fig, util_ax = plt.subplots()
util_ax.plot(times , utility_for_plot[0], label='PTI')
util_ax.plot(times , utility_for_plot[1], label='PTO')
util_ax.plot(times , utility_for_plot[2], label='Mech')
util_ax.set_title('Utility values')
util_ax.set_ylabel('Utility') 
util_ax.set_xlabel('Time [s]')
util_ax.legend()

speed_fig, speed_ax = plt.subplots()
speed_ax.plot(times , results['forward speed[m/s]'])
speed_ax.set_title('Ship speeds')
speed_ax.set_ylabel('Speed [m/s]') 
speed_ax.set_ylim(0)
speed_ax.set_xlabel('Time [s]')
speed_ax.axhline(y = 7, color = 'green', linestyle = '--', label = 'Speed Setpoint')
#speed_ax.axhline(y=speed_limits[0], color='r', linestyle='--', label='Warning limit')
#speed_ax.axhline(y=speed_limits[1], color='orange', linestyle='--', label='Critical limit')

d2l_fig, d2l_ax = plt.subplots()
d2l_ax.plot(times, d2l_for_plot)
d2l_ax.set_title("Distance to land")
d2l_ax.set_ylabel("Distance [m/s]")
d2l_ax.set_xlabel("Time [s]") 
d2l_ax.axhline(y=d2l_limits[0], color='r', linestyle='--', label='Warning limit')
d2l_ax.axhline(y=d2l_limits[1], color='orange', linestyle='--', label='Critical limit')


#Plot all leaf-node states
#leaf_state_fig, state_ax = plt.subplots(len(leaf_node_plt_states), 1, figsize=(10, 2*len(leaf_node_plt_states)), sharex=True)
leaf_state_fig, state_ax = plt.subplots(len(leaf_list_test3), 1, figsize=(8, 2*len(leaf_list_test3)), sharex=True)
#for i, leaf in enumerate(leaf_node_plt_states.keys()):
for i, leaf in enumerate(leaf_list_test3):    
    leaf_arr = np.array(leaf_node_plt_states[leaf])
    state_ax.plot(times, leaf_arr, marker = '.')
    state_ax.set_title(f'{leaf} states')
    state_ax.set_yticks([0,1,2], ['Worst', 'Intermediate', 'Best'])
    #state_ax[i].legend()
leaf_state_fig.supxlabel('Time [s]')
leaf_state_fig.supylabel('State')
leaf_state_fig.subplots_adjust(hspace=0.1)
leaf_state_fig.set_layout_engine('tight')

#Plot observed child node states
child_state_fig, state_ax = plt.subplots(len(child_map), 1, figsize=(8, 2*len(child_map)), layout = 'constrained', sharex=True)
for i, child in enumerate(child_map.keys()):
    child_arr = np.array(child_map[child])
    state_ax[i].plot(times, child_arr[:,0], marker = '.', label='Worst')
    state_ax[i].plot(times, child_arr[:,1], marker = '.', label='Intermediate')
    state_ax[i].plot(times, child_arr[:,2], marker = '.', label='Best')
    state_ax[i].set_title(f'Prob. of {child} states')
    state_ax[i].set_ylim(0,1)
    #state_ax[i].legend()
child_state_fig.supxlabel('Time [s]')
child_state_fig.supylabel('Probability')
child_state_fig.legend(['Worst', 'Intermediate', 'Best'])
child_state_fig.subplots_adjust(hspace=0.3)

map_fig.savefig(f'{save_folder}map_figure_test{test_num}.svg', format='svg')
mode_fig.savefig(f'{save_folder}mode_figure_test{test_num}.svg', format='svg')
util_fig.savefig(f'{save_folder}util_figure_test{test_num}.svg', format='svg')
speed_fig.savefig(f'{save_folder}speed_figure_test{test_num}.svg', format='svg')
d2l_fig.savefig(f'{save_folder}d2l_figure_test{test_num}.svg', format='svg')
leaf_state_fig.savefig(f'{save_folder}leaf_state_figure_test{test_num}.svg', format='svg')
child_state_fig.savefig(f'{save_folder}child_state_figure_test{test_num}.svg', format='svg')

plt.show()
