# -------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      zj2187
#
# Created:     19/08/2023
# Copyright:   (c) zj2187 2023
# Licence:     <your licence>
# -------------------------------------------------------------------------------

# 0.导入库（具体这个库参考其他资料）
import win32com.client  # 主要库
import time  # 不重要
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
import sys
# eval_model = sys.argv[1]
eval_model = ['no_control','rl_model'][1]
from numpy.ma.core import default_filler

# time_1 = time.time()

# 1.连接VISSIM并创建VISSIM对象
# Vissim = win32com.client.gencache.EnsureDispatch("Vissim.Vissim.25")  # 最后数字为版本号
Vissim = win32com.client.dynamic.Dispatch("Vissim.Vissim.23")  # 最后数字为版本号
print(Vissim)

# 2.加载路网(我们在绘制路网的时候通常会导入一张背景图，然后在上面绘制，注意这里仅加载路网不加载背景图)
basedir = "D:\\vissim experiments\\nyc_nj\\"
name = "Mainz20"
filename = basedir + name + ".inpx"
Filename = basedir + name + ".layx"
Vissim.LoadNet(filename)
Vissim.LoadLayout(Filename)

# time_2 = time.time()
rl_model_path = './models/model_tl.pth'
# rl_model_path = './models/model.pth'
End_of_simulation = [3600, 900, 1200, 2500][-1]  # simulation second [s]
num_episodes = 5
FEEDBACK_STEP = 60
speeds = [None, [15, 30, 45, 55, 60], [20, 40, 60], [30,45,60]][-1]
speed_steps = len(speeds)
def_speed = speeds[-1]
density_critical = 0.3  # 0.6
steps_per_sec = 1

Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
Vissim.Simulation.SetAttValue('SimRes', steps_per_sec)
Vissim.Simulation.SetAttValue("SimBreakAt", End_of_simulation)
metrics = []

rl_links = eval(open('rl_links_mainz.txt', 'r').read())
num_links = len(rl_links)

def get_metrics():
    netperform = Vissim.Net.VehicleNetworkPerformanceMeasurement

    speed_avg = netperform.AttValue('SpeedAvg(Current, Last, All)')
    delay_avg = netperform.AttValue('DelayAvg(Current, Last, All)')
    stops_avg = netperform.AttValue('StopsAvg(Current, Last, All)')
    delay_stop_avg = netperform.AttValue('DelayStopAvg(Current, Last, All)')
    dist_tot = netperform.AttValue('DistTot(Current, Last, All)')
    trav_tm_tot = netperform.AttValue('TravTmTot(Current, Last, All)')
    delay_tot = netperform.AttValue('DelayTot(Current, Last, All)')
    stops_tot = netperform.AttValue('StopsTot(Current, Last, All)')
    delay_stop_tot = netperform.AttValue('DelayStopTot(Current, Last, All)')
    veh_act = netperform.AttValue('VehAct(Current, Last, All)')
    veh_arr = netperform.AttValue('VehArr(Current, Last, All)')
    delay_latent = netperform.AttValue('DelayLatent(Current, Last)')
    demand_latent = netperform.AttValue('DemandLatent(Current, Last)')

    return [speed_avg, delay_avg, stops_avg, delay_stop_avg, dist_tot, trav_tm_tot, delay_tot, stops_tot, delay_stop_tot, \
         veh_act, veh_arr, delay_latent, demand_latent]


if(eval_model == 'no_control'):
    def run_one_episode():
        while Vissim.Simulation.AttValue('SimSec') < End_of_simulation - 1:
            Vissim.Simulation.RunSingleStep()
            # if Vissim.Simulation.AttValue('SimSec') % FEEDBACK_STEP:
            #     continue
            # for i, list_link in enumerate(rl_links):
            #     for link_no in list_link:
            #         link_obj = Vissim.Net.Links.ItemByKey(link_no)
            #         lanes = link_obj.Lanes.GetAll()
            #         for lane in lanes:
            #             lane.Vehs.SetAllAttValues("DesSpeed", def_speed)

        Vissim.Simulation.Stop()

    # for eps in range(num_episodes):
    for eps in range(1):
        Random_Seed = eps+1
        Vissim.Simulation.SetAttValue('RandSeed', Random_Seed)
        run_one_episode()

        metrics.append(get_metrics())
        pd.DataFrame(data=metrics,
                     columns=['speed_avg', 'delay_avg', 'stops_avg', 'delay_stop_avg', 'dist_tot', 'trav_tm_tot', \
                              'delay_tot', 'stops_tot', 'delay_stop_tot', 'veh_act', 'veh_arr', 'delay_latent', 'demand_latent']).to_csv('./eval_no_control.csv')


if (eval_model == 'rl_model'):
    '''
    boilerplate definition of NN
    '''
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    class NeuralNetwork(nn.Module):
        def __init__(self, num_links, num_features, num_actions):
            super().__init__()
            self.num_links = num_links
            self.num_actions = num_actions
            self.num_features = num_features
            self.input_length = num_links * num_features
            self.output_length = num_links * num_actions
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_length, 2 * self.input_length),
                nn.ReLU(),
                # nn.Linear(2*self.input_length, 4*self.input_length),
                # nn.ReLU(),
                # nn.Linear(4*self.input_length, 2*self.output_length),
                # nn.ReLU(),
                nn.Linear(2 * self.input_length, self.output_length),
            )

        def forward(self, x):
            x = torch.flatten(x)
            logits = self.linear_relu_stack(x)
            logits = logits.view((self.num_links, self.num_actions))
            return logits


    def get_actions(state, rl_model, exploration_flag):
        # num_links = len(state)
        # rand = np.random.uniform(0, 1)
        # if (rand >= exploration_flag):
        logits = rl_model(state)
        logits = F.softmax(logits, dim=1)
        actions = torch.argmax(logits, dim=1).detach()
        # else:
        #     actions = torch.randint(high=speed_steps, size=(len(state),)).to(device)
        # print("num_links",num_links)
        return (actions)




    # 3.仿真参数设置
    links = Vissim.Net.Links.GetAll()
    rl_model = NeuralNetwork(num_links, 6, speed_steps).to(device)
    rl_model.load_state_dict(torch.load(rl_model_path, weights_only=True))
    rl_model.eval()
    print('Load model from', rl_model_path)

    optimizer = optim.Adam(rl_model.parameters(), lr=0.05)  # Adam optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    discount = 0.9


    def init_data(links):
        data = {}
        # BASE CASE config
        # links = [7,12,11]
        for i in range(len(links)):
            link_no = links[i].AttValue('No')
            link_obj = Vissim.Net.Links.ItemByKey(link_no)

            data[link_no] = {}
            # data[link_no]["model"] = NeuralNetwork().to(device)
            data[link_no]["length"] = link_obj.AttValue('Length2D')
            data[link_no]["num_lanes"] = link_obj.AttValue('NumLanes')
            data[link_no]["density"] = 0
            # data[link_no]["density_prev"] = 0
            data[link_no]["input_rate"] = 0
            data[link_no]["exit_rate"] = 0
            data[link_no]["all_vehicles"] = []
            data[link_no]["prev_all_vehicles"] = set([])
            # data[link_no]["link_idx"] = i
            # data[link_no]["group"] = group_mapping[link_no][0]

            # data[link_no]["next_link"] = 0  #
            # if i < len(links) - 1:
            #     data[link_no]["next_link"] = links[i + 1]  #

        return (data)


    def run_one_episode(data, exploration_flag=0.0):
        states, actions, state_vector = [], [], []
        while Vissim.Simulation.AttValue('SimSec') < End_of_simulation - 1:
            # print(Vissim.Simulation.AttValue('SimSec'))
            # print('.', end='')
            Vissim.Simulation.RunSingleStep()
            # if(Vissim.Simulation.AttValue('SimSec')>600):
            #     all_vehicles = Vissim.Net.Vehicles.GetAll()
            #     if(len(all_vehicles)==0):
            #         end = Vissim.Simulation.AttValue('SimSec')
            #         print(end)
            #         Vissim.Simulation.Stop()

            if Vissim.Simulation.AttValue('SimSec') % FEEDBACK_STEP:
                continue



            # # Get all vehicles in the simulation

            all_vehicles = Vissim.Net.Vehicles.GetAll()
            state_vector = []
            for list_link in rl_links:
                features = []
                for link_no in list_link:

                    link_length = data[link_no]["length"]
                    data[link_no]['all_vehicles'] = [vehicles for vehicles in all_vehicles if link_no == int(vehicles.AttValue('Lane').split('-')[0])]
                    # link_obj = Vissim.Net.Links.ItemByKey(link_no)
                    # lanes = link_obj.Lanes.GetAll()
                    # for lane in lanes:
                        # data[link_no]['all_vehicles'] += list(lane.Vehs.GetAll())

                    if( not len(data[link_no]['all_vehicles'])):
                        num_cars = 0
                    else:
                        d = np.array([[veh.AttValue('No'),veh.AttValue('Speed'),veh.AttValue('FollowDistGr')] for veh in data[link_no]['all_vehicles']])
                        # d = np.array(link_obj.Vehs.GetMultipleAttributes(["No", "Speed", "FollowDistGr"]))
                        num_cars = len(d)
                    if num_cars == 0:
                        d = np.zeros((1,3))
                        sd0 = set([])
                    else:
                        sd0 = set(d[:, 0])

                    num_lanes = data[link_no]["num_lanes"]
                    density = (num_cars * 4.5) / (link_length * num_lanes)
                    data[link_no]["density"] = density

                    avg_speed = np.nanmean(d[:,1])
                    avg_gap = np.nanmean(d[:,2])
                    input_rate = len(sd0 - data[link_no]["prev_all_vehicles"])
                    exit_rate = len( data[link_no]["prev_all_vehicles"] - sd0)

                    data[link_no]["prev_all_vehicles"] = sd0
                    data[link_no]["input_rate"] = input_rate
                    data[link_no]["exit_rate"] = exit_rate

                    features.append([density, num_lanes, avg_speed, avg_gap, input_rate, exit_rate])
                features = np.nanmean(np.array(features),axis=0)
                state_vector.append(features)

            state_vector = np.array(state_vector)
            state_vector = np.nan_to_num(state_vector)
            state_vector = torch.tensor(state_vector, dtype=torch.float).to(device)
            states.append(state_vector)
            action = get_actions(state_vector, rl_model, exploration_flag)
            actions.append(action)
            # desired_speed = (action * ((max_speed - min_speed) / speed_steps)) + min_speed
            

            # MIXED TRAFFIC. SET RATIO CONTROLLED
            # mixed_traffic = [25, 50, 75, 100]
            # mix = mixed_traffic[1]  # 25% of vehicles controlled

            for i, list_link in enumerate(rl_links):
                for link_no in list_link:
                    link_obj = Vissim.Net.Links.ItemByKey(link_no)
                    lanes = link_obj.Lanes.GetAll()
                    s = speeds[action[i]]
                    for lane in lanes:
                        vehs = lane.Vehs.GetAll()
                        
                        for veh in  vehs:
                            # print(veh.AttValue('VehType'))

                            # if "300" in veh.AttValue("VehType"):
                            if veh.AttValue("No") % 2 == 0:
                                veh.SetAttValue("DesSpeed", s)
            # exit()
                    # for lane in lanes:
                        # s = speeds[action[i]]
                        # hi = [5,5,0][action[i]]
                        # s2 = np.random.randint(s-5,s+hi)
                        # lane.Vehs.SetAllAttValues("DesSpeed", s)
                    # for veh in data[link_no]['all_vehicles']:
                    #     veh.SetAttValue("DesSpeed", desired_speed[i])

        end = Vissim.Simulation.AttValue('SimSec')
        print(end)
        Vissim.Simulation.Stop()

        return (states, actions)

    # Simulate and Train
    # MIXED TRAFFIC. SET RATIO CONTROLLED
    mixed_traffic = [25, 50, 75, 100]
    mix = mixed_traffic[1]  # 25% of vehicles controlled

    for eps in range(3):
        # Training code
        Random_Seed = eps+1
        Vissim.Simulation.SetAttValue('RandSeed', Random_Seed)
        data = init_data(links)
        time_3 = time.time()
        states, actions = run_one_episode(data, 0.0)
        time_4 = time.time()
        states = torch.stack(states)
        actions = torch.stack(actions)
        data_collection_points = Vissim.Net.DataCollectionMeasurements.GetAll()
        point = data_collection_points[0]
        thruput = point.AttValue('Vehs(Current, Last, All)')
        avg_speed = states[:,:,2].mean(axis=0).mean(axis=0).item()

        # metrics.append(
        #     [thruput, avg_speed])
        # pd.DataFrame(metrics).to_csv('./metrics_eval.csv')
        metrics.append(get_metrics())
        pd.DataFrame(data=metrics,
                     columns=['speed_avg', 'delay_avg', 'stops_avg', 'delay_stop_avg', 'dist_tot', 'trav_tm_tot', \
                              'delay_tot', 'stops_tot', 'delay_stop_tot', 'veh_act', 'veh_arr', 'delay_latent', 'demand_latent']).to_csv('./eval_rl_control_{}mix_SRC-TL.csv'.format(mix))
        print('EPS:', eps, 'done')

print('Done')