import win32com.client  
import time 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict

from numpy.ma.core import default_filler


End_of_simulation = [3600, 900, 1200, 2500][-1]  # simulation second [s]
num_episodes = 60
Random_Seed = 41
steps_per_sec = 1
FEEDBACK_STEP = 60
speeds = [None, [15,30,45,55,60], [20,40,60], [30,45,60]][-1]
speed_steps = len(speeds)
# max_speed = 60
# min_speed = [15, 20][-1]
density_critical = 0.3 # 0.6
discount = 0.9
exploration_flag,exploration_limit,exploration_decay = [(0.9, 0.1, 0.05), (0.9, 0.1, 0.02)][0]
best_loss = 1e9
model_dir = './models/'

if 1:  # 988
    basedir = "D:\\vissim experiments\\nyc_nj\\"
    name = ["Mainz", "Mainz-2-bursts", "Mainz-decay", "Mainz20", "Mainz20copy"][-1]
    filename = basedir + name + ".inpx"
    Filename = basedir + name + ".layx"
else:
    filename = r"D:\\Users\\ra3106\Desktop\\vissim experiments\\base case\\base.inpx"  # 好像不能用相对路径
    Filename = r"C:\\Users\\ra3106\Desktop\\vissim experiments\\base case\\base.layx"


time_1 = time.time()  # seconds between each state snapshot

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
    rand = np.random.uniform(0, 1)
    if (rand >= exploration_flag):
        logits = rl_model(state)
        logits = F.softmax(logits, dim=1)
        actions = torch.argmax(logits, dim=1).detach()
    else:
        actions = torch.randint(high=speed_steps, size=(len(state),)).to(device)
    # print("num_links",num_links)
    return (actions)


# features = []
# X = np.array(features) # density,
# model = NeuralNetwork().to(device)
# print(model)


Vissim = win32com.client.dynamic.Dispatch("Vissim.Vissim.25")
print(Vissim)

Vissim.LoadNet(filename)
Vissim.LoadLayout(Filename)

# Simulation.Stop resets the simulation
# Simulation.Run after stop starts a new run
# for i in range(3):
#     for j in range(1000):
#         Vissim.Simulation.RunSingleStep()
#     time.sleep(5)
#     Vissim.Simulation.Stop()
#
# exit()
time_2 = time.time()
print("Load Time:", time_2-time_1)

# 3.仿真参数设置

Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
Vissim.Simulation.SetAttValue('RandSeed', Random_Seed)
Vissim.Simulation.SetAttValue('SimRes', steps_per_sec)
Vissim.Simulation.SetAttValue("SimBreakAt", End_of_simulation)


links = Vissim.Net.Links.GetAll()
rl_links = eval(open('rl_links_mainz.txt','r').read())
num_links = len(rl_links)
rl_model_path = './models/model.pth'

rl_model = NeuralNetwork(num_links, 6, speed_steps).to(device)
rl_model.load_state_dict(torch.load(rl_model_path, weights_only=True))
optimizer = optim.Adam(rl_model.parameters(), lr=0.05)  # Adam optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss function


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
        for i, list_link in enumerate(rl_links):
            for link_no in list_link:
                link_obj = Vissim.Net.Links.ItemByKey(link_no)
                lanes = link_obj.Lanes.GetAll()
                for lane in lanes:
                    lane.Vehs.SetAllAttValues("DesSpeed", speeds[action[i]])
                # for veh in data[link_no]['all_vehicles']:
                #     veh.SetAttValue("DesSpeed", desired_speed[i])

    end = Vissim.Simulation.AttValue('SimSec')
    print(end)
    Vissim.Simulation.Stop()

    return (states, actions)

# Simulate and Train
metrics, losses = [], []
for eps in range(num_episodes):
    #
    #  code
    data = init_data(links)
    time_3 = time.time()
    states, actions = run_one_episode(data, exploration_flag)
    time_4 = time.time()
    print(End_of_simulation, "second episode time:", time_4-time_3)
    states = torch.stack(states)
    actions = torch.stack(actions)

    # Decay the exploration
    if exploration_flag > exploration_limit:
        exploration_flag -= exploration_decay
        print('exploration_flag', exploration_flag)

    reward_1 = (states[:, :, 0] > density_critical).type(torch.float) * (-100) #(-500)
    reward_2 = states[:, :, 2]
    reward = reward_1 + reward_2 * 0.2 # 0.1

    q_values = []
    q_sa, q_sa_max = [], []
    for i in range(len(states)):
        q_values.append(rl_model(states[i]))
    q_values = torch.stack(q_values)
    for i in range(q_values.shape[0]):
        for j in range(q_values.shape[1]):
            k = actions[i, j].detach().item()
            q_sa.append(q_values[i, j, k])
            q_sa_max.append(q_values[i, j].max())
    q_sa = torch.stack(q_sa).view(actions.shape)
    q_sa_max = torch.stack(q_sa_max).view(actions.shape)
    loss = 0.5 * (q_sa[:-1] - (reward[1:] + discount * q_sa_max[1:])) ** 2
    loss2 = loss.sum(dim=1)
    optimizer.zero_grad()
    for i in range(len(loss)):
        loss2[i].backward(retain_graph=True)
    optimizer.step()
    loss_value = loss2.mean().detach().cpu().item()
    losses.append(loss_value)

    if loss_value < best_loss:
        best_loss = loss_value
        path = model_dir+'{}.pth'.format(eps)
        torch.save(rl_model.state_dict(), path)
        print('Saved model to', path)

    #     Validation code
    if eps % 5 == 0 and 0:
        print("-----Validation Run------")
        data = init_data(links)

        # rl_model.load_state_dict(torch.load(PATH, weights_only=True))
        rl_model.eval()
        states, actions = run_one_episode(data, 0)
        rl_model.train()

        states = torch.stack(states)
        actions = torch.stack(actions)
        data_collection_points = Vissim.Net.DataCollectionMeasurements.GetAll()
        point = data_collection_points[0]
        thruput = point.AttValue('Vehs(Current, Last, All)')
        avg_speed = states[:,:,2].mean(axis=0).mean(axis=0)

        metrics.append(
            [thruput, avg_speed])
        pd.DataFrame(metrics).to_csv('./metrics_rl.csv')
        pd.DataFrame(losses).to_csv('./losses_rl.csv')
    print('EPS:', eps, 'Loss:', loss_value)
    print("--------------------------------")
    time_5 = time.time()
    print("Training time", time_5 - time_4)

print('Done')