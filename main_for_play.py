import gym
import os
import mujoco_py
from agent import Agent
from play import Play
import numpy as np
import datetime
import time
import make_adv_length_ant
import csv
import matplotlib.pyplot as plt 
import requests
import random
import pandas as pd
import graph_plot2

TRAIN_FLAG = False 

seed_id = 1

dt_now = datetime.datetime.now() 
time_info = " " + str(dt_now.month) +"."+ str(dt_now.day) +"."+ str(dt_now.hour) +":"+ str(dt_now.minute)+""+ str(dt_now.second)+" N:"

ENV_NAME = "Ant"

ATTACK = "length"

num_rep = 1000

epsilon = 0.05

body_list = ["normal","random","adversarial"]

if ENV_NAME == "Walker2d":
	num_variables = 7	
	
elif ENV_NAME == "Ant":
	if ATTACK == "length":
		num_variables = 12
	else:
		num_variables = 13

elif ENV_NAME == "Humanoid":
	if ATTACK == "length":
		num_variables = 16
	else:
		num_variables = 19

test_env = gym.make(ENV_NAME + "-v2")
n_states = test_env.observation_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]
n_iterations = 10000
lr = 3e-5

if __name__ == "__main__":
	device = "cpu"

for body in body_list:
	if ENV_NAME == "Ant":
		size_ratio=  [1,1,1,1,1,1,1,1,1,1,1,1,1] # ant
		length_ratio=[1,1,1,1,1,1,1,1,1,1,1,1] # ant			

		make_adv_length_ant.set_adv_length(length_ratio)
	
	env = gym.make(ENV_NAME + "-v2")
	env.seed(seed_id)	
	agent = Agent(n_states=n_states,
				n_iter=n_iterations,
				env_name=ENV_NAME,
				action_bounds=action_bounds,
				n_actions=n_actions,
				lr=lr,
				device=device)	
						
	player = Play(env, agent, ENV_NAME)
	player.evaluate()

	reward_list = []
	
	with open("./Hist_"+ENV_NAME+"/"+ENV_NAME+time_info+str(num_rep)+".csv", 'a') as f:
		writer = csv.writer(f)
		writer.writerow(["num","reward"])
		
	for i in range(num_rep):
		# random body
		if body == "random":
			adv_list = [] 
			for k in range(num_variables):
				random.seed(i*100+k)
				r = random.random()*epsilon*2
				adv_list.append(1.0 + r - epsilon)
				
			if ENV_NAME == "Ant" and ATTACK == "length":
				make_adv_length_ant.set_adv_length(adv_list)
				
			env = gym.make(ENV_NAME + "-v2")
			env.seed(11)
			agent = Agent(n_states=n_states,
					n_iter=n_iterations,
					env_name=ENV_NAME,
					action_bounds=action_bounds,
					n_actions=n_actions,
					lr=lr,
					device=device)			
		# random body
				
		# adversarial body
		if body == "adversarial":	
			if ENV_NAME == "Ant" and ATTACK == "length":
				df = pd.read_csv("./Csv_for_attack/a_l_di_005.csv")
				
			seed=time.time()
			rate=random.random()
			df1 = df.sample()
			adv_list = list(df1.iloc[0])
			
			if ENV_NAME == "Ant" and ATTACK == "length":
				make_adv_length_ant.set_adv_length(adv_list)

			env = gym.make(ENV_NAME + "-v2")
			env.seed(seed_id)
			agent = Agent(n_states=n_states,
					n_iter=n_iterations,
					env_name=ENV_NAME,
					action_bounds=action_bounds,
					n_actions=n_actions,
					lr=lr,
					device=device)	
		# adversarial body
										
		player = Play(env, agent, ENV_NAME)
		
		if i%200==0:
			print(i)
		reward = player.evaluate()
		
		reward_list.append(reward)
		with open("./Hist_"+ENV_NAME+"/"+ENV_NAME+time_info+str(num_rep)+".csv", 'a') as f:
			writer = csv.writer(f)
			writer.writerow([i,reward])
		
	
	avr_reward = sum(reward_list)/i
	std_reward = np.std(reward_list)
	reward_list.sort()
	reward_25 = np.percentile(reward_list,25)
	reward_50 = np.percentile(reward_list,50)
	reward_75 = np.percentile(reward_list,75)
	print("average:",avr_reward,"std",std_reward,"25%:",int(reward_25),"50%:",int(reward_50),"75%:",int(reward_75))
	info = ["0","0","0"]
	info[0] = ENV_NAME+" num:"+str(len(reward_list))+" avr:"+str((avr_reward))+" std:"+str((std_reward))+" 1/4:"+str((reward_25))+" 2/4(med):"+str((reward_50))+" 3/4:"+str((reward_75))
	info[1] = "size_ratio" + str(size_ratio)
	info[2] = "length_ratio"+str(length_ratio)
	
	graph_plot2.plot_hist2(reward_list,ENV_NAME,info,time_info) # hist write
	
	fig,ax = plt.subplots()
	bp = ax.boxplot(reward_list)
	#plt.show()


		
