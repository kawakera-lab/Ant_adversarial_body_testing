import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import time
import datetime

def plot_hist2(y_list,env_name,info,time_info):
	plt.rcParams["font.size"] = 16	
	fig = plt.figure(figsize=(10,10))
	ax2 = fig.add_subplot(1,1,1)
	plt.title(env_name)
	ax2.set_xlabel("reward")
	ax2.set_ylabel("num")
	ax2.hist(y_list)
	
	fig.savefig("./Hist_"+env_name+"/"+env_name+time_info+str(len(y_list))+".png")
	path = "./Hist_"+env_name+"/"+env_name+time_info+str(len(y_list))+".txt"
	f = open(path,"a")
	f.write(info[0]+"\n\n")
	f.write(info[1]+"\n\n")
	f.write(info[2]+"\n\n")
	f.write("\n\n")
