              
from PIL.Image import coerce_e
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.weight': 'bold'})

std_mn_array = [
[4378.08601874254, 140.77433397932523],
[4101.907019567675, 146.15250679814378],
[4165.905561616521, 116.84475528558013],
[4410.16322798502, 119.57133440696832],
[4309.167712073669, 156.78323859816976],
[4361.037888798672, 56.92343243615098],
[4444.677203859725, 92.7261195827413],
[3943.185319556332, 98.47953900942304],
[4304.816954780604, 80.64603987118095],
[4268.017235413873, 152.57377002688122],
[3935.2485942637873, 163.0533561241104],
[4449.23541022094, 181.56855968264543],
[4297.427360948148, 172.26783708993932],
[4173.687422530407, 198.74278107680013]]

colorst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#130221','#05fa2a','#f2fa05','#fa05f2']
laels = [ 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
datap = np.load('/home/n10370986/Dreamer/data/segmented/Subject10/Subject_10_Video_15_3_2_4_Segment78.0_EEG_STIM.npy',allow_pickle=True)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for i in range(14):
    plt.plot(datap[i].tolist(),linestyle='-',color=colorst[i],label=laels[i])
plt.ylabel('Loss',weight = 'bold')
plt.xlabel('Samples',weight = 'bold')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("Channel.pdf",bbox_inches='tight',dpi=3000)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
for i in range(14):
    plt.plot(((datap[i]-std_mn_array[i][0])/std_mn_array[i][1]).tolist(),linestyle='-',color=colorst[i],label=laels[i])
plt.ylabel('Loss',weight = 'bold')
plt.xlabel('Samples',weight = 'bold')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.weight': 'bold'})
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("ChannelNorm.pdf",bbox_inches='tight',dpi=3000)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# for i in range(14):
#     plt.plot(datap[1].tolist(),linestyle='-',color=colorst[i],label=laels[i])
#     break
# plt.ylabel('Loss',weight = 'bold')
# plt.xlabel('Epochs',weight = 'bold')

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig("onechannel.pdf",bbox_inches='tight',dpi=3000)