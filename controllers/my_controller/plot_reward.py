import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

for k in [1, 2,3]:
    plt.figure()
    robo_num = [1, 2, 3]
    EVAL_INTERVAL = 20
    for i in robo_num:
        reward_files = listdir(f"./reward/{i}")
        reward_files.sort()
        print(reward_files[-1])

        with open(os.path.join(f"./reward/{i}",reward_files[-k]),'r') as f:
            entire_f = f.read()
        l_reward = [eval(a) for a in entire_f.split()]
        # print(l_reward)
        mean_reward = []
        for i in range(EVAL_INTERVAL-1, len(l_reward),EVAL_INTERVAL):
            mean_reward.append( np.mean(l_reward[i-EVAL_INTERVAL+1:i]) )


        plt.plot(mean_reward)

plt.legend([f'robot {x}' for x in robo_num])
plt.show()