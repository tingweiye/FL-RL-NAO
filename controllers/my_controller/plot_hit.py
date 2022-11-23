import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

for k in [1,2,6,7,8]:
    plt.figure()
    plt.ylim([0,1])
    robo_num = [1, 2, 3]
    EVAL_INTERVAL = 50
    for i in robo_num:
        hit_files = listdir(f"./hit/{i}")
        hit_files.sort()
        print(hit_files[-1])

        with open(os.path.join(f"./hit/{i}",hit_files[-k]),'r') as f:
            entire_f = f.read()
        l_hit = [eval(a) for a in entire_f.split()]
        # print(l_hit)
        mean_hit = []
        for i in range(EVAL_INTERVAL-1, len(l_hit),EVAL_INTERVAL):
            mean_hit.append( np.mean(l_hit[i-EVAL_INTERVAL+1:i]) )


        plt.plot(mean_hit)

    plt.legend([f'robot {x}' for x in robo_num])
plt.show()