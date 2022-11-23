import numpy as np
import os
import torch
import time
from ddpg import Actor, Critic

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(os.getcwd())
dirModels = './ddpgModels/'
dirFedModels = './ddpgModels/fed/'

#'./ddpgModels/1/02-11-2021-16-52-47/actor.pth'
def federation(total, current_time, current_episode):
    # synchronization process
    
    actor_parameters = []
    critic_parameters = []
    global dirModels
    global dirFedModels
    try:
        os.mkdir(dirFedModels + '/' + str(current_time))
    except:
        pass

    for i in range(1, total+1):
        actor_parameters.append(torch.load(dirModels + str(i) + '/' + str(current_time) + '/actor_'+str(current_episode)+'.pth')) 
        critic_parameters.append(torch.load(dirModels + str(i) + '/' + str(current_time) + '/critic_'+str(current_episode)+'.pth'))

    
    actor_model = actor_parameters[0]
    for i in range(1, total):
        for key in actor_model:
            actor_model[key] += actor_parameters[i][key]

    critic_model = critic_parameters[0]
    for i in range(1, total):
        for key in critic_model:
            critic_model[key] += critic_parameters[i][key]

    for key in actor_model:
        actor_model[key] /= total
    for key in critic_model:
        critic_model[key] /= total
    
    
    torch.save(actor_model, dirFedModels + str(current_time) + '/' + 'actor_'+str(current_episode)+'.pth')
    torch.save(critic_model, dirFedModels + str(current_time) + '/' + 'critic_'+str(current_episode)+'.pth')

    with open('fed_control.txt', 'w') as f:
        f.write('0')

    print("Federation completed, all robots may proceed training...")



if __name__ == '__main__':
    federation(3, '02-11-2021-20-44-32')