
import torch
import os
import glob
import gym
import re
from lib.Model import ActorCritic



def loadModel():
  list_of_files = glob.glob('./'+'checkpoints/*') 
  model_name = max(list_of_files, key=os.path.getctime)
  print('Loading the following file:\n')
  print(model_name)
  return model_name

def test_env(env, model, device, deterministic=True):
    state = env.reset()
    done = False
    total_reward = 0
    i = 0
    while (not done) and (i<2048):
        env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        i +=1
    env.close()
    print(total_reward)
    return total_reward


HIDDEN_SIZE = 256
device = torch.device("cuda")
env = gym.make('BipedalWalker-v2')
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
model_name = loadModel()
model.load_state_dict(torch.load(model_name))
for i in range(30):
    test_env(env,model,device,deterministic = False)
