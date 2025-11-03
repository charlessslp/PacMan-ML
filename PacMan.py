import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

class Network(nn.Module):

  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
    self.bn4 = nn.BatchNorm2d(128)
    self.fc1 = nn.Linear(10 * 10 * 128, 512) # applying a math formula for flattening
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, action_size)

  def forward(self, state):
    x = F.relu(self.bn1(self.conv1(state)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = x.view(x.size(0), -1)           # flattening
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
    
    
import ale_py
import gymnasium as gym
env = gym.make('MsPacmanNoFrameskip-v0', full_action_space = False) # The MsPacman environment was renamed 'MsPacmanNoFrameskip-v0'
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)



learning_rate = 5e-4            # 0.0005 (that number to get to the bottom of the curve, if it's too great we can miss the sweet spot, if it's too small it could take forever or even stop in a small outcome, it seems this magic number was decided after trial and error)
minibatch_size = 64            # the number of observations before updating the model's parameters (good usual size)
discount_factor = 0.99          # gamma/alpha. small gamma/alpha (0.0001) = short sided (consider only considering current rewards). big gamma/alpha (1) consider future rewards more. For this case, future rewards are important


from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
  frame = Image.fromarray(frame) # transform the frame which comes in a numpy array into an actual image
  resizer = transforms.Resize((128,128)) # it's easier to resize the image from (210,160) (see the print of State shape above) to (128,128) so the convolutional layer of 32 can do it more easily
  tensor_maker = transforms.ToTensor() # this will convert the resized image into a pytorch tensor, which is needed for the agent to understand the image. This also normalize the image, meaning it will scale the pixel values into a value between 0 and 1
  preprocessor = transforms.Compose([resizer, tensor_maker]) # the object that takes in an image and compose a new one with the needed transformations (resizing + turn into pytorch tensor)
  return preprocessor(frame).unsqueeze(0) # something about the "batch"? that the dimension of the batch will be "the first dimension" (that's why the 0)



class Agent():

  def __init__(self, action_size): # creates the 2 neural networks. Also uses a famous optimization method called Adam and saves the the action_size (9). The memory is simplier than the Lunar Lander, only a deque
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device)
    self.target_qnetwork = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.replayMemory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done): # Given the event that just happened, save it into memory and get a batch of 100 experiences and learn from them.
    state = preprocess_frame(state) # before doing anything, remember to preprocess the image by resizing it to 128x128 and transform it into a pytorch tensor
    next_state = preprocess_frame(next_state) # same for the next state

    event = (state, action, reward, next_state, done)
    self.replayMemory.append(event)
    if len(self.replayMemory) > minibatch_size:
      experiences = random.sample(self.replayMemory, minibatch_size) # get a random sample from the deque memory by using the random lib
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.): # Given an state, calculates the Q-Values of the 9 possible actions. Epsilon is meant for e-greedy, so it sometimes do a random action instead.
    state = preprocess_frame(state).to(self.device)
    self.local_qnetwork.eval()                                      # puts network in evaluation mode, without gradients
    with torch.no_grad():                                           # this is to make calculations faster
      action_values = self.local_qnetwork(state)                    # calls forward() (see beginning of file) and gets the 9 Q-Values of the 9 possible actions
    self.local_qnetwork.train()                                     # sets it back to train mode
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())            # takes the action with higher Q-Value (Explotation)
    else:
      return random.choice(np.arange(self.action_size))             # takes a totally random action (Exploration)

  def learn(self, experiences, discount_factor): # Given a batch of experiences (see step method) 
    states, actions, rewards, next_states, dones = zip(*experiences)                        # this zip function will separate the experiences in different arrays. Have in mind the states and next states are already pytorch tensors after preprocess (see step() method)
    
    states = torch.from_numpy(np.vstack(states)).float().to(self.device) # vstack function can also take in a pytorch tensor to stack, returns a numpy. Then we have to transfrom that numpy to a pytorch tensor again
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
    
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)      # Get the Q values of the target network using the next states.
    q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))                  # Bellman's equation: calculation of the actual Q-Target based on the reward + the Q-value we had stored (we don't want to substitute it with reward but update it a bit)
    q_expected = self.local_qnetwork(states).gather(1, actions)                             # get the Q-Values expected from this state 
    loss = F.mse_loss(q_expected, q_targets)                                                # the loss is the difference between the actual q-value of the next state (based on reward) minus the Q-Value we thought we would get
    self.optimizer.zero_grad()                                                              # Back propagation using the "Adam" optimization.
    loss.backward()
    self.optimizer.step()



"""
agent = Agent(number_actions)

number_episodes = 2000
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilong_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilong_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 500.0:
    print('\n Enviroment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break


import glob
import io
import base64
import imageio
from IPython.display import HTML, display

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanNoFrameskip-v0') # The MsPacman environment was renamed 'MsPacmanNoFrameskip-v0'

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()

"""


# my stuff down below

videos_folder = 'generated_videos_ALL'
os.makedirs(videos_folder, exist_ok=True)

import glob
import io
import base64
import imageio
from IPython.display import HTML, display

def compute_video_of_model(agent, env_name, frames):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    MAX_FRAMES = 3000
    initial_frame_count = len(frames)
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
        
        if len(frames) - initial_frame_count >= MAX_FRAMES:
            done = True
    env.close()
    return frames

agent = Agent(number_actions)

number_episodes = 2000
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilong_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)

frames = []
print('\nComputing video episode 0')

for episode in range(0, 5):
    state, _ = env.reset()
    frames = compute_video_of_model(agent, 'MsPacmanNoFrameskip-v0', frames)
    video_name = f'video000.mp4'
    complete_path = os.path.join(videos_folder, video_name)
    
imageio.mimsave(complete_path, frames, fps=30)
frames = []

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break

  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilong_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))    
  
  if episode % 20 == 0:
    video_name = f'video{episode}.mp4'
    complete_path = os.path.join(videos_folder, video_name)
    imageio.mimsave(complete_path, frames, fps=30)
    frames = []
  
  if np.mean(scores_on_100_episodes) >= 700.0:
    print('\n Enviroment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    break
    
  frames = compute_video_of_model(agent, 'MsPacmanNoFrameskip-v0', frames)

torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')

def show_total_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_total_video()