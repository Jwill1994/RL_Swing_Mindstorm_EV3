
import socketio
import asyncio
from time import sleep
import numpy as np
from paramiko import SSHClient as SSH
import os
import random
import argparse
from collections import deque
import torch
import torch.optim as optim
from utils import *
from model import Actor, Critic
from tensorboardX import SummaryWriter
from time import sleep
import pickle 
sio = socketio.AsyncClient()
loop = asyncio.get_event_loop()
ssh = SSH()
ssh.load_system_host_keys()
ssh.connect("169.254.159.129", username="robot", password="maker") #for swing init motor

parser = argparse.ArgumentParser()
#parser.add_argument('--env_name', type=str, default="Pendulum-v0")
parser.add_argument('--load_epi', type=str, default=None)
parser.add_argument('--done_angle',type=int, default=40)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=32) 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--alpha_lr', type=float, default=1e-4)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=-300) 
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor, critic, target_critic, mini_batch, 
                actor_optimizer, critic_optimizer, alpha_optimizer,
                target_entropy, log_alpha, alpha):
    mini_batch = np.array(mini_batch)
    #print(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    #print('rewards',rewards)
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4])

    actions = torch.Tensor(actions).squeeze(1)
    #print('actions',actions)
    rewards = torch.Tensor(rewards).squeeze(1)
    
    masks = torch.Tensor(masks)

    # update critic 
    criterion = torch.nn.MSELoss()
    
    # get Q-values using two Q-functions to mitigate overestimation bias
    q_value1, q_value2 = critic(torch.Tensor(states), actions)

    # get target
    mu, std = actor(torch.Tensor(next_states))
    next_policy, next_log_policy = eval_action(mu, std)
    target_next_q_value1, target_next_q_value2 = target_critic(torch.Tensor(next_states), next_policy)
    
    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    min_target_next_q_value = min_target_next_q_value.squeeze(1) - alpha * next_log_policy.squeeze(1)
    target = rewards + masks * args.gamma * min_target_next_q_value
    #print("q_value1", q_value1, type(q_value1))
    #print("q_value1 squeeze(1)", q_value1.squeeze(1),type(q_value1.squeeze(1)))
    #print("target", target)
    #print("target_detach", target.detach())
    #sleep(10)
    critic_loss1 = criterion(q_value1.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss1.backward()
    critic_optimizer.step()

    critic_loss2 = criterion(q_value2.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss2.backward()
    critic_optimizer.step()

    # update actor 
    mu, std = actor(torch.Tensor(states))
    policy, log_policy = eval_action(mu, std)
    
    q_value1, q_value2 = critic(torch.Tensor(states), policy)
    min_q_value = torch.min(q_value1, q_value2)
    
    actor_loss = ((alpha * log_policy) - min_q_value).mean() 
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # update alpha
    alpha_loss = -(log_alpha * (log_policy + target_entropy).detach()).mean() 
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha = torch.exp(log_alpha) 
    
    return alpha

@sio.event
async def connect():
    print('connection OK')

    await init()

@sio.event
async def disconnect():
    global episode
    save_model(episode)
    print("save model disconnecting : epi -", episode)
    print('disconnected from server')

def env_reset():
    #high = np.array([np.pi*0.5, 1]) #until 90 deg 
    #state = np.random.uniform(low=-high, high=high)
    # @ init standard: rate > 5   :  ang:-10, rate:20 
    state = np.array([float(0),np.cos(float(np.radians(12))), np.sin(float(np.radians(12))), float(np.radians(25))])
    #state = np.array([1, float(np.radians(12)), float(np.radians(25))])
    last_u = None
    return state 
def load_model():
    global actor, critic, target_critic, replay_buffer, episode
    episode = args.load_epi
    
    with open(args.load_epi+'replayBuffer.txt', 'rb') as f:
        replay_buffer = pickle.load(f)
    f.close()

    print('pre-existing replayBuffer loaded!!')
    print('actor before loaded:',actor.state_dict())
    load_epi = args.load_epi 
    actor_path = args.save_path + str(load_epi) + 'actor_model.pth.tar'
    critic_path = args.save_path + str(load_epi) + 'critic_model.pth.tar'
    target_critic_path = args.save_path + str(load_epi) + 'target_critic_model.pth.tar'
    print('Loading models from {} and {}'.format(actor_path, critic_path))
    if actor_path is not None:
        actor.load_state_dict(torch.load(actor_path))
    if critic_path is not None:
        critic.load_state_dict(torch.load(critic_path))
    if target_critic_path is not None:
        target_critic.load_state_dict(torch.load(target_critic_path))
    print('actor after loaded:',actor.state_dict())
def save_model(episode):
    global actor, critic, target_critic, replay_buffer
    with open(str(episode)+'replayBuffer.txt','wb') as f :
        pickle.dump(replay_buffer, f)
    f.close() 

    actor_path = args.save_path + str(episode) + 'actor_model.pth.tar'
    torch.save(actor.state_dict(), actor_path)
    critic_path = args.save_path + str(episode) + 'critic_model.pth.tar'
    torch.save(critic.state_dict(), critic_path)
    target_critic_path = args.save_path + str(episode) + 'target_critic_model.pth.tar'
    torch.save(target_critic.state_dict(), target_critic_path)
replay_buffer = deque(maxlen=100000)
recent_rewards = deque(maxlen=32) #100
steps = 0
done = False
score = []
mask = 1
torch.manual_seed(500)
high = np.array([1., 1., 10], dtype=np.float32)
#state_size =  spaces.Box(low=-high, high=high, dtype=np.float32).shape[0]
#action_size = spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32).shape[0]
state_size = 4
action_size = 1
actor = Actor(state_size, action_size, args)
critic = Critic(state_size, action_size, args)
target_critic = Critic(state_size, action_size, args)
actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
hard_target_update(critic, target_critic)
# initialize automatic entropy tuning
target_entropy = -torch.prod(torch.Tensor(action_size)).item()
log_alpha = torch.zeros(1, requires_grad=True)
alpha = torch.exp(log_alpha)
alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
writer = SummaryWriter(args.logdir)
episode = 0

@sio.event
async def init(data=1): ############
    global args, episode  
    global replay_buffer, recent_rewards, steps, done, score, mask, state_size, action_size 
    global actor,critic,target_critic,actor_optimizer,critic_optimizer,target_entropy,log_alpha,alpha,alpha_optimizer,writer
    ####env reset####

    if args.load_epi is not None: 
        load_model()
        print("pre existing model, ",args.load_epi,"loaded")
    
    steps = 0 
    if args.load_epi != None : 
        if episode > args.load_epi :
            #recent_rewards.append(score)
            print('{} episode | epi_rewards_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
            writer.add_scalar('log/epi_rewards_sum', np.sum(recent_rewards), episode)        
            mini_batch = random.sample(replay_buffer, args.batch_size)    
            actor.train(), critic.train(), target_critic.train()
            alpha = train_model(actor, critic, target_critic, mini_batch, 
                                actor_optimizer, critic_optimizer, alpha_optimizer,
                                target_entropy, log_alpha, alpha)
            soft_target_update(critic, target_critic, args.tau)
            print("episode++ and weights updated!!!")
    else : 
        if episode > 0 :
            #recent_rewards.append(score)
            print('{} episode | epi_rewards_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
            writer.add_scalar('log/epi_rewards_sum', np.mean(recent_rewards), episode)        
            mini_batch = random.sample(replay_buffer, args.batch_size)    
            actor.train(), critic.train(), target_critic.train()
            alpha = train_model(actor, critic, target_critic, mini_batch, 
                                actor_optimizer, critic_optimizer, alpha_optimizer,
                                target_entropy, log_alpha, alpha)
            soft_target_update(critic, target_critic, args.tau)
            print("episode++ and weights updated!!!")

    args.load_epi = None # for just once 
    print('episode:', episode)
    episode += 1
    state = env_reset()
    state = np.reshape(state, [1, state_size])
    print("socket start!!")
    mu, std = actor(torch.Tensor(state))
    action = get_action(mu, std)
    stdin, stdout, stderr = ssh.exec_command('python3 -c "from ev3dev2.motor import LargeMotor, OUTPUT_A, SpeedPercent; LargeMotor(OUTPUT_A).on_for_degrees(-100,360);"')
    await sio.emit("motor_on", action.flatten().tolist()) 
    await sio.wait()
'''
def angle_normalize(x):
    return (((x+np.pi/2) % (np.pi)) + np.pi/2)

# min : 0 / max : 4
def total_energy(ang,vel):
    return 4*(1-np.cos((float(np.radians(ang))))) + .5*float(np.radians(vel))**2 
'''
#prior ang, prior rate, post ang, rate, action 
@sio.event
async def resp(states):
    print('states',states)
    global args 
    global replay_buffer, recent_rewards, steps, done, score, mask, state_size, action_size, episode
    global actor,critic,target_critic,actor_optimizer,critic_optimizer,target_entropy,log_alpha,alpha,alpha_optimizer,writer
    steps += 1
    print('steps:',steps)
    # speed no sin func ? 
    state = np.array([float(steps),np.cos(float(np.radians(states[0]))), np.sin(float(np.radians(states[0]))), float(np.radians(states[1]))]) # /5 for scale down 
    #state = np.array([float(steps),float(np.radians(states[0])), float(np.radians(states[1]))]) # /5 for scale down 

    #state = np.array([np.cos(float(np.radians(states[0]))), np.sin(float(np.radians(states[0]))), float(np.radians(states[1]))]) # /5 for scale down 

    state = np.reshape(state, [1, state_size])
    next_state = np.array([float(steps), np.cos(float(np.radians(states[2]))), np.sin(float(np.radians(states[2]))), float(np.radians(states[3]))])
    #next_state = np.array([np.cos(float(np.radians(states[2]))), np.sin(float(np.radians(states[2]))), float(np.radians(states[3]))])
    #next_state = np.array([float(steps),float(np.radians(states[2])), float(np.radians(states[3]))]) # /5 for scale down 
    next_state = np.reshape(next_state, [1, state_size])
    mu, std = actor(torch.Tensor(next_state))
    action = get_action(mu, std)
    done = True if max(abs(states[0]),abs(states[2])) > args.done_angle else False
    #costs = 2*angle_normalize(float(np.radians(states[0])))**2 + .1*float(np.radians(states[1]))**2 
    #costs = angle_normalize(float(np.radians(states[0]))) / 3  #cuz this problem is different from inverted pendulum / ort states[2] - states[0] ?
    #costs = 1 - abs(states[0])/90
    #4 ~ 9.8 * 0.4 ( = g * h)
    #cost_prior = 4*(1-np.cos((float(np.radians(states[0]))))) + 33*float(np.radians(states[1]))**2   # 0.6437 / 0.009738 =66 (*0.5 = 33)
    #cost_post = 4*(1-np.cos((float(np.radians(states[2]))))) + 33*float(np.radians(states[3]))**2 
    #cost = cost_post - cost_prior
    #if cost > 0 :
    #    reward = np.array([0])
    #else :  
    #    reward = np.array([cost]) # why - sign ?
    reward = np.array([-1])
    if max(abs(states[0]),abs(states[2])) > 20 :
        reward = np.array([-0.8])
    if max(abs(states[0]),abs(states[2])) > 25 :
        reward = np.array([-0.6])
    if max(abs(states[0]),abs(states[2])) > 30 :
        reward = np.array([-0.4])
    if max(abs(states[0]),abs(states[2])) > 35 :
        reward = np.array([-0.2])
    if done:
        reward = np.array([0])
        print('training done, let me save model!! done angle is : ',max(abs(states[0]),abs(states[2])))
        save_model(episode)
        #args.done_angle += 5 
    recent_rewards.append(reward)
    #reward = -1. if not done else 0.
    mask = 0 if done else 1
    replay_buffer.append((state, action, reward, next_state, mask))
    
    #done = False if abs(states[0]) < 40 else True 
    
    print('s,ns,rew:',state,next_state,reward)
    
    #if steps % args.batch_size == 0:   # does it needed to be changed ? every episode ?

    '''
    #if episode % args.log_interval == 0:
        #recent_rewards.append(score)
        #print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
        #writer.add_scalar('log/score', float(score), episode)
        
        while(True):
            print("your target angle accomplished...")
            sleep(3)
        
    
    if episode % args.log_interval == 0:
        #recent_rewards.append(score)
        print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
        writer.add_scalar('log/score', float(score), episode)
    
    #if np.mean(recent_rewards) > args.goal_score:
    
    if episode % 100 == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        ckpt_path = args.save_path + str(episode) + 'model.pth.tar'
        torch.save(actor.state_dict(), ckpt_path)
        print('100 episode++ . So save weight')  #-30000 : customed # 

    if np.mean(recent_rewards) > args.goal_score:
    #if episode % 100 == 0
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        ckpt_path = args.save_path + str(recent_rewards) + 'model.pth.tar'
        torch.save(actor.state_dict(), ckpt_path)
        print('Recent rewards exceed -30000. So save weight') 
    '''
    await sio.emit("motor_on", action.flatten().tolist())
    await sio.wait()


    
async def start_server():
    await sio.connect('http://192.168.0.5:8080') 
    await sio.wait()

if __name__ == '__main__':
    loop.run_until_complete(start_server())



