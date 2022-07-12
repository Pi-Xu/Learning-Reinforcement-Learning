import gym
import matplotlib.pyplot as plt 
import time

env = gym.make('MountainCar-v0')

#############################################

obs_space = env.observation_space # 对应$O$
action_space = env.action_space # 对应$A$
print(f"The observation space: {obs_space}")
print(f"The action space: {action_space}")

#############################################

# 得到observation
obs = env.reset() # 查看return就可以知道

action = action_space.sample()
print(f"action: {action}(由前面可以了解到, action space的大小是3)")

# 使用step之后可以得到的信息
new_obs, reward, done, info = env.step(action)

print(f'obs: {obs} -> new_obs: {new_obs}')

env.render('human')
env_screen = env.render(mode = 'rgb_array')

plt.imshow(env_screen)
#############################################

# Number of steps you run the agent for 
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()
    
    # apply the action
    obs, reward, done, info = env.step(action)
    
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)
    
    # If the epsiode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()
