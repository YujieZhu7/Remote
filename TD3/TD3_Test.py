import sys
import gym
import torch
import numpy as np
import random
import pickle
import TD3



#

# Load environment
env = gym.make('HalfCheetah-v2')
env_eval = gym.make('HalfCheetah-v2')
# Set seeds
seed =5
offset = 100
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

max_steps = 1e6
memory_size = 1e6 #5e5
step_eval = 1000


batch_size = 256
learning_starts = batch_size

replay_buffer = []
score_history = []
steps = 0
episodes = 0
episodes_eval = 10

# Record wandb metrics
#Hopper_dim = (400,300)
agent = TD3.Agent(state_dim, action_dim, max_action, hidden_dim=(400,300),lr=(1e-3,1e-3),batch_size=batch_size,
                  policy_noise = 0.2, device=device)

while steps < max_steps + 1:
    # Training #
    done = False
    state = env.reset()
    step_env = 0
    while not done:
        action = agent.choose_action(state)
        noise = np.random.normal(0, max_action*0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)
        next_state, reward, done, info = env.step(action)
        steps += 1
        step_env += 1
        if step_env == env._max_episode_steps:
            done_rb = False
            print("Max env steps reached")
        else:
            done_rb = done
        replay_buffer.append((state, action, reward, next_state, done_rb))
        state = next_state

        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

        if steps >= learning_starts:
            ### Train dynamics model  ###
            agent.train(replay_buffer)

        # Evaluation (every step_eval steps)
        env_eval.seed(seed+offset)
        env_eval.action_space.seed(seed+offset)
        if steps % step_eval == 0:
            score_temp = []
            for e in range(episodes_eval):
                done_eval = False
                state_eval = env_eval.reset()
                score_eval = 0
                while not done_eval:
                    with torch.no_grad():
                        action_eval = agent.choose_action(state_eval)
                        state_eval, reward_eval, done_eval, info_eval = env_eval.step(action_eval)
                        score_eval += reward_eval
                score_temp.append(score_eval)
            score_eval = np.mean(score_temp)
            score_history.append(score_eval)

            print("Episode", episodes, "Env Steps", steps, "Score %.2f" % score_eval)

    episodes += 1
np.save(f"Results/HalfCheetah/TD3_S{seed}",score_history)

# torch.save(agent.actor.state_dict(),
#                                    "/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/TD3_expert_actor_halfcheetah")
# torch.save(agent.actor_target.state_dict(),
#                                    "/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/TD3_expert_actortarget_halfcheetah")
# torch.save(agent.critic.state_dict(),
#                                    "/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/TD3_expert_critic_halfcheetah")
# torch.save(agent.critic_target.state_dict(),
#                                    "/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/TD3_expert_critictarget_halfcheetah")
