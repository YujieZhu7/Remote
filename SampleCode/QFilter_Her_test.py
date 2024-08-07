import sys
import gymnasium as gym
import torch
import numpy as np
import random
import pickle
import TD3_Ensemble_NEW_HER_QFilter as TD3
# from Algorithms import TD3_Ensemble_NEW_HER_QFilter as TD3


# process the inputs
clip_range = 5
clip_obs = 200
clip_return = 50
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, ax = 0):
    o_clip = np.clip(o, -clip_obs, clip_obs)
    g_clip = np.clip(g, -clip_obs, clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std+1e-6), -clip_range, clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std+1e-6), -clip_range, clip_range)
    inputs = np.concatenate([o_norm, g_norm], axis = ax)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs



### running mean implementation according to Welford's algorithm
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (mean, np.ones_like(mean), np.ones_like(mean))
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
########################################################################

# Load environment
env_name = 'FetchPush'
env = gym.make('FetchPush-v2')
env_train = gym.make('FetchPush-v2')
env_eval = gym.make('FetchPush-v2')
method = "MCDropout"
if method == "MCDropout":
    drop_rate = 0.1
else:
    drop_rate = 0.0

steps_accept = 0
ensemble_size = 2
# Set seeds
seed =5
offset = 100
env.reset(seed =seed)
env.action_space.seed(seed)
env_train.reset(seed = seed)
env_train.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
obs_dim = state_dim+goal_dim
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
#0.01 == Expert data, 0.5 == 62% success rate (non-expert)
open_file =  open(f"/home/zhu_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/pythonProject/Data/{env_name}/DemoData_test0.5+1.pkl", "rb")
dataset = pickle.load(open_file)
open_file.close()

gamma = 0.98

demos = []
states_agg = (0, np.zeros(state_dim), np.zeros(state_dim))  # (count, mean, M2)
goals_agg = (0, np.zeros(goal_dim), np.zeros(goal_dim))
for i in range(len(dataset)):
    demos.append((dataset[i][0], dataset[i][1], dataset[i][2], dataset[i][3], dataset[i][4], dataset[i][5]))
    states_agg = update(states_agg, np.array(dataset[i][0]))
    goals_agg = update(goals_agg, np.array(dataset[i][4]))

max_steps = 4e6
memory_size = 1e6 #5e5
step_eval = 50


batch_size = 1024
learning_starts = 2000

replay_buffer = []
score_history = []
success_history = []
percent_accept_demos = []
steps = 0
episodes = 0
episodes_eval = 25
eps_eval = 10 # Evaluate every 10 episodes
model_iters = 5
lambda_BC = 1
# Record wandb metrics
#Hopper_dim = (400,300)
# agent = TD3.Agent(obs_dim, action_dim, max_action, hidden_dim=(256,256,256),lr=(1e-3,1e-3),batch_size=batch_size,
#                   policy_noise = 0.2, device=device)
agent = TD3.Agent(state_dim, goal_dim, action_dim, max_action, hidden_dim=(256,256,256,256),method = method,
                  ensemble_size = ensemble_size, lmbda1=0.001, lmbda2 = 1/128, batch_size_buffer=1024,
                  batch_size_demo = 128, gamma=0.98, tau=0.005, lr=(1e-3,1e-3), policy_noise=0.2, noise_clip=0.5,
                  policy_freq=2, device=device)

num_accept_demos = 0
num_tot_demos = 0
while steps < max_steps + 1:
    # Training #
    done = False
    step_env = 0
    obs_ep = []
    # obs = env_train.reset(seed=int(np.random.randint(1e6, size=1)[0]))[0]
    obs = env_train.reset()[0]
    state = obs['observation']
    desired_goal = obs['desired_goal']
    observation = np.concatenate([state, desired_goal])
    states_agg = update(states_agg, np.array(state))
    goals_agg = update(goals_agg, np.array(desired_goal))

    while not done:
        state_stats = finalize(states_agg)
        goal_stats = finalize(goals_agg)
        inputs = process_inputs(state, desired_goal, o_mean=state_stats[0], o_std=np.sqrt(state_stats[1]),
                                g_mean=goal_stats[0], g_std=np.sqrt(goal_stats[1]))
        action = agent.choose_action(inputs)
        noise = np.random.normal(0, max_action*0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)
        next_obs, reward, terminated, done, info = env_train.step(action)
        next_state = next_obs['observation']
        next_desired_goal = next_obs['desired_goal']

        steps += 1
        step_env += 1
        if step_env == env_train._max_episode_steps:
            done_rb = False
            # print("Max env steps reached")
        else:
            done_rb = done
        states_agg = update(states_agg, np.array(next_state))
        goals_agg = update(goals_agg, np.array(desired_goal))
        replay_buffer.append((state, action, reward, next_state, desired_goal, done_rb))

        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)
        obs_ep.append((obs, action, next_obs, info))
        obs = next_obs
        state = next_state


    substitute_goal = obs["achieved_goal"].copy()
    for i in range(len(obs_ep)):
        observation, action, next_observation, info = obs_ep[i]
        state = observation['observation']
        states_agg = update(states_agg, np.array(state))
        goals_agg = update(goals_agg, np.array(substitute_goal))
        obs = np.concatenate([state, substitute_goal])
        next_state = next_observation['observation']
        next_obs = np.concatenate([next_state, substitute_goal])
        substitute_reward = env.compute_reward(observation["achieved_goal"], substitute_goal, info)
        substitute_terminated = env.compute_terminated(observation["achieved_goal"], substitute_goal, info)
        substitute_truncated = env.compute_truncated(observation["achieved_goal"], substitute_goal, info)
        done_HER = False #done flag is always false
        replay_buffer.append((state, action, substitute_reward, next_state, substitute_goal ,substitute_terminated))
        if len(replay_buffer) > memory_size:
            replay_buffer.pop(0)

    if len(replay_buffer)> learning_starts:
        state_stats = finalize(states_agg)
        goal_stats = finalize(goals_agg)
        agent.train_buffer(replay_buffer, normalizers = (state_stats[0], np.sqrt(state_stats[1]),goal_stats[0],
                                                  np.sqrt(goal_stats[1])), iterations = 2)
        if steps < steps_accept:
            num_accept, num_total = agent.train_demos(demos, normalizers=(state_stats[0], np.sqrt(state_stats[1]),
                                                                       goal_stats[0], np.sqrt(goal_stats[1])), iterations=2, BC_only = True)
        else:
            num_accept, num_total = agent.train_demos(demos, normalizers=(state_stats[0], np.sqrt(state_stats[1]),
                                                                          goal_stats[0], np.sqrt(goal_stats[1])), iterations=2, BC_only = False)
        num_accept_demos += num_accept
        num_tot_demos += num_total

    # Evaluation (every step_eval steps)
    env_eval.reset(seed = seed+offset)
    env_eval.action_space.seed(seed+offset)
    if episodes % eps_eval == 0:
        score_temp = []
        fin_temp =[]
        for e in range(episodes_eval):
            done_eval = False
            obs_eval = env_eval.reset()[0]
            state_eval = obs_eval['observation']
            desired_goal_eval = obs_eval['desired_goal']
            # observation_eval = np.concatenate([state_eval, desired_goal_eval])
            score_eval = 0
            steps_eval = 0
            while not done_eval:
                with torch.no_grad():
                    state_stats = finalize(states_agg)
                    goal_stats = finalize(goals_agg)
                    inputs = process_inputs(state_eval, desired_goal_eval, o_mean = state_stats[0],o_std = np.sqrt(state_stats[1]),
                                g_mean = goal_stats[0], g_std = np.sqrt(goal_stats[1]))
                    action_eval = agent.choose_action(inputs)
                    obs_eval, reward_eval, terminated_eval, done_eval, info_eval = env_eval.step(action_eval)
                    steps_eval+=1
                    state_eval = obs_eval['observation']
                    desired_goal_eval = obs_eval['desired_goal']
                    # observation_eval = np.concatenate([state_eval, desired_goal_eval])
                    score_eval += reward_eval
            fin_eval = info_eval['is_success']
            score_temp.append(score_eval)
            fin_temp.append(fin_eval)
        score_eval = np.mean(score_temp)
        fin_eval = np.mean(fin_temp)
        score_history.append(score_eval)
        success_history.append(fin_eval)
        if num_tot_demos == 0:
            percent_accept_demos.append(0)
        else:
            percent_accept_demos.append((num_accept_demos / num_tot_demos))
        print("Episode", episodes, "Env Steps", steps, "Score %.2f" % score_eval, "Success rate %.2f" %fin_eval)
        print("Percentage acceptance of Demos = %.2f " % (100 * percent_accept_demos[-1]))
        num_accept_demos = 0
        num_tot_demos = 0

    episodes += 1
# np.save(f"Results/{env_name}/QFilter_TD3HER_NonExpert_Method_{method}_EnsSize_{ensemble_size}_S{seed}",success_history)
# np.save(f"Results/{env_name}/QFilter_TD3HER_NonExpert_Method_{method}_EnsSize_{ensemble_size}_%acceptdemos_S{seed}", percent_accept_demos)

# torch.save(agent.actor.state_dict(),
#                                    f"/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/{env_name}/TD3_expert_actor")
# torch.save(agent.actor_target.state_dict(),
#                                    f"/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/{env_name}/TD3_expert_actortarget")
# torch.save(agent.critic.state_dict(),
#                                    f"/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/{env_name}/TD3_expert_critic")
# torch.save(agent.critic_target.state_dict(),
#                                    f"/home/hepbur_c@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/PhD/Demonstrations/Models/{env_name}/TD3_expert_critictarget")
