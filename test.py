import argparse
import numpy as np
from environment import Environment
import matplotlib.pyplot as plt
import matplotlib.animation as animation

seed = 10000
def parse():
    parser = argparse.ArgumentParser(description="cell RL Project")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards_each_episode = []
    rewards_each_step = []
    env.seed(seed)
    for i in range(total_episodes):
        state,target,observation_method = env.reset()
        all_state_buffer = [state[:][:][:]];
        all_step_reward_buffer = [];
        all_step_action_buffer = [];
        done = False
        episode_reward = 0.0
        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            all_state_buffer.append(state[:][:][:]);
            all_step_reward_buffer.append(reward)
            all_step_action_buffer.append(action)

        all_step_reward_buffer = np.array(all_step_reward_buffer)
        all_step_action_buffer = np.array(all_step_action_buffer)
        rewards_each_episode.append(episode_reward)
        rewards_each_step.append(all_step_reward_buffer)
        if i == 9:
            movement_animation(i,all_state_buffer,all_step_reward_buffer,all_step_action_buffer,target,observation_method)
    print('Run %d episodes'%(total_episodes))
    print("Reward from each episode:",rewards_each_episode)
    print('Mean:', np.mean(rewards_each_episode))

def run(args):
    if args.test_dqn:
        env = Environment('cell-RL', args, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=50)

def movement_animation(episode_i,all_state_buffer,all_step_reward_buffer,all_step_action_buffer,goal,observation_method):
    grid_SIZE = 10
    fig, ax = plt.subplots()
    ax.plot([goal.x],[goal.y],'go')
    ax.axis([0,grid_SIZE,0,grid_SIZE])
    l, = ax.plot([],[],'ro')
    def animate(step_i):
        if step_i > 0:
            del fig.texts[0:len(fig.texts)]
        ax.plot([goal.x],[goal.y],'go')
        if observation_method == 0:
            l.set_data(all_state_buffer[step_i-1][0,0,:], all_state_buffer[step_i-1][0,1,:])
        elif observation_method == 1:
            l.set_data(goal.x-all_state_buffer[step_i-1][0,0,:], goal.y-all_state_buffer[step_i-1][0,1,:])
        fig.text(0.1, 0.9, 'Step Reward:'+str(all_step_reward_buffer[step_i-1]), size=10, color='purple')
        fig.text(0.5, 0.9, 'action:'+str(all_step_action_buffer[step_i-1]), size=10, color='purple')
    ani = animation.FuncAnimation(fig, animate, frames=len(all_state_buffer))
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())
    f = r"animation/cell_movement_animation"+str(episode_i)+".gif" 
    writergif = animation.PillowWriter(fps=4) 
    ani.save(f, writer=writergif)

if __name__ == '__main__':
    args = parse()
    run(args)