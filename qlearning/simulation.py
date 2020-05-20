import numpy as np
import time
from colorama import Fore, Style


def moving_average(x, window_size):
    """
    compute the moving average with a given window_size (equivalent to a convolution)
    """
    return np.convolve(x, np.ones((window_size))/window_size, mode='valid')


def simulate(environment, agent, brain_name, learn=True, n_episodes=1, window_size=100):
    print(
        Fore.GREEN, "# Training" if learn else "# Evaluation",
        " ({} episodes)".format(n_episodes), Style.RESET_ALL
    )
    print(Fore.CYAN, "Agent: {}".format(agent.__class__.__name__),  Style.RESET_ALL)
    scores = []
    t_start = time.perf_counter()
    for i in range(n_episodes):
        env_info = environment.reset(train_mode=learn)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = env_info.local_done[0]
        while not done:  # while episode is not done
            action = agent.get_action(state)                # determine action
            env_info = environment.step(action)[brain_name]         # apply action
            next_state = env_info.vector_observations[0]    # get the next state
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]
            agent.observe(state, action, reward, next_state, done)
            score += reward                                 # update the score
            state = next_state                              # advance state

        scores.append(score)
        current_score = np.mean(scores[max(0, len(scores)-window_size):])

        # output
        print(
            "\r",  # clear line
            "   episode {:6d}/{:d}".format(i+1, n_episodes),
            "   {:6.2f}%".format((i+1)/n_episodes*100),
            "   avg. score: {:+6.2f}".format(np.mean(scores)),
            "   current score: ", Fore.RED,
            "{:+6.2f}".format(current_score),
            "                          ",
            "\n" if (i+1)%window_size == 0 or (i+1)==n_episodes else "",
            Style.RESET_ALL, end=''
        )

    t_end = time.perf_counter()
    print("done ({:.2f}s)\n".format(t_end-t_start))

    return scores
