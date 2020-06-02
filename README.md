[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree  
udacity-deep_rl-navigation

![Trained Agent][image1]


### Project Description

This project is part of the Udacity Deep Reinforcement Learning Nanodegree.
The agent can apply four discrete actions to move through a 2D square plane:
`move forward`, `move backward`, `turn left`, `turn right`.
The agent can accumulate rewards (`+1`) by collecting yellow bananas 
but will be penalized (`-1`) for collecting blue bananas.

Each frame is observed in the form of a 37-dimensional state vector encoding
the agent's velocity and ray-based perception information.

The agent learns from experience through repeated interaction with the
Unity simulation environment.


### Results

The environment is considered solved when the agent accumulates an average reward 
of +13 per episode.  

Using a Deep Q-Network with hidden layer sizes [74,37,16,8] to approximate the action-value function,
the episode can be solved in about 300 episodes.

![Reward evolution][graphics/reward_evolution.png]


### Installation

*TODO: virtualenv installation*

Download Unity simulator for Linux [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
or directly through the command line

``` sh
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip  # with visualization
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip  # no visualization
```
and unzip them in the root directory of this repository (simulator files for 
[MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) and
[Windows](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
).



### Running Tests

Run the complete test suite with the command

```
python -m unittest qlearning tests
```



<!-- ### (Optional) Challenge: Learning from Pixels -->

<!-- After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels! -->

<!-- To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.) -->

<!-- You need only select the environment that matches your operating system: -->
<!-- - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip) -->
<!-- - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip) -->
<!-- - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip) -->
<!-- - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip) -->

<!-- Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent. -->

<!-- (_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above. -->
