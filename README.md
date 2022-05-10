# DeepRL-Tennis-Project


## General 
The goal of this project is to train two agents, are controlling two rackets with the aim to to bounce a ball over a net. The environment is considered solved when an average score of 0.5 over 100 episodes has been reached. The agents get +0.1 points for successfully bouncing the ball over the net and -0.01 for letting it drop.

## Project environment
The environment used is based on but not entirely similar UnityML's Tennis environment. This special env is provided by Udacity (see below).

<img width="1358" alt="Bildschirmfoto 2022-05-10 um 22 37 26" src="https://user-images.githubusercontent.com/23191357/167717562-3b1674a6-0d71-461f-b953-b23d691d7b78.png">

### Environment details
In this environment, both rackets can move independently. 

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket, whereas each racket/agent receives its own local observation. Each action is a vector with two continuous numbers, corresponding to movement in relation to the net, and jumping.

## Method

The approach of my implementation considers the following elements in the model:
- Implementation of the MADDPG algorithm (meaning Multi-Agent Deep Deterministic Policy Gradient, source here https://arxiv.org/pdf/1706.02275.pdf)
  - Model contains an two agents, each with an actor (deterministic approximator of optimal policy mu given the states s, i.e. returning four action values) and the critic (approximates the optimal action-value-function for the actor's action)
  - Both, actor and critic, are neural networks with two fully connected layers after the input (with 400 and 300 units) activated by a relu-function, and an output layer, whereas the size of the second layer for the critic is extended by 2 to additionally consider the actions from mu
  - The output layer of the actor is activated with a tanh to provide action values between -1 and 1
  - The output layer of the critic is activated with a relu to provide a positive action value

The MADDPG agent is described in more detail in Report.md.

## 	Instructions for installing dependencies or downloading needed files.
- Make sure to use Python v3.6 (I set up a separate environment in my Anaconda to run this)
- Install packages:
  - numpy
  - torch (include torchvision)
  - pandas (for the rolling average)
- Clone and extract this git repository
- Set up the environment:
  - Download the environment (I provided 20 agents) from one of the links below and place the file in the DRLND project repository. You need only select the environment that matches your operating system:
    - Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
    - Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
    - Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
    - Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

## 	How to run the code in the repository, to train the agent
- Follow instructions above to install dependencies and do required download
- Open the notebook "Tennis.ipynb"
- Run all cells in shown order
- Notebook include options to (re-)train the model, or to load the trained model and replay without further learning

## Results
Results are reported and documented in Report.md
