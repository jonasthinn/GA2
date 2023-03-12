# Graded Assignment 2

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take. The task was to translate the code from TensorFlow to PyTorch
***
Sample games from the best performing [agent](../models/v15.1/model_188000.h5)<br>
<img width="400" height="400" src="https://github.com/jonasthinn/GA2/blob/main/images2/game_visual_v17.1_199000_14_ob_0.gif" alt="model v17.1 agent" ><
***

## Code Structure
Ive mainly made changes into [agent.py](../agent.py), and some minor changes in [training.py](../training.py)"
and [game_visualization.py](../game_visualization.py).
The network im using in DeepQLearningAgentTorch-class, is created in a seperate file;[cnn.py](../cnn.py), for convinience.
I train my agent by running [training.py], where I specify which agent are to be trained. Logs from training are saved 
in (../model_logs/v17.1.csv), and my models are saved in ('model_torch/v17.1.csv')
By running [game_visualization.py](../game_visualization.py) with current setup, it saves the visualization in image2, 
to keep results seperated from the original files.

[agent.py](../agent.py) contains the agent for playing the game. It implements and trains a convolutional neural network for the action values. Following classes are available
<table>
This are the classes Ive made in agent.py :
    <head>
        <tr>
        <th> Class </th><th> Description</th>
        </tr>
    </head>
    <tr><td>DeepQLearningAgentTorch</td><td>Pytorch Deep Q Learning Algorithm with CNN Network</td></tr>
    <tr><td>PolicyGradientAgentTorch</td><td>Pytorch Policy Gradient Algorithm with CNN Network</td></tr>
    <tr><td>AdvantageActorCriticAgentTorch</td><td>Pytorch Advantage Actor Critic (A2C) Algorithm with CNN Network</td></tr>

</table>
