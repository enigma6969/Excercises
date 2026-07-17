## Final Lab submission for Deep Reinforcement Learning for Human Decision Strategies
### Professor Bernd Ludwig

#### Om Preetam, Valasa 

## Submission Details
All the Python files with solutions are present in the folder Final_Lab
Please download the folder and access the Solution files
- All the programs are available [here!](Final_Lab/gymnasiun_env)

## Environments
The environment is available at [grid_world.py](Final_Lab/gymnasiun_env/gymnasium_envCatandMouse/envs/grid_world.py)

## Dynamics

- The default size of the grid is set to 6x6
- Both the cat and the mouse take 8-directional steps
- Terminal state is reached if 1. Cat reaches Mouse or 2. Mouse reaches Cheese
- Mouse is the learning agent
- Cat is always deterministic, takes 1 step towards the absolute direction of the Mouse
- The reward model can be viewed in the program grid_world.py

## Solutions 
- The REINFORCE model with N-parallel environments is implemented in [CatandMouse_Nparallel.py](Final_Lab/gymnasiun_env/CatandMouse_Nparallel.py)
- The A2C redesigned model (removing the dependency on ptan) is implemented in [CatandMouse_a2c.py](Final_Lab/gymnasiun_env/CatandMouse_a2c.py)
- The DQN model for the cat and mouse game is implemented in [CatandMouse_DQN.py](Final_Lab/gymnasiun_env/CatandMouse_DQN.py)

## Installation

Please find all the library details in the requirements.txt

To install your new environment, run the following commands:

```{shell}
cd gymnasium_envCatandMouse
pip install gymnasium_envCatandMouse
```
## How to run the code?
- First, run the program as it is to set the initial weights.
- Once the training is complete, you can visualize some episodes.
- For all the 3 programs reach out to if ```__name__ == '__main__':```
- There you can set ```train_mode = False```, to visualize some episodes





