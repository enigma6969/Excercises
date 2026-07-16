## Final Lab submission for Deep Reinforcement Learning for Human Decision Strategies
### Professor Bernd Ludwig

#### Om Preetam, Valasa 


## Environments
The environment is available at [grid_world.py](Final_Lab\gymnasiun_env\gymnasium_envCatandMouse\envs\grid_world.py)

## Dynamics

- The default size of the grid is set to 6x6
- Both cat and mouse take 8-directional steps
- Terminal state is reached if 1. Cat reaches Mouse or 2. Mouse reaches Cheese
- Mouse is the learning agent
- Cat is always deterministic, takes 1 step towards the absolute direction of the Mouse
- Reward model can be viewed in the program grid_world.py

## Solutions 
- The REINFORCE model with N-parallel environments is implemented in [CatandMouse_Nparallel.py](gymnasiun_env\CatandMouse_Nparallel.py)
- The A2C redesigned model (removing the dependency on ptan) is implemented in [CatandMouse_a2c.py](gymnasiun_env/CatandMouse_a2c.py)
- The DQN model for the cat and mouse game is implemented in [CatandMouse_DQN.py](gymnasiun_env/CatandMouse_DQN.py)

## Installation

To install your new environment, run the following commands:

```{shell}
cd gymnasium_envCatandMouse
pip install gymnasium_envCatandMouse.
```
## How to run the code?
- First run the program as it is to set the initial weights.
- Once the training is complete you can visulize some episodes.
- For all the 3 programs reach out to if ```__name__ == '__main__':```
- There you can set ```train_mode = False```, to visualize some episodes





