# DQN Agent for Cartpole-v1
Simple Reinforced Learning Model trained to perfection using DQN. The model has been tuned to follow the best policy in the cartpole environment yielding perfect scores everytime. In addition to hours of training and hyperparameter tuning this result is achieved due to the deterministic nature of the cartpole environment and box2d physics.

# How to Use
**Visualization and Evaluation** - *cartpole_eval.py*

Run this file to load a pre-trained agent and visualize it in the cartpole environment. A list of all the previously trained model can be found in Models folder and the plots are saved in the Plots folder.

**Training** - *cartpole.py*

Use this file to train the model even further. All the hyperparameters can be found at the top. Increment the gen variable to avoid overwriting on previous model and plot files.

# Dependencies
* Python 3.11.11
*  Pytorch 2.5.1
*  Gymnasium 0.28.1
*  swig 4.3.0
*  OpenCV 4.11.0.86
*  box2d-py 2.3.5
*  NumPy
*  MatPlotLib

# References

* https://arxiv.org/abs/1312.5602v1
* https://gymnasium.farama.org/environments/classic_control/cart_pole/
