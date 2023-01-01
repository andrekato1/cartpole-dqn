# 🎢 Cartpole DQN

This repository contains a Deep Q-Network (DQN) trained to solve the Cartpole environment using reinforcement learning. The network was implemented using PyTorch.

# 🚀 Getting Started

* Clone the repository: `git clone https://github.com/andrekato1/cartpole-dqn.git`
* Install dependencies: `pip install -r requirements.txt`
* Train the DQN: `python main.py`
* Test the trained model: `python test.py`

# 📝 Note

The main.py script will save two trained model weights to a file called `best_model.pt` and `latest_model.pt`. The names of the files are self-explanatory. Since DQN is by nature very sensitive to hyperparameter tuning, we've observed frequently that training randomly collapsed at some point. That's why we always keep track of the best model we got so far.

The test.py script will load the trained model from this file and run it on the Cartpole environment.

# 💡 Tips

* Adjust the hyperparameters to your liking inside the `main.py` file.
* Adjust the model to use inside the `test.py` file. By default, we always use the best model.

# TODOs

* Set a seed on `random` and `torch`.
* Implement a way to monitor training. Gym's Monitor wrapper has been deprecated in the version we're using.