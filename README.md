\# Deep Q-Network (DQN) on CartPole-v1



\## Project Overview



This project implements and evaluates a Deep Q-Network (DQN) agent on the Gymnasium `CartPole-v1` environment. It is an educational reproduction inspired by the core DQN methods introduced by Mnih et al. (2015).



The project investigates how learning rate and target-network updates affect learning performance. It also examines the contribution of two important DQN components: the replay buffer and the target network.



\*\*Important scope note:\*\* This is a simplified DQN implementation using a low-dimensional CartPole state and a multilayer perceptron. It reproduces selected DQN principles, not the original Atari-based architecture and experimental setup from the paper.



\## Reference Paper



Mnih, V. et al. (2015). \*Human-level control through deep reinforcement learning.\* Nature, 518, 529–533.



https://doi.org/10.1038/nature14236



\## Environment and Agent



\- \*\*Environment:\*\* Gymnasium CartPole-v1

\- \*\*State space:\*\* 4 continuous observations

\- \*\*Action space:\*\* 2 discrete actions

\- \*\*Maximum episode reward:\*\* 500

\- \*\*Network:\*\* Fully connected neural network

\- \*\*Hidden layers:\*\* 24 and 24 neurons, with ReLU activation

\- \*\*Output:\*\* 2 Q-values, one for each action



\### DQN components



The implementation includes:



1\. An online Q-network for estimating action values.

2\. A target network for stabilizing target-value estimation.

3\. Experience replay for sampling past transitions.

4\. Epsilon-greedy exploration.

5\. The Bellman target for Q-learning.

6\. Adam optimizer and mean squared error loss.

7\. Gradient clipping.



\## Experimental Configuration



| Parameter | Configuration |

|---|---:|

| Environment | CartPole-v1 |

| Training steps | 100,000 |

| Replay buffer capacity | 50,000 |

| Batch size | 64 |

| Discount factor (gamma) | 0.99 |

| Baseline learning rate | 0.0005 |

| Baseline target update interval | 1,000 |

| Hidden layers | \[24, 24] |

| Initial epsilon | 1.0 |

| Final epsilon | 0.1 |

| Epsilon decay steps | 20,000 |

| Gradient clipping | 10 |

| Training seeds | 42, 123, 2025, 7, 21 |



The same set of five random seeds was used for the comparisons to make the experiments more consistent.



\## Experimental Results



\### 1. Evaluation comparison



Each trained model was evaluated over 100 episodes using a greedy policy. The table reports the mean and sample standard deviation of the five seed-level evaluation means.



| Experiment | Mean reward | Std. dev. | Median | Seeds reaching 500 |

|---|---:|---:|---:|---:|

| Baseline (LR 0.0005) | 184.32 | 176.55 | 109.03 | 1/5 |

| LR 0.0001 | 224.04 | 50.30 | 219.26 | 0/5 |

| LR 0.00025 | 289.49 | 194.24 | 189.00 | 2/5 |

| LR 0.00075 | 258.65 | 217.37 | 105.25 | 1/5 |

| LR 0.001 | \*\*428.50\*\* | 159.87 | \*\*500.00\*\* | \*\*4/5\*\* |

| Target update 500 | 255.91 | 222.85 | 98.13 | 2/5 |

| Without target network | 226.13 | 169.24 | 140.20 | 0/5 |

| Without replay buffer | 78.15 | 70.22 | 85.15 | 0/5 |



\### 2. Learning-rate sensitivity



Three alternative learning rates were compared with the baseline learning rate:



\- 0.0001

\- 0.00025

\- 0.00075

\- 0.001



The learning rate of \*\*0.001 achieved the highest mean evaluation reward (428.50)\*\* and four of the five seeds reached the maximum episode reward of 500.



However, its standard deviation was still relatively high. This indicates that the result is not equally consistent across all random seeds.



The learning rate of 0.0001 produced a lower mean reward but considerably lower variation across seeds.



\### 3. Target-network comparison



The target-network update interval was changed from the baseline value of 1,000 to 500.



The shorter update interval did not produce a consistent improvement. Removing the target network also produced mixed results, suggesting that training stability depends on the interaction between network updates and other hyperparameters.



\### 4. Ablation comparison



Two components were examined:



\- \*\*Without target network:\*\* Investigates the effect of removing the separate target network.

\- \*\*Without replay buffer:\*\* Investigates the effect of removing experience replay.



The no-replay configuration achieved the lowest mean evaluation reward (78.15). This is consistent with the importance of replay-based learning in the tested setup.



The ablation results should be interpreted in the context of the implementation and its configuration; they do not establish universal conclusions about every DQN environment.



\## Figures and Visual Results



\### Baseline learning curves



!\[Baseline learning curves](results/report\_figures/01\_baseline\_learning\_curves.png)



\### Learning-rate sensitivity



!\[Learning-rate comparison](results/report\_figures/02\_learning\_rate\_training\_curves.png)



\### Target-network update comparison



!\[Target update comparison](results/report\_figures/03\_target\_update\_comparison.png)



\### Ablation comparison



!\[Ablation comparison](results/report\_figures/04\_ablation\_comparison.png)



\### Evaluation comparison



!\[Evaluation comparison](results/report\_figures/05\_evaluation\_comparison.png)



\## Trained Agent Animation



The following animation shows a trained DQN agent controlling the CartPole environment.



The demonstration uses the model trained with learning rate 0.001 and training seed 42. In the displayed evaluation episode, the agent achieved the maximum reward of 500 over 500 steps.



!\[Trained DQN CartPole demonstration](results/report\_figures/cartpole\_trained\_dqn\_seed42.gif)



This is one successful demonstration episode, not a substitute for the multi-seed evaluation.



\## Model Quality and Interpretation



The experiments demonstrate that the implemented DQN agent can learn an effective CartPole control policy.



The strongest evaluation result was obtained with a learning rate of 0.001:



\- Mean reward: 428.50

\- Median reward: 500

\- Four out of five seeds reached 500 in their evaluation mean.



Nevertheless, the standard deviation across seeds remains substantial. This indicates sensitivity to random initialization and training variation.



The results therefore show promising performance, but do not demonstrate uniformly reliable performance across all experimental conditions.



\## Limitations



1\. CartPole is a relatively simple control environment compared with the Atari tasks used in the original DQN paper.

2\. The network is a small multilayer perceptron rather than the original convolutional architecture.

3\. Only five training seeds were used per configuration.

4\. Hyperparameter comparisons cover a limited range of settings.

5\. Results from this environment should not be generalized directly to more complex control problems.



\## Running the CartPole Animation



From the project root, activate the virtual environment and run:



```powershell

.\\.venv\\Scripts\\Activate.ps1

python -m src.animate\_cartpole

```



The animation script loads the trained checkpoint and generates the GIF in:



`results/report\_figures/cartpole\_trained\_dqn\_seed42.gif`



\## Conclusion



This project implements the central components of DQN and evaluates their effects on CartPole-v1 through baseline experiments, learning-rate sensitivity analysis, target-network comparison, and ablation studies.



The experiments identify a learning rate of 0.001 as the strongest configuration by mean evaluation reward among the tested settings. They also highlight the importance of examining variability across random seeds rather than relying on a single successful training run.



Overall, the project provides a practical implementation and experimental study of DQN-based reinforcement learning, including both successful outcomes and limitations.



\---



\*\*Author:\*\* Kazi Rahat Hossain  

\*\*Project:\*\* Deep Q-Network implementation and experimental evaluation  

\*\*Frameworks:\*\* Python, PyTorch, Gymnasium, NumPy, Matplotlib





















optional

\# Deep Q-Network (DQN) Reproduction on CartPole-v1



\## 1. Project Overview



This project implements and evaluates a Deep Q-Network (DQN) reinforcement learning agent on the Gymnasium CartPole-v1 environment.



The implementation is inspired by the paper:



\*\*Human-level control through deep reinforcement learning\*\*  

Volodymyr Mnih et al., \*Nature\*, 2015.  

https://doi.org/10.1038/nature14236



The main objectives are to:



\- Implement the core DQN algorithm.

\- Train an agent to balance the CartPole.

\- Investigate the effect of different learning rates.

\- Examine the effect of target network updates.

\- Perform ablation experiments on key DQN components.

\- Evaluate performance across multiple random seeds.



\*\*Important scope clarification:\*\* This is a simplified DQN implementation using a small multilayer perceptron on CartPole-v1. It is inspired by the 2015 paper but is not an exact reproduction of the original Atari-based DQN architecture and training pipeline.



\## 2. Environment and Implementation



| Component | Configuration |

|---|---|

| Environment | Gymnasium CartPole-v1 |

| State dimension | 4 |

| Action dimension | 2 |

| Neural network | Fully connected MLP |

| Hidden layers | 24, 24 neurons |

| Activation | ReLU |

| Optimizer | Adam |

| Loss function | Mean Squared Error |

| Discount factor | 0.99 |

| Replay buffer capacity | 50,000 |

| Batch size | 64 |

| Gradient clipping | 10 |

| Training steps | 100,000 |

| Evaluation | 100 episodes per trained model |



\### Neural Network Architecture



The Q-network maps the four-dimensional CartPole state to two action-value estimates.



\- Input layer: 4 neurons

\- Hidden layer 1: 24 neurons, ReLU

\- Hidden layer 2: 24 neurons, ReLU

\- Output layer: 2 neurons



The two output values represent the estimated Q-values for the available actions.



\### Core DQN Components



The implementation includes:



1\. \*\*Online network:\*\* Estimates action values and is updated during training.

2\. \*\*Target network:\*\* Provides more stable target Q-value estimates.

3\. \*\*Experience replay:\*\* Stores previous transitions and samples mini-batches.

4\. \*\*Epsilon-greedy exploration:\*\* Balances exploration and exploitation.

5\. \*\*Bellman target:\*\* Uses the reward and discounted next-state value to calculate the learning target.



The exploration rate decreases from 1.0 to 0.1 over 20,000 steps.



\## 3. Experimental Design



The experiments use five random seeds:



`\[42, 123, 2025, 7, 21]`



Each configuration is trained independently using these seeds.



The trained models are evaluated over 100 greedy evaluation episodes. The evaluation seeds are 1000–1099.



The primary performance metric is the mean episode reward. The standard deviation reported below is calculated across the five seed-level evaluation means, not across all individual evaluation episodes.



\### 3.1 Baseline Configuration



| Parameter | Value |

|---|---:|

| Learning rate | 0.0005 |

| Target update interval | 1000 |

| Replay buffer | Enabled |

| Batch size | 64 |

| Random seeds | 5 |



\### 3.2 Learning Rate Sensitivity



The following learning rates were investigated:



\- 0.0001

\- 0.00025

\- 0.0005 (baseline)

\- 0.00075

\- 0.001



\### 3.3 Target Network Update Experiment



The target network update interval was changed from 1000 to 500 steps.



\### 3.4 Ablation Experiments



Two ablation configurations were investigated:



\- \*\*Without target network:\*\* Examines the effect of removing the separate target network.

\- \*\*Without replay buffer:\*\* Examines learning without the standard experience replay mechanism.



These ablations help investigate the contribution of important DQN components. They should be interpreted in the context of the specific implementation and experimental settings.



\## 4. Experimental Results



\### 4.1 Overall Evaluation Results



| Experiment | Mean reward | Seed-level SD | Median reward | Seeds reaching 500 |

|---|---:|---:|---:|---:|

| Baseline (LR 0.0005) | 184.32 | 176.55 | 109.03 | 1/5 |

| LR 0.0001 | 224.04 | 50.30 | 219.26 | 0/5 |

| LR 0.00025 | 289.49 | 194.24 | 189.00 | 2/5 |

| LR 0.00075 | 258.65 | 217.37 | 105.25 | 1/5 |

| LR 0.001 | \*\*428.50\*\* | 159.87 | \*\*500.00\*\* | \*\*4/5\*\* |

| Target update 500 | 255.91 | 222.85 | 98.13 | 2/5 |

| Without target network | 226.13 | 169.24 | 140.20 | 0/5 |

| Without replay buffer | 78.15 | 70.22 | 85.15 | 0/5 |



The maximum CartPole-v1 episode reward is 500.



\### 4.2 Learning Rate Analysis



The learning rate of 0.001 achieved the highest mean evaluation reward of 428.50, with four out of five seeds reaching the maximum reward of 500.



However, its seed-level standard deviation was 159.87, indicating that performance was not identical across all runs.



The learning rate of 0.0001 produced a lower mean reward of 224.04 but showed comparatively lower variability.



These findings indicate that learning rate has a substantial influence on training performance and consistency.



\### 4.3 Target Network Analysis



Changing the target update interval from 1000 to 500 steps did not consistently improve performance.



The target update experiment achieved a mean reward of 255.91, compared with 184.32 for the baseline. However, the large standard deviation indicates considerable variation between seeds.



\### 4.4 Ablation Analysis



Removing the replay buffer resulted in the lowest mean reward (78.15) among the evaluated configurations.



The configuration without a target network achieved a mean reward of 226.13, but its results also varied substantially between seeds.



These results suggest that replay memory was particularly important for this implementation. Nevertheless, the ablation results should not be interpreted as universal conclusions about DQN because implementation details and training dynamics can affect the outcome.



\## 5. Figures and Visualization



\### Baseline Learning Curves



!\[Baseline learning curves](results/report\_figures/01\_baseline\_learning\_curves.png)



\### Learning Rate Comparison



!\[Learning rate comparison](results/report\_figures/02\_learning\_rate\_training\_curves.png)



\### Target Network Update Comparison



!\[Target update comparison](results/report\_figures/03\_target\_update\_comparison.png)



\### Ablation Comparison



!\[Ablation comparison](results/report\_figures/04\_ablation\_comparison.png)



\### Evaluation Comparison



!\[Evaluation comparison](results/report\_figures/05\_evaluation\_comparison.png)



\## 6. CartPole Agent Demonstration



The following animation shows a trained DQN agent interacting with the CartPole environment.



\*\*Model used:\*\* Learning rate 0.001, training seed 42.



\*\*Demonstration result:\*\* 500 steps, corresponding to the maximum episode reward.



!\[Trained DQN CartPole demonstration](results/report\_figures/cartpole\_trained\_dqn\_seed42.gif)



This animation demonstrates that the trained agent can successfully balance the pole for the full 500-step episode. It is an illustrative demonstration, not by itself proof of generalization across all initial conditions.



\## 7. Model Quality and Discussion



The experiments demonstrate that the implemented DQN agent can learn the CartPole control task.



The strongest observed configuration was the learning rate of 0.001, which achieved a mean evaluation reward of 428.50 across five seeds.



Several important observations emerged:



\- Learning rate strongly influenced performance.

\- A lower learning rate produced more consistent results in this experiment.

\- A learning rate of 0.001 achieved the highest average reward.

\- Changing the target update interval did not guarantee better performance.

\- Removing the replay buffer substantially reduced the observed performance.

\- Different random seeds produced noticeably different outcomes.



Although the best configuration performed well, the high variability in several experiments indicates that further improvements in training stability would be valuable.



\## 8. Limitations



This project has several limitations:



1\. The experiment uses CartPole-v1, a relatively simple control environment.

2\. The neural network is a small MLP rather than the convolutional architecture used in the original Atari DQN paper.

3\. Only five random seeds were evaluated per configuration.

4\. The experiments investigate a limited range of hyperparameters.

5\. The findings should not be generalized directly to more complex reinforcement learning tasks.



Therefore, this work should be considered a simplified educational reproduction and experimental investigation of core DQN principles.



\## 9. Project Structure



```text

DQN\_Mnih2015\_Reproduction/

│

├── src/

│   ├── network.py

│   ├── environment.py

│   ├── animate\_cartpole.py

│   └── generate\_report\_figures.py

│

├── results/

│   ├── sensitivity/

│   ├── ablation/

│   └── report\_figures/

│       ├── 01\_baseline\_learning\_curves.png

│       ├── 02\_learning\_rate\_training\_curves.png

│       ├── 03\_target\_update\_comparison.png

│       ├── 04\_ablation\_comparison.png

│       ├── 05\_evaluation\_comparison.png

│       ├── evaluation\_comparison\_summary.csv

│       └── cartpole\_trained\_dqn\_seed42.gif

│

├── requirements.txt

└── README.md

```



The structure above highlights the main project components. Additional training, evaluation, and experiment scripts are also included in the repository.



\## 10. How to Run



\### Step 1: Clone the repository



```bash

git clone https://github.com/Rahat4790/DQN-Mnih2015-Reproduction.git

cd DQN-Mnih2015-Reproduction

```



\### Step 2: Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### Step 3: Install dependencies



```powershell

pip install -r requirements.txt

```



\### Step 4: Run the CartPole animation



From the project root:



```powershell

python -m src.animate\_cartpole

```



The animation script uses the trained checkpoint configured in the script. Ensure the corresponding checkpoint is available under the results directory.



\## 11. Conclusion



This project implemented a simplified Deep Q-Network agent and evaluated its behavior on CartPole-v1.



The sensitivity and ablation experiments provided practical insight into the effects of learning rate, target network updates, and experience replay.



The best observed learning-rate configuration achieved a mean evaluation reward of 428.50, while the animation demonstrated a successful 500-step episode.



Overall, the project provides an experimental demonstration of the core DQN learning mechanism, its important components, and the challenges of obtaining consistent reinforcement learning performance across random seeds.

