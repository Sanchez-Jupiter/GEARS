### The NCSU GEARS research on Cartpole Problem 

#### Background
The CartPole problem is a classic control problem in reinforcement learning and robotics. It involves swinging up and balancing a pole attached to a cart that moves along a one-dimensional track as is shown in figure. The goal is to keep the pole upright by applying forces to the cart. The system is inherently unstable, meaning that if no corrective action is taken, the pole will eventually fall over. The task is to design a controller that can swing the pole up and keep it balanced for as long as possible. 

![cartpole](https://github.com/Sanchez-Jupiter/GEARS/blob/master/Cart-Pole%20Model%20Example2.png)
#### Main Process
We chose the Deep Deterministic Policy Gradient (DDPG) algorithm to train the model in which we focus on designing the reward function and the structure of the policy network. After achieving a relatively good result on simulation environment, we tried to apply the model on the real lab where we modified the previous curcuit of Linear Quadratic Regulator(LQR) and added the trained model to it as is shown in the figure.

![curcuit](https://github.com/Sanchez-Jupiter/GEARS/blob/master/curcuit.PNG)

In terms of reward function, we decided to encourage the progress and punish the retrogress as is shown below. 

$$

reward =
\begin{cases}
b_1 * isClose_{pre}  - b_2 * \cos(\theta) * \left| \dot{\theta} \right| - b_3 * (x - x_{pre})& \text{if  } \theta \geq threshold \\
b_4 * (\cos(\theta) - \cos(\theta_{pre})) - isClose_{pre} * total_{pre} & \text{if } \theta < threshold \\
b_5 & \text{if }  \text{ } \text{ } \text{ } offtrack \\
\end{cases}

$$
where $ b_i $ ($ 1 \leq i \leq 5 $) is the constant and is used to adjust the influence of each factor,  'pre'means previous step, $ isClose $ is a bool which indicates whether the pole has reached the ideal range, $ \theta $ is the angle of the pole, $ x $ is the position of the cart, $ total $ means the sum of the reward gained when the pole is in the ideal range

#### Result and Current Challenge
Our model can swing up and keep balance in the simulation environment, the performance is shown in github, you can scan the QR code to see it.

![performance](https://github.com/Sanchez-Jupiter/GEARS/blob/master/great%20balance.gif)

We implement our model to the real lab, but the wheel broke at a certain point while running. After observation and analysis, we think it is due to the sudden change of force, and we are still working on this by trying to modify our reward function to punish the sudden change of force.
