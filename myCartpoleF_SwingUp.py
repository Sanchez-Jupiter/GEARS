"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class CartPoleSwingUp(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along 
        a frictionless track. The pendulum starts upright, and the goal is to 
        prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by 
        Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min                         Max
        0	Cart Position             -4.8                          4.8
        1	Cart Velocity             -Inf                          Inf
        2	Pole Angle                -24 deg (-0.418 rad)          24 deg (0.418 rad)
        3	Pole Velocity At Tip      -Inf                          Inf
        
    Actions:
        Type: Box(1)
        Num	Observation                 Min         Max
        0	Voltage to motor            -10          10     

        #so, input from -1 to 1, then multiplied by max_volt
        
        Note: The amount the velocity that is reduced or increased is not fixed; 
        it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 90 degrees #12
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.fps = 100                      # 控制环境更新的频率, Nikki changed from 50 to 100 
        self.gravity = 9.81
        self.masscart = 0.57 + 0.37         # mass of the cart
        self.masspole = 0.230               # mass of the pole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.3302                # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.r_mp = 6.35e-3                 # motor pinion radius(马达齿轮半径)
        self.Jm = 3.90e-7                   # rotor moment of inertia(马达转动惯量)
        self.Kg = 3.71                      # planetary gearbox gear ratio（马达与小车之间的转速比）
        self.Rm = 2.6                       # motor armature resistance(马达电阻)
        self.Kt = 0.00767                   # motor torque constant(马达的扭矩常数)
        self.Km = 0.00767                   # Back-ElectroMotive-Force (EMF) Constant V.s/RAD（反电动势常数）
                                            # both of these are in N.m.s/RAD, not degrees
        self.Beq = 5.4                      # equivalent viscous damping coecient as seen at the motor pinion（等效粘性阻尼系数）
        self.Bp = 0.0024                    # viscous damping doecient, as seen at the pendulum axis（杆子轴上的粘性阻力系数）

        self.force_mag = 10.0               # （小车上的最大作用力大小）should be 8 for our case?
        self.tau = 1 / self.fps             # seconds between state updates
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.tau)),
        }# 环境的元数据，包含了渲染模式和渲染的 FPS，方便在渲染时使用

        # self.kinematics_integrator = "semi-implicit-euler" # newly added from native CartPole
        self.kinematics_integrator = "RK4"
        # 运动学积分器，用于计算每个时间步小车和杆子的状态更新。
        # "RK4" 是一种常用的四阶龙格-库塔方法，它用于更精确地模拟物理系统的状态变化。
        # copied the following from mountain_car


        # Angle at which to fail the episode（当杆子的角度超过某个阈值时，任务将被视为失败并结束）
        # self.theta_threshold_radians = 12  * math.pi / 180 according to Dylan's thesis
        # approximatedly 12 deg
        self.theta_threshold_radians = 0.2 

        # self.x_threshold = 0.25 # lab result triggers watchdog
        self.x_threshold = 0.25
        # Angle limit set to 2 * theta_threshold_radians so failing observation（小车的最大水平偏移量）
        # is still within bounds

        # recall observation = sin theta, cos theta, theta_dot, x, x_dot !!!
        # recall state = x, x_dot, theta, theta_dot

        # 设置状态空间的最大值
        high = np.array(
                [
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max
                ],
                dtype = np.float32)
        # 设置动作动作空间
        # action is sampled uniformly from [-1,1]
        self.max_action = 1.0  
        # self.action_space = spaces.Box(low = -self.max_action, high = self.max_action)
        # self.observation_space = spaces.Box(low = -high, high = high)

        # 动作空间采用 spaces.Box，表示动作的值可以在一个连续范围内变化。
        self.action_space = spaces.Box(
                low = -self.max_action, 
                high = self.max_action, 
                shape = (1, ),          # 动作是一个标量，即一个一维的动作空间。通常这个值控制的是小车施加的推力或力矩。
                dtype = np.float32
                ) 
        # \in \mathbb{R}^1 bounded between -1 to 1
        # 观测空间也采用 spaces.Box，表示观测的状态是一个连续的实数空间
        self.observation_space = spaces.Box(-high, high, dtype = np.float32) 
        # in \mathbb{R}^5
        
        # from new native CartPole
        # 渲染设置
        self.render_mode = render_mode
        # self.render_mode：设置渲染模式。render_mode 变量可以是 "human" 或 "rgb_array"，用于控制渲染输出的方式。
        # "human" 模式通常用于显示给人类用户的图像，可能通过图形界面呈现。
        # "rgb_array" 模式返回的是一个包含 RGB 图像的数组，用于程序化地处理图像数据。
        self.screen_width = 600
        self.screen_height = 400
        # 设置屏幕的宽度和高度，用于渲染环境的显示，这里是 600 x 400 像素。
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        # 用于追踪在回合结束后，代理是否继续进行步骤。
        # 当环境的状态为 terminated = True 时，代理应该调用 reset() 来重置环境。
        # 如果继续调用 step()，应该根据这个变量来检测并处理。
    
    # 计算状态方程的右侧，也就是一个描述倒立摆（CartPole）系统动力学的函数。
    # 根据当前的状态（包括小车的位置、速度和杆子的角度、角速度）以及外部施加的力，返回系统的加速度（即状态的变化率）。
    # 函数的输出用于在仿真中更新系统状态。
    def RHS(self, y, force):

        assert self.state is not None, "Call reset before using step method."
        # 确保在调用该函数前，已经正确初始化了状态（即调用了 reset() 方法）
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2         
        y = self.state
        
        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        return np.array( (x_dot, xacc, theta_dot, thetaacc), dtype = np.float32).flatten()

    # 根据当前状态、施加的力以及所选的积分方法，计算出系统的新状态并返回。
    def stepPhysics(self, force):
        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2         

        
        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force


        # stepPhysics 函数使用不同的积分方法来计算状态更新。
        # 可以选择 euler、semi-euler 或 RK4（四阶龙格-库塔法）来进行数值积分，更新系统状态。
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

        if self.kinematics_integrator == "semi-euler":
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.kinematics_integrator == "RK4":
            y = self.state
            k1 = self.RHS(y, force = force)
            k2 = self.RHS(y + self.tau * k1 / 2, force = force)
            k3 = self.RHS(y + self.tau * k2 / 2, force = force)
            k4 = self.RHS(y + self.tau * k3, force = force)
            y = y + self.tau/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, x_dot, theta, theta_dot = y
        
        # self.state = (x, x_dot, theta, theta_dot)
        return np.array( (x, x_dot,theta, theta_dot), dtype = np.float32).flatten()
    
    # 类似于 stepPhysics，但主要用于模拟摆杆从垂直状态（倒立）摆动起来的过程。
    # 根据施加的外部力以及系统的当前状态，更新小车和摆杆的状态。
    # 虽然与 stepPhysics 非常相似，但有一些额外的步骤和调整。
    def stepSwingUp(self, force):
        # assert self.action_space.contains(
        #     0.1 * force
        # ), f"{force!r} ({type(force)}) invalid"
        # Currently same as stepPhysics
        # Need to modify??
        
        assert self.state is not None, "Call reset before using step method."
        # x, theta, x_dot, theta_dot = self.state
        # to be consistent with native CartPole state = x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = self.state
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        theta = math.atan(sintheta/costheta)
        # denominator used in a bunch of stuff
        d = 4 * self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + 4 * self.Jm * self.Kg**2         

        
        xacc = ((-4 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km)) / (self.Rm *(d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * self.Bp * self.r_mp**2 * costheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-4 * self.masspole * self.length * self.r_mp**2 * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * self.masspole * self.gravity * self.r_mp**2 * costheta * sintheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) + (4 * self.r_mp * self.Kg * self.Kt) / (self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        thetaacc = ((-3 * (self.Rm * self.r_mp**2 * self.Beq + self.Kg**2 * self.Kt * self.Km) * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * x_dot + ((-3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp) / (self.masspole * self.length**2 * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) * theta_dot + ((-3 * self.masspole * self.r_mp**2 * sintheta * costheta) / (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * theta_dot**2 + ((3 * (self.masscart * self.r_mp**2 + self.masspole * self.r_mp**2 + self.Jm * self.Kg**2) * self.gravity * sintheta) / (self.length * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2))) + (3 * self.r_mp * self.Kg * self.Kt * costheta) / (self.length * self.Rm * (d + 3 * self.r_mp**2 * self.masspole * sintheta**2)) * force

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        
        if self.kinematics_integrator == "semi-euler":
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.kinematics_integrator == "RK4":
            # RK4
            y = self.state
            k1 = self.RHS(y, force = force)
            k2 = self.RHS(y + self.tau * k1 / 2, force = force)
            k3 = self.RHS(y + self.tau * k2 / 2, force = force)
            k4 = self.RHS(y + self.tau * k3, force = force)
            y = y + self.tau/6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, x_dot, theta, theta_dot = y
        
        # self.state = (x, x_dot, theta, theta_dot)
        return np.array( (x, x_dot,theta, theta_dot), dtype = np.float32).flatten()

    def reward(self):
        
        x, x_dot, theta, theta_dot = self.state
        energy_total = 2/3 * self.masspole * self.length ** 2 * theta_dot + self.masspole * self.length * self.gravity * (math.cos(theta) - 1); 
        lyapunov_2 = 1/2 * ( energy_total )**2 + 1e-4 * (1 - (math.cos(theta))**3 ); 

        return energy_total
        #(GEARS (?))
        
    # 用于执行一次环境的状态更新，并返回新的观察、奖励、终止标志和一些额外信息
    # !!this block may contain bugs!!
    def step(self, action):
        # Cast action to float to strip np trappings
        # 输入的 action（可以是一个连续值）会被裁剪到合适的范围内。
        # action 被乘以 max_action 来调整其幅度，并确保它不超过 max_action 的范围。
        action = np.clip( self.max_action * action, -self.max_action, self.max_action)
        
        
        # 通过 force_mag 和 action 来计算应用到小车上的力 force，这个力的大小决定了小车的加速度。
        # force = float(np.clip( self.max_action * action, -self.max_action, self.max_action))
        # Nikki modified the following force from continuous_mountain_car
        # force = min(max(action[0], -self.max_action), self.max_action) * self.max_action
        force = self.force_mag * float(action)
        
        # 根据当前的力计算更新后的系统状态
        self.state = self.stepSwingUp(force)
        
        # 状态信息提取
        x, x_dot, theta, theta_dot = self.state

        # 1 if pole stands up - can turn off
        # 1 if cart is off track - can turn off and reset
        terminated = bool(abs(theta) < self.theta_threshold_radians / 4) 
        off_track = bool(abs(x) > self.x_threshold) 
        # Original reward function
        '''
        if not terminated:
            if off_track:
                reward = -100000.0 # (GEARS (?))
            else:
 #               reward = reward(self.state) #(GEARS (?))
                    reward = 100
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
#            reward = reward(self.state)
            reward = 100
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = +1000.0
        '''
        reward = 0
        offset = abs(self.state[0])  # 横向偏移
        angle = self.state[2]        # 杆子的角度（假设是与垂直方向的偏离角度）
        angle_velocity = self.state[3]  # 杆子的角速度

        max_distance = 0.01             # 设置最大奖励距离
        punishment_threshold = 0.05     # 设置惩罚开始的距离
        max_reward = 100                # 最大奖励值
        balance_threshold = 0.02        # 设置杆子接近平衡状态的角度范围
        balance_reward = 5000             # 接近平衡时的奖励

        if not terminated:
            if off_track:
                reward = -10000.0  # 超出轨道的惩罚
            else:
                # 添加角度奖励，杆子越接近垂直，奖励越高
                if abs(angle) <= balance_threshold:
                    reward += balance_reward  # 当角度接近垂直时，增加奖励
                reward += 100 * (3.14 / 2 - angle) 
                reward += 10 * self.reward()  # 其他奖励（例如，系统提供的额外奖励）
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 100  # 如果杆子倒下，给予较高的初始奖励（根据需求可以调整）
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 1000.0  # 额外奖励（可以根据需求调整）

        




        # Nikki copied the following reward from continuous_mountain_car
        # reward = 0.0
        # if terminated:
        #     reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1
        # Nikki changed from above to below according  to:
        # Matlab Train DDPG to swing up and balance pole 

        # 返回的观察是一个数组，包含了小车的位置、速度，摆杆的角度、角速度（通过 cos(theta) 和 sin(theta) 
        # 表示角度的余弦和正弦部分），这构成了当前环境的状态信息。
        obs = np.array( (x, x_dot, np.cos(theta), np.sin(theta), theta_dot), dtype = np.float32).flatten()
        
        # 调用 render() 函数来显示环境的可视化。
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, off_track, {}

    # This def is copied from native cartpole - !!may contain bugs!!
    # 重置环境的状态，并准备好进行新的 episode。
    # 重新初始化状态、设置随机种子，并返回一个初始的观察。
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        super().reset(seed = seed)
        # 可选的随机种子，确保实验的可重复性。
        # 传入该参数时，可以在生成随机数时使用相同的种子，从而使得每次训练或测试时生成的随机数一致。
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state / observations.

        # 通过 utils.maybe_parse_reset_bounds 函数解析 options 参数中的初始状态范围。
        # low 和 high 是一个 4 元组，分别定义了环境中各个状态变量的初始范围。
        # 默认情况下，x 和 x_dot（小车的位置和速度）将在 -0.08 到 0.08 之间初始化，
        # 而 theta 和 theta_dot（摆杆的角度和角速度）将根据提供的 options 进一步确定其范围。
        low, high = utils.maybe_parse_reset_bounds(
            # options, -0.05, 0.05  
            # default low
            options, -0.08, 0.08 # NX changed from above
        )  # default high

        # 通过 np_random.uniform(low = low, high = high, size = (4, )) 从指定的范围内生成 4 个随机值，
        # 分别对应状态空间中的 x、x_dot、theta 和 theta_dot。
        # - [0, 0, math.pi, 0] 是平移操作，将 theta 初始化为 pi，即摆杆开始时垂直于小车。
        # 最终生成的状态值存储在 self.state 中。
        # 状态是一个包含 4 个元素的数组，分别是 x、x_dot、theta 和 theta_dot。
        self.state = np.array( self.np_random.uniform(low = low, high = high, size = (4, )) - [0, 0, math.pi, 0] ).flatten()
        # 在新的回合开始时，steps_beyond_terminated 被设为 None，用于追踪在终止状态后的额外步骤。
        self.steps_beyond_terminated = None
        
        x, x_dot, theta, theta_dot = self.state

        obs = np.array( (np.sin(theta), np.cos(theta), theta_dot, x, x_dot), dtype = np.float32).flatten() 
        # I think (?) might need rotation
        # to go from observation (obs) angles to state space angle, need the following transformation 
        # (rotate the coordinate by pi / 2):
        # state_angle = math.atan2(obs_cosangle, -obs_sinangle) - pi / 2
        
        if self.render_mode == "human":
            self.render()

        return np.array(obs, dtype=np.float32), {}

    
   

    ## this render def is copied from CartPole and never changed...
    # 环境可视化显示
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        # 这里检查 pygame 是否安装，
        # pygame 是一个常用于游戏开发的图形库，用于实现环境的渲染。
        # 如果没有安装，抛出异常并提示用户安装 pygame。

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        # 如果 screen 为空（即还没有初始化），则会根据 render_mode 进行初始化：
        # 在 human 模式下，创建一个显示窗口 (pygame.display.set_mode)，
        # 并设置窗口尺寸为 screen_width 和 screen_height。
        # 在 rgb_array 模式下，创建一个 pygame.Surface 对象，用于存储环境的图像数据。

        if self.clock is None:
            self.clock = pygame.time.Clock()
        # 如果 clock 为空，初始化一个 pygame.time.Clock() 对象，用于控制帧率。

        world_width = self.x_threshold * 2
        # world_width 是小车可以移动的最大范围
        scale = self.screen_width / world_width
        # 然后使用屏幕宽度和世界宽度计算缩放比例 scale，用于将环境中的物体适配到屏幕上。
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0

        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100                                     # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]  # cart_coords 存储了小车的四个角的坐标。
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        # 绘制小车的轮廓和填充颜色。
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # 根据摆杆的长度、宽度、角度 (x[2] 为摆杆的角度) 计算摆杆的位置。
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        # 用 gfxdraw 来绘制摆杆。
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        # 渲染到屏幕
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    # 关闭图形显示窗口并清理资源。
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    



'''
# below is copied without modification from native cartpole

class myCartPoleFVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        num_envs: int = 2,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.steps = np.zeros(num_envs, dtype=np.int32)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.clocks = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        done = terminated | truncated

        if any(done):
            # This code was generated by copilot, need to check if it works
            self.state[:, done] = self.np_random.uniform(
                low=self.low, high=self.high, size=(4, done.sum())
            ).astype(np.float32)
            self.steps[done] = 0

        reward = np.ones_like(terminated, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state.T, reward, terminated, truncated, {}
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.low, self.high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        ).astype(np.float32)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return self.state.T, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screens is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screens = [
                    pygame.display.set_mode((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
            else:  # mode == "rgb_array"
                self.screens = [
                    pygame.Surface((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
        if self.clocks is None:
            self.clock = [pygame.time.Clock() for _ in range(self.num_envs)]

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        for state, screen, clock in zip(self.state, self.screens, self.clocks):
            x = self.state.T

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100  # TOP OF CART
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            [clock.tick(self.metadata["render_fps"]) for clock in self.clocks]
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return [
                np.transpose(
                    np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
                )
                for screen in self.screens
            ]

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
'''
