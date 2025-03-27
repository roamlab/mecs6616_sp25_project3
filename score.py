import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
from render import Renderer
import argparse
import time
import torch
import math
import tqdm
import ray
from math import pi
import os
np.set_printoptions(suppress=True)


# part 2 scoring
def reset(arm_teacher, arm_student, torque):
    initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    arm_teacher.set_state(initial_state)
    arm_student.set_state(initial_state)

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)

    arm_teacher.set_t(0)
    arm_student.set_t(0)


def set_torque0(arm_teacher, arm_student, torque):
    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)


def score_random_torque(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques = np.random.uniform(-1.0, 1.0, num_tests) # changed range from [-1.5, 1.5] to [-1.0, 1.0]
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque = {torque} Nm)\n')
        reset(arm_teacher, arm_student, torque)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.008:
          score = 0.5
        if mse < 0.0005:
          score = 1
        if mse >= 0.008:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))


def score_linear_torques(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques = np.random.uniform(0.5, 1.0, num_tests) # changed 1.5 to 1.0
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque 0 -> {torque} Nm)\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            set_torque0(arm_teacher, arm_student, arm_teacher.get_t() / time_limit * torque)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.008:
          score = 0.5
        if mse < 0.0005:
          score = 1
        if mse >= 0.008:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))


def score_two_torques(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques1 = np.random.uniform(-1, 1, num_tests)
    torques2 = np.random.uniform(-1, 1, num_tests)
    for i, (torque1, torque2) in enumerate(zip(torques1, torques2)):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque 1 = {torque1} Nm,  Torque 2 = {torque2} Nm)\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            if arm_teacher.get_t() < time_limit / 2:
                set_torque0(arm_teacher, arm_student, torque1)
            else:
                set_torque0(arm_teacher, arm_student, torque2)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.05:
          score = 0.5
        if mse < 0.015:
          score = 1
        if mse >= 0.05:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))
    

# part 3 scoring
def get_args():
    parser = argparse.ArgumentParser()
    # Arm
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)

    # Dynamics
    parser.add_argument('--model_dir', type=str, default="models")

    # Controller
    # parser.add_argument('--action_delta', type=float, default=0.1)
    # parser.add_argument('--action_max', type=float, default=1.5)
    # parser.add_argument('--horizon', type=int, default=20)
    # parser.add_argument('--control_horizon', type=int, default=10)
    # parser.add_argument('--gradient_step', type=float, default=1.0)
    # parser.add_argument('--gradient_iters', type=int, default=100)
    # parser.add_argument('--verbose', type=int, default=0)
    
    return parser.parse_known_args()



def test(arm, dynamics, goal, renderer, controller, gui, args, dist_limit, time_limit):

    num_steps = round(time_limit / args.time_step)
    initial_state = np.zeros((arm.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    # Controller to reach goals
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    arm.goal = goal
    arm.set_state(initial_state)
    if renderer is not None:
        renderer.plot([(arm, 'tab:blue')])
    for s in range(num_steps):
        state = arm.get_state()
        if s % controller.control_horizon == 0:
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
        arm.advance()
        if renderer is not None and gui:
            renderer.plot([(arm, 'tab:blue')])
        new_state = arm.get_state()
        pos_ee = arm.dynamics.compute_fk(new_state)
        dist = np.linalg.norm(pos_ee - goal)
        vel_ee = np.linalg.norm(arm.dynamics.compute_vel_ee(state))
    if dist < dist_limit[0] and vel_ee < 0.5:
            return 'full', pos_ee, vel_ee
    elif dist < dist_limit[1] and vel_ee < 0.5:
            return 'partial', pos_ee, vel_ee
    else:
        return 'fail', pos_ee, vel_ee

# Take random Goal
def sample_goal():
  goal = np.zeros((2,1))
  r = np.random.uniform(low=0.05, high=1.95)
  theta = np.random.uniform(low=np.pi, high=2.0*np.pi)
  goal[0,0] = r * np.cos(theta)
  goal[1,0] = r * np.sin(theta)
  return goal

def get_goal(radius, angle):
    angle -= np.pi/2
    return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1,1)

def score_mpc_true_dynamics(controller, gui):
    args, unknown = get_args()
    GOALS = {
        1 : [get_goal(1, 0.4), get_goal(1, -0.75)],
        2 : [get_goal(1.75, 0.4), get_goal(1.75, -0.75)],
        3 : [get_goal(2.7, 0.5), get_goal(2.5, -1.0)]
    }

    renderer = None
    if gui:
        renderer = Renderer()
        time.sleep(1)

    # Part1: Evaluate controller with perfect dynamics
    score = 0.0
    print("")
    print("Part1: EVALUATING CONTROLLER (with perfect dynamics)")
    print("-----------------------------------------------------")
    for num_links in range(1, 4):
        print("NUM_LINKS:", num_links)

        # Arm
        arm = Robot(
            ArmDynamicsTeacher(
                num_links=num_links,
                link_mass=args.link_mass,
                link_length=args.link_length,
                joint_viscous_friction=args.friction,
                dt=args.time_step
            )
        )

        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        for i, goal in enumerate(GOALS[num_links]):
            print("Test ", i+1)
            try:
                result, pos_ee, vel_ee = test(arm, dynamics, goal, renderer, \
                          controller, gui, args, dist_limit=[0.1, 0.2], time_limit=5.0)
            except NotImplementedError as e:
                print(e)
                print("Skipping tests")
                continue
            if result=='full':
                print(f'Success! :)\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                if i == 0:
                    print('score:', '1.5/1.5')
                    score += 1.5
                else:
                    print('score:', '1.0/1.0')
                    score += 1.0
            elif result=='partial':
                print(f'Partial Success:|\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                if i==0:
                  print('score:', '1.0/1.5')
                  score += 1.0
                else:
                  print('score:', '0.5/1.0')
                  score += 0.5
            elif result=='fail':
                print(f'Fail! :(\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                if i==0:
                  print('score:', '0/1.5')
                else:
                  print('score:', '0/1.0')
    score = (score / 7.5) * 5
    print("       ")
    print("-------------------------")
    print("Part 1 SCORE: ", f"{score}/5")
    print("-------------------------")

    if renderer is not None:
        renderer.plotter.terminate()

def score_mpc_learnt_dynamics(controller, arm_student, model_path, gui):
    args, unknown = get_args()
    GOALS = {
        1 : [get_goal(1, 0.4), get_goal(1, -0.75)],
        2 : [sample_goal() for _ in range(16)],
        3 : [get_goal(2.2, -1.0), get_goal(1.8, -0.25), get_goal(1.5, 7.1), get_goal(1.3, -0.5), get_goal(0.9, 5.1)]
    }

    renderer = None
    if gui:
        renderer = Renderer()
        time.sleep(1)
    # Part2: Evaluate controller with learned dynamics
    score = 0.0
    print("Part2: EVALUATING CONTROLLER + LEARNED DYNAMICS")
    print("-----------------------------------------------")
    for num_links in range(2, 3):
        print("NUM_LINKS:", num_links)
        # Arm
        arm = Robot(
            ArmDynamicsTeacher(
                num_links=num_links,
                link_mass=args.link_mass,
                link_length=args.link_length,
                joint_viscous_friction=args.friction,
                dt=args.time_step
            )
        )

        # Learnt dynamics
        dynamics = arm_student
        if not os.path.exists(model_path):
            print(f"model not found at {model_path}, skipping tests")
            continue
        try:
            dynamics.init_model(model_path, num_links, args.time_step, device=torch.device('cpu'))
        except Exception as e:
            print(e)
            print(f"Skipping tests")
            continue
        
        for i, goal in enumerate(GOALS[num_links]):
            print("Test ", i+1)
            try:
                result, pos_ee, vel_ee = test(arm, dynamics, goal, renderer, \
                          controller, gui, args, dist_limit=[0.2, 0.3], time_limit=2.5)
            except Exception as e:
                print(e)
                continue
            if result=='full':
                print(f'Success! :)\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                print('score:', '0.5/0.5')
                score += 0.5
            elif result=='partial':
                print(f'Partial success :|\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                print('score:', '0.3/0.5')
                score += 0.25
            else:
                print(f'Fail :(\n Goal: {GOALS[num_links][i].reshape(-1)}, Final position: {pos_ee.reshape(-1)}, Final velocity: {vel_ee.reshape(-1)}')
                print('score:', '0/0.5')
    score = (score / 7.5) * 5
    print("       ")
    print("-------------------------")
    print("Part 2 SCORE: ", f"{min(score, 5)}/5")
    print("-------------------------")

    if renderer is not None:
        renderer.plotter.terminate()
