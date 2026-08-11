#!/usr/bin/env python3
"""
Optimized EV3 line-following robot using Q-learning.

Changes vs. the original version:
  1. Continuous reward shaping (based on raw sensor error, not just bucket).
  2. State now includes "last action" -> reduces zig-zag/oscillation.
  3. Explicit oscillation penalty (hard-left right after hard-right, etc).
  4. Per-episode reward logging to a CSV for progress tracking.
  5. Warm-start retraining: loads existing table instead of always
     starting from zero, and lowers epsilon automatically for fine-tuning.
  6. Confidence-based speed scaling at run time (faster when sure,
     slower when the top two actions are close in value).
  7. Hysteresis on action switching at run time, to avoid chatter from
     sensor noise/glare.
  8. Q-table files are saved with an episode-count + timestamp suffix
     so old runs aren't overwritten, plus a stable "latest" copy.
"""

import json
import random
import time
import os
import csv
from datetime import datetime
from ev3dev2.motor import MoveTank, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.sensor import INPUT_2, INPUT_4

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
drive = MoveTank(OUTPUT_B, OUTPUT_C)
color = ColorSensor(INPUT_2)
sonar = InfraredSensor(INPUT_4) # Using Infrared as per your setup

color.mode = 'COL-REFLECT'

# ==========================================
# 2. Q-LEARNING PARAMETERS
# ==========================================
ALPHA = 0.1       
GAMMA = 0.9       
EPSILON = 0.9     
EPSILON_MIN = 0.1 
EPSILON_DECAY = 0.95 

NUM_LINE_STATES = 5     # black / dark edge / perfect edge / light edge / white
NUM_ACTIONS = 6          # forward, slight-left, hard-left, slight-right, hard-right, reverse
NUM_STATES = NUM_LINE_STATES * NUM_ACTIONS   # line-state x last-action = 30 states

OSCILLATION_PENALTY = -3        # extra penalty for reversing hard turns back-to-back
HYSTERESIS_MARGIN = 0.5         # min Q advantage needed to switch actions at run time

CALIB_MIN = 2
CALIB_MAX = 32

QTABLE_LATEST = 'trained_qtable_latest.json'
LOG_FILE = 'training_log.csv'

# Pairs considered "oscillating" if you do one right after the other
OPPOSING_ACTIONS = {2: 4, 4: 2, 1: 3, 3: 1}  # hard-left<->hard-right, slight-left<->slight-right


# ==========================================
# 3. DYNAMIC CALIBRATION
# ==========================================
def calibrate_sensor():
    print("\n--- SENSOR CALIBRATION ---")
    print("Quickly slide the robot over the BLACK line and WHITE floor for 10 seconds...")
    global CALIB_MIN, CALIB_MAX

    end_time = time.time() + 10.0
    min_val = 100
    max_val = 0

    while time.time() < end_time:
        val = color.reflected_light_intensity
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
        time.sleep(0.05)

    CALIB_MIN = min_val
    CALIB_MAX = max_val
    print("Calibration Complete! Black Min: {}, White Max: {}\n".format(CALIB_MIN, CALIB_MAX))


# ==========================================
# 4. STATE / REWARD HELPERS
# ==========================================
def get_line_bucket():
    """Discrete 0-4 bucket, used for compatibility / readability only."""
    val = color.reflected_light_intensity
    span = max(CALIB_MAX - CALIB_MIN, 1)

    if val < CALIB_MIN + (span * 0.2):
        return 0  # Black
    if val < CALIB_MIN + (span * 0.4):
        return 1  # Dark Edge
    if val < CALIB_MIN + (span * 0.6):
        return 2  # Perfect Edge
    if val < CALIB_MIN + (span * 0.8):
        return 3  # Light Edge
    return 4      # White


def get_state(last_action):
    """Combined state index = line_bucket * NUM_ACTIONS + last_action."""
    bucket = get_line_bucket()
    return bucket * NUM_ACTIONS + last_action, bucket


def get_reward(bucket, last_action, action):
    val = color.reflected_light_intensity
    span = max(CALIB_MAX - CALIB_MIN, 1)
    target = CALIB_MIN + span * 0.5   

    # Base reward: 10 points for perfect, scales down to negative if way off
    error = abs(val - target) / span  
    reward = 10 - (error * 15)        

    # --- THE FIX: NEW ACTION-BASED RULES ---

    # 1. The Anti-Cheat: Heavily penalize reversing
    if action == 5:
        reward -= 15  

    # 2. The Progress Bonus: Reward pure forward momentum
    elif action == 0:
        reward += 2   

    # 3. The Anti-Wobble: Penalize zig-zagging
    if OPPOSING_ACTIONS.get(last_action) == action:
        reward += OSCILLATION_PENALTY

    return reward


# ==========================================
# 5. ACTIONS
# ==========================================
def execute_action(action, speed=20):
    if action == 0:   # Forward
        drive.on(speed, speed)
    elif action == 1:  # Slight Left
        drive.on(speed * 0.2, speed)
    elif action == 2:  # Hard Left
        drive.on(-speed / 2, speed)
    elif action == 3:  # Slight Right
        drive.on(speed, speed * 0.2)
    elif action == 4:  # Hard Right
        drive.on(speed, -speed / 2)
    elif action == 5:  # Reverse
        drive.on(-speed, -speed)

    time.sleep(0.15)


# ==========================================
# 6. OBSTACLE AVOIDANCE (unchanged logic, kept as-is)
# ==========================================
def avoid_obstacle_and_find_path():
    print("Obstacle! Doing hardcoded avoidance.")
    drive.off()
    time.sleep(0.5)

    drive.on_for_degrees(40, -40, 360)
    drive.on_for_seconds(40, 40, 2)
    drive.on_for_degrees(-40, 40, 360)

    print("Searching for line...")
    drive.on(20, 20)

    start_time = time.time()
    target_edge = CALIB_MIN + ((CALIB_MAX - CALIB_MIN) * 0.6)

    while color.reflected_light_intensity > target_edge:
        if time.time() - start_time > 4.0:
            print("Failsafe triggered! Re-adjusting angle...")
            drive.on_for_seconds(-20, -20, 1.5)
            drive.on_for_degrees(30, -30, 90)
            drive.on(20, 20)
            start_time = time.time()
        time.sleep(0.05)

    drive.off()
    print("Found it! Back to RL.")
    time.sleep(0.5)


# ==========================================
# 7. Q-TABLE LOAD / SAVE HELPERS
# ==========================================
def new_q_table():
    return [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]


def load_q_table(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_q_table(q_table, episodes_done):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_name = 'trained_qtable_ep{}_{}.json'.format(episodes_done, timestamp)

    with open(versioned_name, 'w') as f:
        json.dump(q_table, f)

    with open(QTABLE_LATEST, 'w') as f:
        json.dump(q_table, f)

    print("Saved Q-table to {} and {}".format(versioned_name, QTABLE_LATEST))


def log_episode(episode, total_reward, epsilon):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'total_reward', 'epsilon'])
        writer.writerow([episode, round(total_reward, 3), round(epsilon, 4)])


# ==========================================
# 8. TRAINING LOOP (with warm-start)
# ==========================================
def train_robot(episodes=75, warm_start=True, steps_per_episode=50):
    global EPSILON

    calibrate_sensor()

    if warm_start and os.path.isfile(QTABLE_LATEST):
        print("Warm-starting from {}".format(QTABLE_LATEST))
        q_table = load_q_table(QTABLE_LATEST)
        EPSILON = max(EPSILON_MIN, EPSILON * 0.5)  # explore less, we already know something
    else:
        print("Starting from a fresh Q-table")
        q_table = new_q_table()

    print("Training started...")

    for episode in range(episodes):
        last_action = 0
        state, bucket = get_state(last_action)
        episode_reward = 0.0

        for step in range(steps_per_episode):
            if sonar.proximity < 4:
                avoid_obstacle_and_find_path()
                last_action = 0
                state, bucket = get_state(last_action)
                continue

            if random.uniform(0, 1) < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = q_table[state].index(max(q_table[state]))

            execute_action(action)

            reward = get_reward(bucket, last_action, action)
            episode_reward += reward

            new_state, new_bucket = get_state(action)

            old_value = q_table[state][action]
            future_max = max(q_table[new_state])
            q_table[state][action] = old_value + ALPHA * (reward + GAMMA * future_max - old_value)

            state = new_state
            bucket = new_bucket
            last_action = action

        drive.off()
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        log_episode(episode + 1, episode_reward, EPSILON)
        print("Episode {} done. Total reward: {:.1f}, epsilon: {:.3f}".format(
            episode + 1, episode_reward, EPSILON))

    save_q_table(q_table, episodes)


# ==========================================
# 9. OPTIMIZED RUN (with hysteresis + confidence-based speed)
# ==========================================
def run_optimized(qtable_path=QTABLE_LATEST):
    calibrate_sensor()
    optimized_q_table = load_q_table(qtable_path)

    print("Running with trained policy from {}".format(qtable_path))
    last_action = 0

    while True:
        if sonar.proximity < 25:
            avoid_obstacle_and_find_path()
            last_action = 0
            continue

        state, _ = get_state(last_action)
        values = optimized_q_table[state]

        sorted_vals = sorted(values, reverse=True)
        best_value = sorted_vals[0]
        second_best = sorted_vals[1] if len(sorted_vals) > 1 else best_value
        confidence = best_value - second_best

        best_action = values.index(best_value)

        # Hysteresis: only switch away from the current action if the new
        # best action clearly beats it. Reduces jitter from sensor noise.
        if best_action != last_action:
            current_value = values[last_action]
            if best_value - current_value < HYSTERESIS_MARGIN:
                best_action = last_action

        # Confidence-based speed: drive faster when sure, slower when unsure.
        speed = min(30, 15 + confidence * 2)
        speed = max(10, speed)  # floor so it never stalls

        execute_action(best_action, speed=speed)
        last_action = best_action


# ==========================================
# EXECUTE
# ==========================================
if __name__ == '__main__':
    # To (re)train, warm-starting from any existing trained_qtable_latest.json:
    train_robot(episodes=50, warm_start=False, steps_per_episode=100)

    # To run the best saved policy:
    # run_optimized()