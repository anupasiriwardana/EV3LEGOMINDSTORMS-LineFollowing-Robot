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
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.console import Console

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
drive = MoveTank(OUTPUT_B, OUTPUT_C)
color = ColorSensor(INPUT_1)
sonar = InfraredSensor(INPUT_4) 

screen = Console()
screen.set_font('Lat15-TerminusBold32x16', reset_console=False)

color.mode = 'COL-REFLECT'

# ==========================================
# 2. Q-LEARNING PARAMETERS
# ==========================================
ALPHA = 0.1       
GAMMA = 0.9       
EPSILON = 0.9     
EPSILON_MIN = 0.1 
EPSILON_DECAY = 0.95 

NUM_LINE_STATES = 5     
NUM_ACTIONS = 6          
NUM_STATES = NUM_LINE_STATES * NUM_ACTIONS   

OSCILLATION_PENALTY = -3        
HYSTERESIS_MARGIN = 0.5         

CALIB_MIN = 2
CALIB_MAX = 32

QTABLE_LATEST = 'trained_qtable_latest.json'
LOG_FILE = 'training_log.csv'

OPPOSING_ACTIONS = {2: 4, 4: 2, 1: 3, 3: 1}  

# ==========================================
# 3. DYNAMIC CALIBRATION
# ==========================================
def calibrate_sensor():
    print("\n--- SENSOR CALIBRATION ---")
    print("Hover over the BLACK and WHITE floor for 10 seconds...")
    # Send massive text to the robot's screen!
    # screen.text_at('CALIBRATING!', column=1, row=2)
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
    print("Black Min: {}, White Max: {}\n".format(CALIB_MIN, CALIB_MAX))

    # Update the EV3 physical screen (Row 2 gets a title, Row 3 gets the numbers)
    # screen.text_at('Done!', column=1, row=2)
    # screen.text_at('Min:{} Max:{}'.format(CALIB_MIN, CALIB_MAX), column=1, row=3)
    


# ==========================================
# 4. STATE / REWARD HELPERS
# ==========================================
def get_line_bucket():
    val = color.reflected_light_intensity
    span = max(CALIB_MAX - CALIB_MIN, 1)

    if val < CALIB_MIN + (span * 0.2): return 0  
    if val < CALIB_MIN + (span * 0.4): return 1  
    if val < CALIB_MIN + (span * 0.6): return 2  
    if val < CALIB_MIN + (span * 0.8): return 3  
    return 4      

def get_state(last_action):
    bucket = get_line_bucket()
    return bucket * NUM_ACTIONS + last_action, bucket

def get_reward(bucket, last_action, action):
    val = color.reflected_light_intensity
    span = max(CALIB_MAX - CALIB_MIN, 1)
    target = CALIB_MIN + span * 0.5   

    error = abs(val - target) / span  
    reward = 10 - (error * 15)        

    # 1. The Anti-Cheat: Heavily penalize reversing
    if action == 5:
        reward -= 15  

    # 2. The Smart Progress Bonus
    elif action == 0:
        # Only reward forward momentum if we are actually near the edge!
        if bucket in [1, 2, 3]: 
            reward += 2   
        # If we are lost in the black (0) or white (4), penalize forward driving!
        else:
            reward -= 5

    # 3. The Anti-Wobble
    if OPPOSING_ACTIONS.get(last_action) == action:
        reward += OSCILLATION_PENALTY

    return reward

# ==========================================
# 5. ACTIONS
# ==========================================
def execute_action(action, speed=15):
    if action == 0:    # Forward
        drive.on(speed, speed)
    elif action == 1:  # Slight Left
        drive.on(0, speed) # Stop left tread, drive right
    elif action == 2:  # Hard Left Pivot (For Treads!)
        drive.on(-speed, speed) 
    elif action == 3:  # Slight Right
        drive.on(speed, 0) # Stop right tread, drive left
    elif action == 4:  # Hard Right Pivot (For Treads!)
        drive.on(speed, -speed)
    elif action == 5:  # Reverse
        drive.on(-speed, -speed)

    # Increase execution time slightly to give treads time to grip and turn
    # Increase execution time slightly to give treads time to grip and turn
    time.sleep(0.05)

# ==========================================
# 6. OBSTACLE AVOIDANCE
# ==========================================
def avoid_obstacle_and_find_path():
    # screen.text_at('AVOIDING OBSTACLE!', column=1, row=2)
    print("Obstacle! Executing Triangle Evasion.")
    drive.off()
    time.sleep(0.5)

    drive.on_for_seconds(-20, -20, 1)

    # 1. Turn slightly right (~60 degrees) to angle away from the obstacle
    drive.on_for_degrees(20, -20, 220) 
    
    # 2. Drive past the obstacle
    drive.on_for_seconds(20, 20, 3)

    drive.on_for_seconds(10, 20, 3)
    
    # 3. Turn heavily left (~120 degrees) to face BACK towards the line
    drive.on_for_degrees(0, 20, 480) 

    drive.on_for_seconds(20, 20, 2)
    
    screen.text_at('SEARCHING FOR LINE!', column=1, row=2)
    drive.on(20, 20)

    target_edge = CALIB_MIN + ((CALIB_MAX - CALIB_MIN) * 0.6)

    # 4. Simply drive forward until it hits the black line. 
    # Because it is angled inward, it is geometrically guaranteed to hit it.
    while color.reflected_light_intensity > target_edge:
        time.sleep(0.05)

    # 5. Stop. The RL script will immediately read "Pure Black" or "Dark Edge"
    # and automatically steer right to correct itself!
    drive.off()
    screen.text_at('FOUND THE LINE!', column=1, row=2)
    time.sleep(0.5)

# ==========================================
# 7. Q-TABLE LOAD / SAVE / PRINT HELPERS
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

# --- NEW FORMATTED PRINT FUNCTION ---
def print_current_q_table(q_table):
    """Prints a neatly formatted grid of the Q-Table to the terminal."""
    print("\n" + "="*65)
    print("                 CURRENT Q-TABLE SNAPSHOT")
    print("="*65)
    print(" State |   Fwd | S-Lft | H-Lft | S-Rgt | H-Rgt |   Rev ")
    print("-" * 65)
    for s, row in enumerate(q_table):
        # Format each number in the row to perfectly align with 1 decimal place
        row_str = " | ".join(["{:5.1f}".format(val) for val in row])
        print(" S_{:02d}  | {}".format(s, row_str))
    print("="*65 + "\n")

# ==========================================
# 8. TRAINING LOOP 
# ==========================================
def train_robot(episodes=75, warm_start=True, steps_per_episode=50):
    global EPSILON

    calibrate_sensor()

    if warm_start and os.path.isfile(QTABLE_LATEST):
        print("Warm-starting from {}".format(QTABLE_LATEST))
        q_table = load_q_table(QTABLE_LATEST)
        EPSILON = max(EPSILON_MIN, EPSILON * 0.5)  
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
                action_type = "EXPLORE"
            else:
                action = q_table[state].index(max(q_table[state]))
                action_type = "EXPLOIT"

            execute_action(action)

            reward = get_reward(bucket, last_action, action)
            episode_reward += reward

            new_state, new_bucket = get_state(action)

            old_value = q_table[state][action]
            future_max = max(q_table[new_state])
            
            # The Q-Learning Equation
            new_value = old_value + ALPHA * (reward + GAMMA * future_max - old_value)
            q_table[state][action] = new_value
            
            # --- NEW LIVE UPDATE PRINT ---
            print("Step {:03d} [{}] | State: {:02d} -> Action: {} | Reward: {:+.1f} | Q: {:+.2f} -> {:+.2f}".format(
                step + 1, action_type, state, action, reward, old_value, new_value))

            state = new_state
            bucket = new_bucket
            last_action = action

        drive.off()
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        log_episode(episode + 1, episode_reward, EPSILON)
        
        # --- PRINT THE FULL TABLE AT THE END OF THE EPISODE ---
        print_current_q_table(q_table)
        
        print("Episode {} done. Total reward: {:.1f}, epsilon: {:.3f}\n".format(
            episode + 1, episode_reward, EPSILON))
        
        time.sleep(1) # Brief pause so you can read the table before the next episode starts

    save_q_table(q_table, episodes)

# ==========================================
# 9. OPTIMIZED RUN 
# ==========================================
def run_optimized(qtable_path=QTABLE_LATEST):
    calibrate_sensor()
    optimized_q_table = load_q_table(qtable_path)

    screen.text_at('RUNNING...', column=1, row=2)
    print("Running with trained policy from {}".format(qtable_path))
    last_action = 0
    
    # Track left and right spins independently
    consecutive_lefts = 0
    consecutive_rights = 0

    while True:
        # 1. Check for physical obstacles
        if sonar.proximity < 4:
            avoid_obstacle_and_find_path()
            last_action = 0
            consecutive_lefts = 0
            consecutive_rights = 0
            continue

        # 2. Get Q-Table Action
        state, _ = get_state(last_action)
        values = optimized_q_table[state]

        sorted_vals = sorted(values, reverse=True)
        best_value = sorted_vals[0]
        second_best = sorted_vals[1] if len(sorted_vals) > 1 else best_value
        confidence = best_value - second_best

        best_action = values.index(best_value)

        # 3. Apply Hysteresis (Stubbornness)
        if best_action != last_action:
            current_value = values[last_action]
            if best_value - current_value < HYSTERESIS_MARGIN:
                best_action = last_action

        # --- THE DIZZINESS OVERRIDE ---
        if best_action in [1, 2]: # Left Turns
            consecutive_lefts += 1
            consecutive_rights = 0 
        elif best_action in [3, 4]: # Right Turns
            consecutive_rights += 1
            consecutive_lefts = 0 
        else: # Forward (0) or Reverse (5)
            consecutive_lefts = 0
            consecutive_rights = 0

        # If it spins endlessly in one specific direction
        if consecutive_lefts > 130 or consecutive_rights > 130:
            print("Dizziness detected! Breaking out of the spin.")
            screen.text_at('ESCAPING VOID!', column=1, row=2)
            
            # Force the robot to drive straight out of the void
            drive.on(20, 20)
            
            # Calculate the threshold for "White" (Bucket 4)
            span = max(CALIB_MAX - CALIB_MIN, 1)
            white_threshold = CALIB_MIN + (span * 0.8)
            
            # Keep driving forward until the sensor sees the white floor
            while color.reflected_light_intensity < white_threshold:
                time.sleep(0.05)
                
            drive.off() # Stop once white is found
            
            # Reset variables and resume normal behavior
            consecutive_lefts = 0 
            consecutive_rights = 0
            last_action = 0
            screen.text_at('RUNNING...', column=1, row=2)
            continue 
        # ----------------------------------------

        # 4. Calculate Speed and Execute
        speed = min(20, 10 + confidence * 2)
        speed = max(5, speed)  

        execute_action(best_action, speed=speed)
        last_action = best_action
# ==========================================
# EXECUTE
# ==========================================
if __name__ == '__main__':
    # train_robot(episodes=50, warm_start=True, steps_per_episode=100)
    run_optimized()