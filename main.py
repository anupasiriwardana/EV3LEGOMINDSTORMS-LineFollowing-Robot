#!/usr/bin/env python3
import json
import random
import time
from ev3dev2.motor import MoveTank, OUTPUT_B, OUTPUT_C
from ev3dev2.sensor.lego import ColorSensor, InfraredSensor
from ev3dev2.sensor import INPUT_1, INPUT_4

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
drive = MoveTank(OUTPUT_B, OUTPUT_C)
color = ColorSensor(INPUT_1)
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

# UPDATED: The Q-Table now has 5 rows (states) and 6 columns (actions)
q_table = [[0.0 for _ in range(6)] for _ in range(5)]

CALIB_MIN = 2 
CALIB_MAX = 32

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
        if val < min_val: min_val = val
        if val > max_val: max_val = val
        time.sleep(0.05)
        
    CALIB_MIN = min_val
    CALIB_MAX = max_val
    print("Calibration Complete! Black Min: {}, White Max: {}\n".format(CALIB_MIN, CALIB_MAX))

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_state():
    val = color.reflected_light_intensity
    span = CALIB_MAX - CALIB_MIN
    if span == 0: span = 1 
    
    if val < CALIB_MIN + (span * 0.2): return 0 # Black
    if val < CALIB_MIN + (span * 0.4): return 1 # Dark Edge
    if val < CALIB_MIN + (span * 0.6): return 2 # Perfect Edge
    if val < CALIB_MIN + (span * 0.8): return 3 # Light Edge
    return 4                                    # White

def get_reward(state):
    if state == 2: return 10
    if state in [1, 3]: return 2
    return -5

# UPDATED: 6 Actions for Smoother Turning
def execute_action(action):
    speed = 20 
    
    if action == 0:   # Forward
        drive.on(speed, speed)
    elif action == 1: # Slight Left
        drive.on(speed * 0.2, speed)
    elif action == 2: # Hard Left
        drive.on(-speed / 2, speed)
    elif action == 3: # Slight Right
        drive.on(speed, speed * 0.2)
    elif action == 4: # Hard Right
        drive.on(speed, -speed / 2)
    elif action == 5: # Reverse
        drive.on(-speed, -speed)
        
    time.sleep(0.15) 

# ==========================================
# 5. OBSTACLE AVOIDANCE
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
# 6. TRAINING LOOP
# ==========================================
def train_robot(episodes=75): # Increased to 75 episodes!
    global EPSILON
    calibrate_sensor() 
    print("Training started...")
    
    for episode in range(episodes):
        state = get_state() 
        
        for step in range(50):
            if sonar.proximity < 4: 
                avoid_obstacle_and_find_path()
                state = get_state() 
                continue 
                
            if random.uniform(0, 1) < EPSILON:
                action = random.randint(0, 5) # UPDATED: Can now randomly pick actions 0 through 5
            else:
                action = q_table[state].index(max(q_table[state])) 
                
            execute_action(action)
            new_state = get_state()
            reward = get_reward(new_state)
            
            old_value = q_table[state][action]
            future_max = max(q_table[new_state])
            new_value = old_value + ALPHA * (reward + GAMMA * future_max - old_value)
            
            q_table[state][action] = new_value 
            state = new_state 
            
        drive.off() 
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            
        print("Episode {} done.".format(episode + 1))

    with open('trained_qtable.json', 'w') as f:
        json.dump(q_table, f)

# ==========================================
# 7. OPTIMIZED RUN
# ==========================================
def run_optimized():
    calibrate_sensor() 
    
    with open('trained_qtable.json', 'r') as f:
        optimized_q_table = json.load(f)

    print("Running perfectly optimized!")
    while True:
        if sonar.proximity < 25:
            avoid_obstacle_and_find_path()
            continue
            
        state = get_state()
        best_action = optimized_q_table[state].index(max(optimized_q_table[state]))
        execute_action(best_action)

# ==========================================
# EXECUTE
# ==========================================
if __name__ == '__main__':
    # train_robot(episodes=100) 
    run_optimized()