# EV3 Q-Learning Line Follower

This document explains the logic, structure, and execution flow of the optimized reinforcement learning script used to train a LEGO EV3 robot to dynamically follow a line and avoid obstacles.

## 1. Core Logic & Parameters
The robot relies on **Q-Learning**, a trial-and-error algorithm where the robot explores actions, receives a reward or penalty, and updates its "brain" (the Q-Table) using this formula:
$$Q(s, a) = Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **ALPHA** | 0.1 | Learning rate. How much new information overrides old information. |
| **GAMMA** | 0.9 | Discount factor. How much the robot cares about future rewards vs. immediate rewards. |
| **EPSILON** | 0.9 | Starting exploration rate (90% random actions). |
| **EPSILON_DECAY** | 0.95 | Reduces the randomness after each episode so the robot starts relying on its training. |

## 2. The 30-State Memory System
The robot operates using a 30-state Q-table (`NUM_STATES = 30`). This is calculated by multiplying the 5 physical light buckets (Black, Dark Edge, Perfect Edge, Light Edge, White) by the 6 possible physical actions (Forward, Slight/Hard Left, Slight/Hard Right, Reverse). 

Including the `last_action` in the state definition gives the robot short-term memory, preventing erratic zig-zagging.

## 3. Dynamic Calibration (`calibrate_sensor`)
Lighting conditions change rapidly in physical rooms. Instead of hardcoding light thresholds, the program runs a 10-second loop at startup. It records the absolute lowest value (darkest black) and highest value (brightest white). These values (`CALIB_MIN` and `CALIB_MAX`) are used to dynamically calculate the boundaries for the 5 line states.

## 4. The Reward System (`get_reward`)
The robot is trained to follow the exact midpoint between the calibrated black and white readings (the "Perfect Edge"). 
*   **Continuous Points:** The closer the sensor is to the mathematical midpoint, the closer it gets to a +10 reward. As it drifts into solid black or white, the score scales down into negative numbers.
*   **The Anti-Cheat:** Reversing is heavily penalized (-15) so the robot only uses it as a last resort emergency recovery.
*   **Progress Bonus:** Forward movement is rewarded (+2), but *only* if the robot is near the edge.
*   **Oscillation Penalty:** If the robot chooses a hard left immediately after a hard right, it receives a -3 penalty.

## 5. Non-RL Failsafes (`avoid_obstacle_and_find_path`)
The assignment requires obstacle avoidance without Reinforcement Learning. 
*   **Detection:** If the Infrared sensor detects an object closer than 4 units during training (or 25 units during execution), the Q-learning loop is interrupted.
*   **Bypass:** The robot executes a hardcoded 360-degree tread pivot, drives forward, and pivots back.
*   **Recovery & Timeout:** It drives forward slowly searching for the line. If it fails to find the line within 4.0 seconds, a failsafe is triggered to adjust the robot's angle, preventing it from driving endlessly into a wall.

## 6. Training Phase (`train_robot`)
*   **Warm-Starting:** If a previous Q-table file exists, the robot will load it and cut `EPSILON` in half. This allows you to pause training, change batteries, and resume without losing data.
*   **Live Feedback:** During the `steps_per_episode` loop, the terminal prints real-time updates of the State, Action, Reward, and the exact mathematical shift in the Q-table value.
*   **Logging:** At the end of every episode, the full grid of the Q-table is printed to the console, and episode metrics are appended to `training_log.csv`. 
*   **Versioning:** Tables are saved with a timestamp so no training data is accidentally overwritten.

## 7. Execution Phase (`run_optimized`)
When the robot is fully trained, the script is switched to `run_optimized()`. 
*   **Hysteresis:** The robot evaluates the Q-table for the best action. However, to prevent stuttering from minor sensor noise, it will only switch actions if the new action's score beats the current action's score by at least 0.5 (`HYSTERESIS_MARGIN`).
*   **Confidence-Based Speed:** The robot compares its #1 best action score to its #2 best action score. If the gap is large (high confidence), the robot dynamically speeds up. If the gap is small (unsure), the robot slows down to carefully navigate the corner.