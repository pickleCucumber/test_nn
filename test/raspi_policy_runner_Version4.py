#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal runner: initialize kinematics, policy agent and normalizer, then put robot
into neutral (standing) pose.

This file performs:
 - hardware init (two PCA9685 boards)
 - MPU init
 - SpotModel, BezierGait, BezierStepper creation
 - ARSAgent load of policy (and normalizer if present)
 - compute IK neutral joints and send neutral servo angles (uses provided servo_angles & OFFSET)
 - keep process alive (idle) for further manual actions

Before running:
 - adjust MODELS_PATH and AGENT_NUM to point to your model files
 - verify MAP_JOINT_TO_SERVO kit indices/channels match wiring
 - ensure power/grounds are correct for servos
"""

import os
import sys
import time
import math
import logging
import pickle
import copy

from collections import OrderedDict

# Project root - adjust as needed
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# imports from repo
try:
    from ars_lib.ars import ARSAgent, Normalizer, Policy
    from spotmicro.Kinematics.SpotKinematics import SpotModel
    from spotmicro.GaitGenerator.Bezier import BezierGait
    from spotmicro.OpenLoopSM.SpotOL import BezierStepper
    # optional env for ARSAgent.load compatibility
    from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
except Exception as e:
    raise RuntimeError("Failed to import repository modules; check PROJECT_ROOT") from e

# hardware libs
try:
    from adafruit_servokit import ServoKit
    import mpu6050
except Exception as e:
    raise RuntimeError("Missing hardware libraries (adafruit_servokit or mpu6050).") from e

# ----------------- Logging -----------------
LOG_FILE = os.path.join(THIS_DIR, "raspi_policy_runner_init.log")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, "w"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("raspi_policy_runner")

# ----------------- Configuration -----------------
# model to load
MODELS_PATH = os.path.join(PROJECT_ROOT, "spot_bullet", "models", "no_contact")
AGENT_NUM = 2099
POLICY_BASE = os.path.join(MODELS_PATH, f"spot_ars_{AGENT_NUM}")

# Hardware: two PCA9685 boards
SERVO_CHANNELS = 16
KIT0_ADDR = 0x40  # front board
KIT1_ADDR = 0x41  # rear board

# instantiate kits
kit0 = ServoKit(channels=SERVO_CHANNELS, address=KIT0_ADDR)
kit1 = ServoKit(channels=SERVO_CHANNELS, address=KIT1_ADDR)
kits = {0: kit0, 1: kit1}

# IMU
MPU_ADDR = 0x68
mpu = mpu6050.mpu6050(MPU_ADDR)

# Servo safety limits
SERVO_MIN = 0.0
SERVO_MAX = 180.0

# User-provided neutral servo angles and offsets (from your hardware)
servo_angles = {
    "FR_clav": 90, "FL_clav": 90, "BL_clav": 90, "BR_clav": 90,
    "FR_hum": 60,  "FL_hum": 60,  "BL_hum": 60,  "BR_hum": 60,
    "FR_rad": 90,  "FL_rad": 90,  "BL_rad": 90,  "BR_rad": 90,
}
OFFSET = {
    "FR_clav": 2,
    "FL_clav": -8,
    "BL_clav": 0,
    "BR_clav": 0,
    "FR_hum": 0,
    "FL_hum": -15,
    "BL_hum": 0,
    "BR_hum": 0,
    "FR_rad": 0,
    "FL_rad": 3,
    "BL_rad": 0,
    "BR_rad": 0,
}

# MAP_JOINT_TO_SERVO: (kit_index, channel, sign, offset_deg)
# update channels if wiring differs. Using prior suggested mapping:
MAP_JOINT_TO_SERVO = {
    ('FL', 0): (0, 11, -1, 0.0),
    ('FL', 1): (0, 10, -1, 0.0),
    ('FL', 2): (0, 12, -1, 0.0),
    ('FR', 0): (0, 4, 1, 0.0),
    ('FR', 1): (0, 5, 1, 0.0),
    ('FR', 2): (0, 3, 1, 0.0),
    ('BL', 0): (1, 1, -1, 0.0),
    ('BL', 1): (1, 2, -1, 0.0),
    ('BL', 2): (1, 0, -1, 0.0),
    ('BR', 0): (1, 14, 1, 0.0),
    ('BR', 1): (1, 13, 1, 0.0),
    ('BR', 2): (1, 15, 1, 0.0),
}

# helper mapping: (leg,j) -> name used in servo_angles/OFFSET
JOINT_NAME = {
    ('FL',0): "FL_clav", ('FL',1): "FL_hum", ('FL',2): "FL_rad",
    ('FR',0): "FR_clav", ('FR',1): "FR_hum", ('FR',2): "FR_rad",
    ('BL',0): "BL_clav", ('BL',1): "BL_hum", ('BL',2): "BL_rad",
    ('BR',0): "BR_clav", ('BR',1): "BR_hum", ('BR',2): "BR_rad",
}

# ----------------- Utility functions -----------------
def clamp(x, a, b):
    return max(a, min(b, x))

def rad_to_deg(rad):
    return math.degrees(rad)

def apply_servo_command(kit_index, channel, angle_deg):
    """Send angle to the appropriate kit."""
    angle_deg = clamp(angle_deg, SERVO_MIN, SERVO_MAX)
    kit = kits.get(kit_index)
    if kit is None:
        log.error("No kit for index %s", kit_index)
        return
    try:
        kit.servo[channel].angle = float(angle_deg)
    except Exception:
        log.exception("Failed to write servo ch %d on kit %s", channel, kit_index)

def ik_to_servo_commands(joint_angles):
    """
    joint_angles: np.array (4,3) in radians, order ['FL','FR','BL','BR']
    returns dict {(kit_index, channel): angle_deg}
    """
    legs = ['FL','FR','BL','BR']
    commands = {}
    for i, leg in enumerate(legs):
        for j in range(3):
            key = (leg, j)
            if key not in MAP_JOINT_TO_SERVO:
                continue
            kit_index, ch, sign, map_offset = MAP_JOINT_TO_SERVO[key]
            name = JOINT_NAME[key]
            base_neutral = servo_angles.get(name, 90.0)
            user_offset = OFFSET.get(name, 0.0)
            deg_ik = rad_to_deg(joint_angles[i][j])
            # final angle: neutral + user_offset + sign * deg_ik
            servo_deg = base_neutral + user_offset + sign * deg_ik
            servo_deg = clamp(servo_deg, SERVO_MIN, SERVO_MAX)
            commands[(kit_index, ch)] = servo_deg
    return commands

def set_neutral_pose(delay=0.03):
    """Set all servos to neutral positions (from servo_angles + OFFSET)."""
    log.info("Setting neutral pose on all servos...")
    # iterate mapping to ensure all mapped channels set
    for key, (kit_index, ch, sign, map_off) in MAP_JOINT_TO_SERVO.items():
        name = JOINT_NAME.get(key)
        if name is None:
            target = 90.0
        else:
            target = servo_angles.get(name, 90.0) + OFFSET.get(name, 0.0)
        apply_servo_command(kit_index, ch, target)
        time.sleep(delay)
    log.info("Neutral pose applied.")

# ----------------- Agent/model loading -----------------
def load_agent_and_normalizer(agent_base_path):
    """
    Create SpotModel, TGP, smach, temporary env and ARSAgent to load policy+normalizer.
    Returns (policy, normalizer, spot, TGP, smach)
    """
    log.info("Creating SpotModel, BezierGait and BezierStepper for agent loading...")
    spot = SpotModel()
    TGP = BezierGait(dt=0.02)
    smach = BezierStepper(dt=0.02)

    # small env just to satisfy ARSAgent
    env = spotBezierEnv(render=False, on_rack=False, height_field=False, draw_foot_path=False)

    # initial placeholders - sizes will be replaced by agent.load
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    normalizer = Normalizer(state_dim)
    policy = Policy(state_dim, action_dim)

    agent = ARSAgent(normalizer, policy, env, smach=smach, TGP=TGP, spot=spot)
    # try candidate suffixes
    candidate = None
    for suffix in ("_policy", ""):
        p = agent_base_path + suffix
        if os.path.exists(p):
            candidate = p
            break
    if candidate is None:
        log.error("Agent file not found at %s(_policy)", agent_base_path)
        return None, None, spot, TGP, smach

    log.info("Loading agent from %s", candidate)
    try:
        agent.load(agent_base_path)  # agent.load accepts base path without suffix in many repo versions
    except Exception:
        # fallback: try load with the exact candidate
        try:
            agent.load(candidate)
        except Exception:
            log.exception("agent.load failed for both %s and %s", agent_base_path, candidate)
            return None, None, spot, TGP, smach

    policy_trained = agent.policy
    normalizer_trained = agent.normalizer if hasattr(agent, "normalizer") else None

    log.info("Loaded policy theta shape: %s", policy_trained.theta.shape)
    if normalizer_trained is not None:
        log.info("Normalizer mean (sample): %s", normalizer_trained.mean)
    else:
        log.info("No normalizer found inside agent (will warmup online).")

    return policy_trained, normalizer_trained, spot, TGP, smach

# ----------------- Main initialization -----------------
def main():
    log.info("=== raspi_policy_runner init - starting ===")

    # 1) Load agent/policy/normalizer
    policy, normalizer, spot, TGP, smach = load_agent_and_normalizer(POLICY_BASE)
    if policy is None:
        log.error("Policy could not be loaded - aborting init.")
        return

    # If normalizer missing, create new one with expected state dim
    if normalizer is None:
        state_dim = policy.theta.shape[1]
        normalizer = Normalizer(state_dim)
        log.info("Created fresh Normalizer of dim %d; it will be warmed up online.", state_dim)

    # 2) Set neutral pose BEFORE warmup/actuators
    set_neutral_pose(delay=0.02)
    # give servos time
    time.sleep(1.0)

    # 3) Compute IK neutral joint angles (for debug)
    try:
        T_bf = copy.deepcopy(spot.WorldToFoot)
        joint_zero = spot.IK([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], T_bf)
        log.info("Joint neutral angles (rad):\n%s", joint_zero)
        # compute servo commands that would be sent from IK (for verification)
        servo_cmds = ik_to_servo_commands(joint_zero)
        log.info("Computed servo commands from IK-neutral (kit,ch)->deg: %s", servo_cmds)
    except Exception:
        log.exception("Failed to compute joint_zero via IK.")

    # 4) Optionally save normalizer template for later (no-op if mean/var zero)
    try:
        normalizer_path = os.path.join(MODELS_PATH, f"normalizer_spot_{AGENT_NUM}.npz")
        # save initial normalizer state (useful if warmup already has stats)
        import numpy as _np
        _np.savez(normalizer_path, state=normalizer.state, mean=normalizer.mean,
                  mean_diff=normalizer.mean_diff, var=normalizer.var)
        log.info("Saved normalizer snapshot to %s", normalizer_path)
    except Exception:
        log.exception("Failed to save normalizer snapshot (non-fatal).")

    log.info("Initialization complete. Robot is in neutral pose. Entering idle loop (Ctrl-C to exit).")
    try:
        while True:
            # idle; real control loop will be added later
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received - shutting down.")
    finally:
        # shutdown: set neutral again to be safe
        set_neutral_pose(delay=0.02)
        log.info("Shutdown complete.")

if __name__ == "__main__":
    main()