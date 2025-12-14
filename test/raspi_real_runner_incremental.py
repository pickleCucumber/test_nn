#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raspi_real_runner_incremental.py

Pure-Python runner with:
 - baseline compensation (joint_zero) for IK->servo mapping
 - per-loop incremental (non-blocking) servo speed limiter:
     each control loop moves each servo by at most RATE_LIMIT_DEG_PER_SEC * DT degrees
 - DRY_RUN mode (no hardware movement) for safe debugging

This variant is intended for a pure-Python (no ROS/Teensy) deployment using PCA9685/ServoKit.
Tune RATE_LIMIT_DEG_PER_SEC and DT for desired smoothness.
"""
import os
import sys
import time
import math
import pickle
import copy
import numpy as np

# Hardware libs
import board
import busio
import adafruit_mpu6050
from adafruit_servokit import ServoKit

# Project imports (must exist in your repo)
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(THIS_DIR)
try:
    from ars_lib.ars import Normalizer, Policy
    from spotmicro.Kinematics.SpotKinematics import SpotModel
    from spotmicro.GaitGenerator.Bezier import BezierGait
    from spotmicro.OpenLoopSM.SpotOL import BezierStepper
except Exception as e:
    print("ERROR: required project modules missing:", e)
    sys.exit(1)

# ---------------- CONFIG ----------------
AGENT_NUM = 2099
MODELS_PATH = os.path.join(THIS_DIR, "models")
POLICY_FILE = os.path.join(MODELS_PATH, f"spot_ars_{AGENT_NUM}_policy")

I2C_ADDR_KIT0 = 0x40
I2C_ADDR_KIT1 = 0x41

# Control timing
DT = 0.02  # control loop period (s) - must match training if possible

# Servo limits and ramping
SERVO_MIN = 0.0
SERVO_MAX = 180.0
DRY_RUN = True                    # Default safe mode: log commands only
RATE_LIMIT_DEG_PER_SEC = 60.0     # max deg/sec for each servo (tune down for slower)
# Derived per-loop max step: RATE_LIMIT_DEG_PER_SEC * DT

# Scales from your walkers (kept same names)
CD_SCALE = 0.05
RESIDUALS_SCALE = 0.03
Z_SCALE = 0.05
ACTIONS_TO_FILTER = 2
ALPHA = 0.7

# Offsets and neutral angles (adjust if needed)
OFFSET = {
    "FR_clav": 2, "FL_clav": -8, "BL_clav": 0, "BR_clav": -8,
    "FR_hum": 0,  "FL_hum": -15, "BL_hum": 0, "BR_hum": 0,
    "FR_rad": 0,  "FL_rad": 3, "BL_rad": 0, "BR_rad": 0,
}
NEUTRAL_ANGLES = {
    "FR_clav": 90, "FL_clav": 90, "BL_clav": 90, "BR_clav": 90,
    "FR_hum": 60,  "FL_hum": 120,  "BL_hum": 120,  "BR_hum": 60,
    "FR_rad": 90,  "FL_rad": 90,  "BL_rad": 90,  "BR_rad": 90,
}

MAP_JOINT_TO_SERVO = {
    ('FL', 0): (0, 4, -1, 0.0), ('FL', 1): (0, 5, -1, 0.0), ('FL', 2): (0, 6, -1, 0.0),
    ('FR', 0): (0, 0, 1, 0.0),  ('FR', 1): (0, 1, 1, 0.0),  ('FR', 2): (0, 2, 1, 0.0),
    ('BL', 0): (1, 8, -1, 0.0), ('BL', 1): (1, 9, -1, 0.0), ('BL', 2): (1,10, -1, 0.0),
    ('BR', 0): (1,12, 1, 0.0),  ('BR', 1): (1,13, 1, 0.0),  ('BR', 2): (1,14, 1, 0.0),
}
JOINT_NAMES = {
    ('FL',0): "FL_clav", ('FL',1): "FL_hum", ('FL',2): "FL_rad",
    ('FR',0): "FR_clav", ('FR',1): "FR_hum", ('FR',2): "FR_rad",
    ('BL',0): "BL_clav", ('BL',1): "BL_hum", ('BL',2): "BL_rad",
    ('BR',0): "BR_clav", ('BR',1): "BR_hum", ('BR',2): "BR_rad",
}

# Last commanded angles mapping
LAST_SERVO_ANGLES = {}  # {(kit_idx,ch): angle_deg}

# ---------------- Hardware init ----------------
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(i2c)
    kit0 = ServoKit(channels=16, address=I2C_ADDR_KIT0)
    kit1 = ServoKit(channels=16, address=I2C_ADDR_KIT1)
    kits = {0: kit0, 1: kit1}
    print("Hardware initialized.")
except Exception as e:
    print("Hardware init failed:", e)
    sys.exit(1)

# ---------------- Helpers ----------------
def clamp(x, a, b):
    return max(a, min(b, x))

def init_last_servo_angles_from_neutral():
    """Init LAST_SERVO_ANGLES from NEUTRAL_ANGLES+OFFSET."""
    global LAST_SERVO_ANGLES
    LAST_SERVO_ANGLES = {}
    for (leg, j), (kit_idx, ch, sign, _) in MAP_JOINT_TO_SERVO.items():
        name = JOINT_NAMES[(leg, j)]
        LAST_SERVO_ANGLES[(kit_idx, ch)] = float(NEUTRAL_ANGLES.get(name, 90.0) + OFFSET.get(name, 0.0))

def ik_to_servo_commands(joint_angles, deg_baseline):
    """
    Map joint_angles (4x3 radians) to servo angles (degrees) with baseline compensation.
    Returns dict {(kit_idx,ch): angle_deg}.
    """
    legs = ['FL','FR','BL','BR']
    commands = {}
    for i, leg in enumerate(legs):
        for j in range(3):
            key = (leg, j)
            kit_idx, ch, sign, _ = MAP_JOINT_TO_SERVO[key]
            name = JOINT_NAMES[key]
            deg_ik = math.degrees(joint_angles[i][j])
            deg_base = deg_baseline[i][j]
            neutral_physical = NEUTRAL_ANGLES.get(name, 90.0) + OFFSET.get(name, 0.0)
            servo_deg = neutral_physical + sign * (deg_ik - deg_base)
            servo_deg = clamp(servo_deg, SERVO_MIN, SERVO_MAX)
            commands[(kit_idx, ch)] = servo_deg
    return commands

def per_loop_incremental_update(kits, servo_cmds, dt, rate_deg_per_sec, dry_run=True):
    """
    Non-blocking per-loop incremental update.
    For each (kit,ch): move from LAST_SERVO_ANGLES to target by at most rate*dt degrees,
    write that intermediate angle to hardware (or log in dry_run).
    Updates LAST_SERVO_ANGLES in place.
    """
    global LAST_SERVO_ANGLES
    max_step = rate_deg_per_sec * dt
    updated = {}
    for (kit_idx, ch), target in servo_cmds.items():
        last = LAST_SERVO_ANGLES.get((kit_idx, ch), target)
        diff = target - last
        if abs(diff) <= max_step:
            new = target
        else:
            new = last + math.copysign(max_step, diff)
        new = clamp(new, SERVO_MIN, SERVO_MAX)
        # Write or log
        if dry_run or kits.get(kit_idx) is None:
            print(f"[DRY_RUN] kit={kit_idx} ch={ch} -> {new:.2f} (target {target:.2f})")
        else:
            try:
                kits[kit_idx].servo[ch].angle = float(new)
            except Exception as e:
                print(f"Servo write failed kit {kit_idx} ch {ch}: {e}")
        updated[(kit_idx, ch)] = new
    LAST_SERVO_ANGLES.update(updated)

def set_servos_to_neutral(dry_run=True):
    """Single-shot set to neutral (no ramp) then populate LAST_SERVO_ANGLES."""
    neutral_map = {}
    for (leg, j), (kit_idx, ch, sign, _) in MAP_JOINT_TO_SERVO.items():
        name = JOINT_NAMES[(leg, j)]
        neutral_map[(kit_idx, ch)] = NEUTRAL_ANGLES.get(name, 90.0) + OFFSET.get(name, 0.0)
    # apply
    for (kit_idx, ch), angle in neutral_map.items():
        if dry_run or kits.get(kit_idx) is None:
            print(f"[DRY_RUN] neutral kit={kit_idx} ch={ch} -> {angle:.2f}")
        else:
            try:
                kits[kit_idx].servo[ch].angle = float(angle)
            except Exception as e:
                print(f"Failed to set neutral kit {kit_idx} ch {ch}: {e}")
    # init last angles
    LAST_SERVO_ANGLES.update({k: float(v) for k, v in neutral_map.items()})
    time.sleep(0.2)

# ---------------- Env wrapper ----------------
class RealSpotEnv:
    def __init__(self, spot, tgp, dt=DT):
        self.spot = spot
        self.tgp = tgp
        self.dt = dt
        self.roll = 0.0
        self.pitch = 0.0
        self.alpha_filter = 0.98

    def get_observation(self):
        acc = mpu.acceleration
        gyro = mpu.gyro
        roll_acc = math.atan2(acc[1], acc[2])
        pitch_acc = math.atan2(-acc[0], math.sqrt(acc[1]**2 + acc[2]**2))
        # Note: adafruit_mpu6050.gyro returns rad/s (verify on your setup). If it returns deg/s, convert.
        self.roll = self.alpha_filter * (self.roll + gyro[0] * self.dt) + (1 - self.alpha_filter) * roll_acc
        self.pitch = self.alpha_filter * (self.pitch + gyro[1] * self.dt) + (1 - self.alpha_filter) * pitch_acc
        phases = getattr(self.tgp, 'Phases', np.zeros(4))
        contacts = np.zeros(4)
        obs = np.concatenate([
            np.array([self.roll, self.pitch], dtype=np.float32),
            np.array(gyro, dtype=np.float32),
            np.array(acc, dtype=np.float32),
            np.array(phases, dtype=np.float32),
            np.array(contacts, dtype=np.float32)
        ])
        return obs, contacts

# ---------------- Main ----------------
def main():
    print("Starting incremental-runner (DRY_RUN=%s, RATE=%s deg/s, DT=%s)" % (DRY_RUN, RATE_LIMIT_DEG_PER_SEC, DT))
    spot = SpotModel()
    tgp = BezierGait(dt=DT)
    smach = BezierStepper(dt=DT)
    env = RealSpotEnv(spot, tgp, dt=DT)

    # compute deg_baseline
    try:
        T_bf0 = copy.deepcopy(spot.WorldToFoot)
        joint_zero = spot.IK([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], T_bf0)
        deg_baseline = [[math.degrees(x) for x in leg] for leg in joint_zero]
        print("deg_baseline:", deg_baseline)
    except Exception as e:
        print("Failed to compute deg_baseline:", e)
        deg_baseline = [[0.0, 0.0, 0.0] for _ in range(4)]

    # load policy
    print("Loading policy:", POLICY_FILE)
    try:
        with open(POLICY_FILE, 'rb') as f:
            policy = pickle.load(f, encoding='latin1')
        state_dim = policy.theta.shape[1]
        normalizer = Normalizer(state_dim)
        print("Policy loaded, normalizer created. state_dim=", state_dim)
    except Exception as e:
        print("Failed to load policy:", e)
        return

    # init neutral and LAST_SERVO_ANGLES
    set_servos_to_neutral(dry_run=DRY_RUN)

    # initial hardware test (one channel) - dry-run respects flag
    fl_clav_neutral = NEUTRAL_ANGLES["FL_clav"] + OFFSET["FL_clav"]
    print("Quick test: FL clav neutral (dry_run={}) -> {:.1f}".format(DRY_RUN, fl_clav_neutral))

    input("Press ENTER to start policy loop (Ctrl+C to stop)...")

    # main loop variables
    T_bf = copy.deepcopy(spot.WorldToFoot)
    T_b0 = copy.deepcopy(spot.WorldToFoot)
    action_dim = policy.action_dim
    action = np.zeros(action_dim)
    old_act = np.zeros(ACTIONS_TO_FILTER)
    ClearanceHeight = smach.ClearanceHeight
    PenetrationDepth = smach.PenetrationDepth

    try:
        iter_cnt = 0
        while True:
            loop_start = time.time()
            obs, contacts = env.get_observation()
            if len(obs) != state_dim:
                # warn but proceed
                pass
            normalizer.observe(obs)
            norm_obs = normalizer.normalize(obs)
            # evaluate policy
            action = policy.evaluate(norm_obs)
            action = np.tanh(action)
            # filter first params
            action[:ACTIONS_TO_FILTER] = ALPHA * old_act + (1.0 - ALPHA) * action[:ACTIONS_TO_FILTER]
            old_act = action[:ACTIONS_TO_FILTER].copy()

            # deployTG
            pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, _, _ = smach.StateMachine()
            ClearanceHeight_Mod = ClearanceHeight + action[0] * CD_SCALE
            ClearanceHeight_Mod = np.clip(ClearanceHeight_Mod, smach.ClearanceHeight_LIMITS[0], smach.ClearanceHeight_LIMITS[1])
            pos[2] += action[1] * Z_SCALE

            T_bf = tgp.GenerateTrajectory(StepLength, LateralFraction, YawRate, StepVelocity,
                                          T_b0, T_bf, ClearanceHeight_Mod, PenetrationDepth, contacts)

            # residuals
            legs_order = ["FL", "FR", "BL", "BR"]
            act_idx = 2
            for leg in legs_order:
                T_bf[leg][3, 0] += action[act_idx] * RESIDUALS_SCALE
                T_bf[leg][3, 1] += action[act_idx+1] * RESIDUALS_SCALE
                T_bf[leg][3, 2] += action[act_idx+2] * RESIDUALS_SCALE
                act_idx += 3

            # IK -> joint angles
            joint_angles = spot.IK(orn, pos, T_bf)

            # debug print first iterations
            if iter_cnt < 5:
                print("iter", iter_cnt, "obs[:8]", np.array2string(obs[:8], precision=3))
                print("action[:8]", np.array2string(action[:8], precision=3))
                for i, leg in enumerate(['FL','FR','BL','BR']):
                    ja_deg = [math.degrees(x) for x in joint_angles[i]]
                    print(f" IK {leg} deg: {ja_deg} baseline: {deg_baseline[i]}")

            # map IK -> servo targets (degrees) with baseline compensation
            servo_targets = ik_to_servo_commands(joint_angles, deg_baseline)

            # PER-LOOP incremental update (non-blocking)
            per_loop_incremental_update(kits, servo_targets, DT, RATE_LIMIT_DEG_PER_SEC, dry_run=DRY_RUN)

            iter_cnt += 1

            # timing
            elapsed = time.time() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)
            else:
                # loop overrun; print occasionally
                if iter_cnt % 100 == 0:
                    print(f"Loop overrun: {elapsed:.4f}s")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Parking to neutral (single-shot)...")
        set_servos_to_neutral(dry_run=DRY_RUN)
        print("Done. Exiting.")

if __name__ == "__main__":
    main()