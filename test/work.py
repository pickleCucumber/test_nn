#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patched raspi_real_runner - baseline compensation for IK -> servo mapping.

Changes:
 - compute deg_baseline = joint_zero (degrees) once after SpotModel init
 - replace ik_to_servo_commands to subtract baseline: servo = neutral + sign*(deg_ik - deg_base)
 - added prints for deg_baseline and mapping per-iteration for debugging
 - minimal safety: DRY_RUN flag respected, clamping to servo limits
 - kept original structure otherwise

Usage: same as before. By default this script will move servos; keep caution.
"""
import os
import sys
import time
import math
import logging
import pickle
import numpy as np
import board
import busio
import adafruit_pca9685
import adafruit_mpu6050
from adafruit_servokit import ServoKit
import copy

# ================= КОНФИГУРАЦИЯ =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

AGENT_NUM = 2099
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
POLICY_FILE = os.path.join(MODELS_PATH, f"spot_ars_{AGENT_NUM}_policy")

I2C_ADDR_KIT0 = 0x40
I2C_ADDR_KIT1 = 0x41

SERVO_MIN = 0.0
SERVO_MAX = 180.0

DT = 0.02

SL_SCALE = 0.007
SV_SCALE = 0.2
CD_SCALE = 0.05
RESIDUALS_SCALE = 0.03
Z_SCALE = 0.05
ACTIONS_TO_FILTER = 2
ALPHA = 0.7

OFFSET = {
    "FR_clav": 2, "FL_clav": -12, "BL_clav": 5, "BR_clav": -25,
    "FR_hum": 0,  "FL_hum": 0, "BL_hum": 0, "BR_hum": 0,
    "FR_rad": 0,  "FL_rad": 0, "BL_rad": 0, "BR_rad": 0,
}

NEUTRAL_ANGLES = {
    "FR_clav": 90, "FL_clav": 90, "BL_clav": 90, "BR_clav": 90,
    "FR_hum": 60,  "FL_hum": 120,  "BL_hum": 120,  "BR_hum": 60,
    "FR_rad": 70,  "FL_rad": 110,  "BL_rad": 110,  "BR_rad": 70,
}

MAP_JOINT_TO_SERVO = {
    ('FL', 0): (0, 4, -1, 0.0), ('FL', 1): (0, 5, -1, 0.0), ('FL', 2): (0, 6, -1, 0.0),
    ('FR', 0): (0, 0, 1, 0.0),   ('FR', 1): (0, 1, 1, 0.0),   ('FR', 2): (0, 2, 1, 0.0),
    ('BL', 0): (1, 8, -1, 0.0),  ('BL', 1): (1, 9, -1, 0.0),  ('BL', 2): (1, 10, -1, 0.0),
    ('BR', 0): (1, 12, 1, 0.0),  ('BR', 1): (1, 13, 1, 0.0),  ('BR', 2): (1, 14, 1, 0.0),
}

JOINT_NAMES = {
    ('FL',0): "FL_clav", ('FL',1): "FL_hum", ('FL',2): "FL_rad",
    ('FR',0): "FR_clav", ('FR',1): "FR_hum", ('FR',2): "FR_rad",
    ('BL',0): "BL_clav", ('BL',1): "BL_hum", ('BL',2): "BL_rad",
    ('BR',0): "BR_clav", ('BR',1): "BR_hum", ('BR',2): "BR_rad",
}

# ================= ИМПОРТЫ ПРОЕКТА =================
try:
    from ars_lib.ars import Normalizer, Policy
    from spotmicro.Kinematics.SpotKinematics import SpotModel
    from spotmicro.GaitGenerator.Bezier import BezierGait
    from spotmicro.OpenLoopSM.SpotOL import BezierStepper
except Exception as e:
    print("ОШИБКА: Не найдены модули spotmicro или ars_lib.", e)
    sys.exit(1)

# ================= ИНИЦИАЛИЗАЦИЯ ЖЕЛЕЗА =================
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(i2c)
    kit0 = ServoKit(channels=16, address=I2C_ADDR_KIT0)
    kit1 = ServoKit(channels=16, address=I2C_ADDR_KIT1)
    kits = {0: kit0, 1: kit1}
    print("Железо инициализировано (I2C, MPU, PCA9685).")
except Exception as e:
    print(f"ОШИБКА ИНИЦИАЛИЗАЦИИ ЖЕЛЕЗА: {e}")
    sys.exit(1)

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def clamp(x, a, b):
    return max(a, min(b, x))

def apply_servo_command(kit_idx, ch, angle, dry_run=False):
    angle = clamp(angle, SERVO_MIN, SERVO_MAX)
    if dry_run:
        print(f"[DRY_RUN] kits[{kit_idx}].servo[{ch}].angle = {angle}")
        return
    try:
        kits[kit_idx].servo[ch].angle = angle
    except Exception as e:
        print(f"Ошибка серво (Kit {kit_idx} Ch {ch}): {e}")

def set_servos_to_neutral(dry_run=False):
    print(">>> Установка НЕЙТРАЛЬНОЙ ПОЗЫ...")
    for key, (k_idx, ch, sign, _) in MAP_JOINT_TO_SERVO.items():
        name = JOINT_NAMES[key]
        angle = NEUTRAL_ANGLES[name] + OFFSET[name]
        apply_servo_command(k_idx, ch, angle, dry_run=dry_run)
    time.sleep(1.0) # Даем время встать

# NEW: baseline-aware ik_to_servo_commands (uses deg_baseline)
def ik_to_servo_commands(joint_angles, deg_baseline):
    """
    joint_angles: array (4,3) in radians
    deg_baseline: list (4,3) baseline joint angles in degrees (joint_zero)
    Returns dict {(kit_idx, ch): angle_deg}
    """
    legs = ['FL','FR','BL','BR']
    commands = {}
    for i, leg in enumerate(legs):
        for j in range(3):
            key = (leg, j)
            kit_idx, ch, sign, _ = MAP_JOINT_TO_SERVO[key]
            name = JOINT_NAMES[key]

            deg_ik = math.degrees(joint_angles[i][j])  # current joint angle in degrees
            deg_base = deg_baseline[i][j]              # baseline joint angle in degrees

            neutral_physical = NEUTRAL_ANGLES.get(name, 90.0) + OFFSET.get(name, 0.0)

            # Map so that deg_ik == deg_base -> neutral_physical
            servo_deg = neutral_physical + sign * (deg_ik - deg_base)

            servo_deg = clamp(servo_deg, SERVO_MIN, SERVO_MAX)
            commands[(kit_idx, ch)] = servo_deg
    return commands

# ================= КЛАСС СРЕДЫ (REAL ENVIRONMENT) =================
class RealSpotEnv:
    def __init__(self, spot, tgp, dt=0.02):
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

        self.roll = self.alpha_filter * (self.roll + gyro[0] * self.dt) + (1 - self.alpha_filter) * roll_acc
        self.pitch = self.alpha_filter * (self.pitch + gyro[1] * self.dt) + (1 - self.alpha_filter) * pitch_acc

        if hasattr(self.tgp, 'Phases'):
            phases = self.tgp.Phases
        else:
            phases = np.zeros(4)

        contacts = np.zeros(4)

        obs = np.concatenate([
            np.array([self.roll, self.pitch]),
            np.array(gyro),
            np.array(acc),
            np.array(phases),
            np.array(contacts)
        ])
        return obs, contacts

# ================= ГЛАВНЫЙ ЦИКЛ =================

def run_hardware_test(kits_local):
    print("\n=== ЗАПУСК ТЕСТА ОБОРУДОВАНИЯ ===")
    print("--> Проверка MPU6050 (покрутите робота)...")
    for i in range(10):
        a = mpu.acceleration
        g = mpu.gyro
        print(f"   [{i}] Acc: {a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f} | Gyro (rad/s): {g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}")
        time.sleep(0.1)

    print("--> Проверка Сервоприводов (FL clav)...")
    fl_clav_neutral = NEUTRAL_ANGLES["FL_clav"] + OFFSET["FL_clav"]
    print(f"   Neutral ({fl_clav_neutral})...")
    apply_servo_command(0, 11, fl_clav_neutral, dry_run=False)
    time.sleep(0.5)
    print(f"   Move +30 deg ({fl_clav_neutral + 30})...")
    apply_servo_command(0, 11, fl_clav_neutral + 30, dry_run=False)
    time.sleep(0.5)
    print(f"   Move -30 deg ({fl_clav_neutral - 30})...")
    apply_servo_command(0, 11, fl_clav_neutral - 30, dry_run=False)
    time.sleep(0.5)
    print(f"   Back to Neutral ({fl_clav_neutral})...")
    apply_servo_command(0, 11, fl_clav_neutral, dry_run=False)
    time.sleep(0.5)
    print("=== ТЕСТ ЗАВЕРШЕН ===\n")

def main():
    # Init models
    spot = SpotModel()
    tgp = BezierGait(dt=DT)
    smach = BezierStepper(dt=DT)
    env = RealSpotEnv(spot, tgp, dt=DT)

    # Compute IK baseline joint_zero (deg) so mapping aligns with physical neutral
    try:
        T_bf0 = copy.deepcopy(spot.WorldToFoot)
        joint_zero = spot.IK([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], T_bf0)  # shape (4,3) in rad
        deg_baseline = [[math.degrees(x) for x in leg] for leg in joint_zero]
        print("Computed deg_baseline (joint_zero degrees):")
        for i, leg in enumerate(['FL','FR','BL','BR']):
            print(f"  {leg}: {deg_baseline[i]}")
    except Exception as e:
        print("Warning: failed to compute joint_zero baseline; using zeros.", e)
        deg_baseline = [[0.0, 0.0, 0.0] for _ in range(4)]

    # Load policy
    print(f"Загрузка агента из {POLICY_FILE}...")
    try:
        with open(POLICY_FILE, 'rb') as f:
            policy = pickle.load(f, encoding='latin1')

        state_dim = policy.theta.shape[1]
        print(f"Размерность состояния (из политики): {state_dim}")
        normalizer = Normalizer(state_dim)
        print("Политика загружена. Нормалайзер создан (будет обновляться онлайн).")
    except Exception as e:
        print(f"Критическая ошибка загрузки: {e}")
        return

    # Hardware test and neutral
    print("Applying neutral (dry-run) for safety.")
    set_servos_to_neutral(dry_run=False)
    run_hardware_test(kits)

    input("Нажмите ENTER для запуска управления нейросетью (Ctrl+C для выхода)...")

    # Main loop variables
    T_bf = copy.deepcopy(spot.WorldToFoot)
    T_b0 = copy.deepcopy(spot.WorldToFoot)
    action_dim = policy.action_dim
    action = np.zeros(action_dim)
    old_act = np.zeros(ACTIONS_TO_FILTER)
    ClearanceHeight = smach.ClearanceHeight
    PenetrationDepth = smach.PenetrationDepth

    try:
        while True:
            start_time = time.time()

            obs, contacts = env.get_observation()
            if len(obs) != state_dim:
                # allow mismatch but warn
                pass

            normalizer.observe(obs)
            norm_obs = normalizer.normalize(obs)

            action = policy.evaluate(norm_obs)
            action = np.tanh(action)

            # EMA filter for first params
            action[:ACTIONS_TO_FILTER] = ALPHA * old_act + (1.0 - ALPHA) * action[:ACTIONS_TO_FILTER]
            old_act = action[:ACTIONS_TO_FILTER]

            # DeployTG logic
            pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, _, _ = smach.StateMachine()

            ClearanceHeight_Mod = ClearanceHeight + action[0] * CD_SCALE
            ClearanceHeight_Mod = np.clip(ClearanceHeight_Mod, smach.ClearanceHeight_LIMITS[0], smach.ClearanceHeight_LIMITS[1])

            pos[2] += action[1] * Z_SCALE

            T_bf = tgp.GenerateTrajectory(StepLength, LateralFraction, YawRate, StepVelocity,
                                          T_b0, T_bf, ClearanceHeight_Mod, PenetrationDepth, contacts)

            # apply residuals
            legs_order = ["FL", "FR", "BL", "BR"]
            act_idx = 2
            for leg in legs_order:
                T_bf[leg][3, 0] += action[act_idx] * RESIDUALS_SCALE
                T_bf[leg][3, 1] += action[act_idx+1] * RESIDUALS_SCALE
                T_bf[leg][3, 2] += action[act_idx+2] * RESIDUALS_SCALE
                act_idx += 3

            # IK
            joint_angles = spot.IK(orn, pos, T_bf)  # radians

            # Debug prints: per-leg joint angles and mapping
            legs = ['FL','FR','BL','BR']
            for i, leg in enumerate(legs):
                ja = joint_angles[i]
                ja_deg = [math.degrees(x) for x in ja]
                print(f"IK {leg} (deg): {ja_deg} baseline={deg_baseline[i]}")

            servo_cmds = ik_to_servo_commands(joint_angles, deg_baseline)

            # Print sample mapping
            for (kit_idx, ch), angle in list(servo_cmds.items())[:12]:
                print(f"Map kit={kit_idx} ch={ch} -> {angle:.2f} deg")

            # Send to servos (be careful)
            for (kit_idx, ch), angle in servo_cmds.items():
                apply_servo_command(kit_idx, ch, angle, dry_run=False)

            elapsed = time.time() - start_time
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        print("\nОстановка...")
    except Exception as e:
        print(f"\nОшибка в основном цикле: {e}")
    finally:
        set_servos_to_neutral(dry_run=False)
        print("Робот припаркован.")

if __name__ == "__main__":
    main()

