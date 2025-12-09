#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Распаренный runner для запуска ARS-политики на Raspberry (MPU6050 + ServoKit),
с онлайн-адаптацией Normalizer и постоянным сохранением его статистик.

Особенности:
- Загружает policy из spot_ars_<AGENT_NUM>_policy (pickle). Если в pickle нет normalizer,
  то создаёт новый и будет его «разгонять» через normalizer.observe(state).
- Warmup: первые N шагов (или пока не достигнута пороговая дисперсия) делаем только normalizer.observe,
  не подаём команды на серво (dry_run). После warmup включаем актуаторы с плавным увеличением масштаба.
- Сохраняем normalizer в файл normalizer_spot_<AGENT_NUM>.npz каждые SAVE_INTERVAL сек.
- Имеется dry_run режим (не подавать команды на сервоприводы).
- Безопасный Ctrl+C — переводит сервы в нейтральную позицию.

Требуется настроить:
- MODELS_PATH, AGENT_NUM
- MAP_JOINT_TO_SERVO (каналы/sign/offset)
- SERVO limits и DT
"""

import os
import sys
import time
import math
import signal
import copy
import logging
from collections import deque
import pickle

import numpy as np
from adafruit_servokit import ServoKit

# Подключаем проект (подстрой путь если нужно)
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# Импорты из репозитория
try:
    from ars_lib.ars import Policy, Normalizer, CD_SCALE, Z_SCALE, RESIDUALS_SCALE, alpha, actions_to_filter
    from spotmicro.Kinematics.SpotKinematics import SpotModel
    from spotmicro.GaitGenerator.Bezier import BezierGait
    from spotmicro.OpenLoopSM.SpotOL import BezierStepper
except Exception as e:
    raise RuntimeError("Не удалось импортировать модули из репозитория. Убедитесь, что PROJECT_ROOT указан верно.") from e

# Драйвер IMU (mpu6050)
try:
    import mpu6050
except Exception:
    raise RuntimeError("Требуется библиотека для MPU6050 (pip install mpu6050-raspberrypi или аналог).")

# --------------- Настройки ---------------
MODELS_PATH = os.path.join(PROJECT_ROOT, "spot_bullet", "models", "no_contact")
AGENT_NUM = 2099
POLICY_BASE = os.path.join(MODELS_PATH, f"spot_ars_{AGENT_NUM}")
NORMALIZER_SAVE = os.path.join(MODELS_PATH, f"normalizer_spot_{AGENT_NUM}.npz")

# Hardware
SERVO_CHANNELS = 16
kit = ServoKit(channels=SERVO_CHANNELS)
MPU_ADDR = 0x68
mpu = mpu6050.mpu6050(MPU_ADDR)

# Control loop
DT = 0.02  # sec, 50 Hz recommended for hardware
WARMUP_STEPS = 200         # сколько вызовов normalizer.observe выполнить до включения актуаторов
MIN_VAR_CLIP = 1e-2        # нижняя граница дисперсии (тот же clip, как в Normalizer)
SAVE_INTERVAL = 30.0       # сохранять normalizer на диск каждые N секунд
SCALE_RAMP_TIME = 10.0     # время (сек) за которое scale_factor увеличится 0->1 после включения
DRY_RUN = False            # True — не отправлять команды на сервоприводы (логировать только)
# Safety servo limits
SERVO_MIN = 0.0
SERVO_MAX = 180.0

# Мэппинг joint -> (servo_channel, sign, offset_deg)
# Заполни под свою плату/механизм
MAP_JOINT_TO_SERVO = {
    ('FL', 0): (11, -1, 0.0),
    ('FL', 1): (10, -1, 0.0),
    ('FL', 2): (12, -1, 0.0),
    ('FR', 0): (4, 1, 0.0),
    ('FR', 1): (5, 1, 0.0),
    ('FR', 2): (3, 1, 0.0),
    ('BL', 0): (1, -1, 0.0),
    ('BL', 1): (2, -1, 0.0),
    ('BL', 2): (0, -1, 0.0),
    ('BR', 0): (14, 1, 0.0),
    ('BR', 1): (13, 1, 0.0),
    ('BR', 2): (15, 1, 0.0),
}

# Complementary filter params
COMP_ALPHA = 0.02
CALIBRATION_SAMPLES = 50
GYRO_BUF_LEN = 5

# Логирование
LOG_FILE = os.path.join(THIS_DIR, "raspi_policy_runner.log")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d - %(levelname)-8s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

running = True
def signal_handler(sig, frame):
    global running
    log.warning("Signal received, stopping loop...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --------------- Вспомогательные функции ---------------

def save_normalizer(normalizer, path):
    """Сохраняем поля normalizer: state (counter array), mean, mean_diff, var"""
    try:
        np.savez(path,
                 state=normalizer.state,
                 mean=normalizer.mean,
                 mean_diff=normalizer.mean_diff,
                 var=normalizer.var)
        log.info("Normalizer saved to %s", path)
    except Exception:
        log.exception("Failed to save normalizer")

def load_normalizer_if_exists(path, expected_dim):
    """Загружает normalizer из .npz если есть, возвращает Normalizer или None"""
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
        normalizer = Normalizer(expected_dim)
        # Assign arrays if shapes match
        for key in ('state', 'mean', 'mean_diff', 'var'):
            if key in data:
                arr = data[key]
                if hasattr(normalizer, key) and getattr(normalizer, key).shape == arr.shape:
                    setattr(normalizer, key, arr)
                else:
                    log.warning("Saved normalizer.%s has shape %s but expected %s",
                                key, arr.shape, getattr(normalizer, key).shape)
        log.info("Loaded normalizer from %s", path)
        return normalizer
    except Exception:
        log.exception("Failed to load normalizer file")
        return None

def load_policy(path_base):
    """Попытка гибко загрузить policy/normalizer из pickle (типы разные в репе)"""
    candidate = None
    for suffix in ("_policy", ""):
        p = path_base + suffix
        if os.path.exists(p):
            candidate = p
            break
    if candidate is None:
        log.error("Policy file not found at %s(_policy)", path_base)
        return None, None
    try:
        with open(candidate, "rb") as f:
            obj = pickle.load(f, encoding='latin1')
        # Several possible formats
        if hasattr(obj, "policy") and hasattr(obj, "normalizer"):
            log.info("Loaded ARSAgent-like object from %s", candidate)
            return obj.policy, obj.normalizer
        if hasattr(obj, "theta"):
            # obj is Policy
            log.info("Loaded Policy object from %s", candidate)
            policy = obj
            # maybe obj has normalizer attribute
            if hasattr(obj, "normalizer"):
                return policy, obj.normalizer
            return policy, None
        if isinstance(obj, dict):
            # sometimes dict with 'theta' key
            if 'theta' in obj:
                policy = Policy(obj['theta'].shape[1], obj['theta'].shape[0])
                policy.theta = obj['theta']
                normalizer = obj.get('normalizer', None)
                return policy, normalizer
        if isinstance(obj, np.ndarray):
            policy = Policy(obj.shape[1], obj.shape[0])
            policy.theta = obj
            return policy, None
    except Exception:
        log.exception("Failed to load pickle %s", candidate)
    return None, None

def rad_to_servo_angle(rad, sign=1.0, offset_deg=0.0):
    deg = math.degrees(rad) * sign + offset_deg
    deg = max(min(deg, SERVO_MAX), SERVO_MIN)
    return deg

def apply_joint_angles_to_servos(joint_angles, mapping):
    """joint_angles: (4,3) in radians, leg order assumed ['FL','FR','BL','BR']"""
    leg_names = ['FL','FR','BL','BR']
    for leg_idx, leg in enumerate(leg_names):
        for j in range(3):
            key = (leg, j)
            if key not in mapping:
                continue
            ch, sign, offset = mapping[key]
            rad = float(joint_angles[leg_idx][j])
            angle = rad_to_servo_angle(rad, sign, offset)
            try:
                kit.servo[ch].angle = angle
            except Exception:
                log.exception("Failed to write servo channel %d", ch)

# --------------- IMU helpers ---------------

def imu_calibrate(mpu, n=CALIBRATION_SAMPLES, dt=DT):
    """Короткая калибровка gyro bias и accel roll/pitch bias"""
    log.info("Calibrating IMU for %d samples...", n)
    gx_b = gy_b = gz_b = 0.0
    roll_acc = 0.0
    pitch_acc = 0.0
    for i in range(n):
        a = mpu.get_accel_data()
        g = mpu.get_gyro_data()
        gx = math.radians(g['x']); gy = math.radians(g['y']); gz = math.radians(g['z'])
        gx_b += gx; gy_b += gy; gz_b += gz
        roll_acc += math.atan2(a['x'], a['z'])
        pitch_acc += math.atan2(a['y'], a['z'])
        time.sleep(dt)
    biases = {
        'gx': gx_b / n,
        'gy': gy_b / n,
        'gz': gz_b / n,
        'roll_acc': roll_acc / n,
        'pitch_acc': pitch_acc / n
    }
    log.info("IMU biases: gx=%.6f gy=%.6f gz=%.6f", biases['gx'], biases['gy'], biases['gz'])
    return biases

# --------------- MAIN runner ---------------

def main():
    global running
    log.info("Starting raspi_policy_runner")

    # 1) Создаём временные объекты TGP/smach/spot только для формирования фаз и IK
    spot = SpotModel()
    TGP = BezierGait(dt=DT)
    smach = BezierStepper(dt=DT)

    # 2) Загружаем policy (и, возможно, normalizer из pickle)
    policy, normalizer_from_pickle = load_policy(POLICY_BASE)
    if policy is None:
        log.error("No policy loaded, abort")
        return
    log.info("policy.theta.shape: %s", policy.theta.shape)

    state_dim = policy.theta.shape[1]
    action_dim = policy.theta.shape[0]

    # 3) Попытка загрузить сохранённый normalizer (если он был ранее сохранён)
    normalizer_saved = load_normalizer_if_exists(NORMALIZER_SAVE, state_dim)
    if normalizer_saved is not None:
        normalizer = normalizer_saved
    elif normalizer_from_pickle is not None:
        normalizer = normalizer_from_pickle
    else:
        normalizer = Normalizer(state_dim)
        log.info("No normalizer available in pickle. Created new Normalizer (will warmup online).")

    # Лог начального состояния normalizer
    log.info("normalizer mean: %s", normalizer.mean)
    log.info("normalizer var:  %s", normalizer.var)

    # 4) IMU calibration (biases) for complementary filter
    biases = imu_calibrate(mpu, n=CALIBRATION_SAMPLES, dt=DT)
    filtered_roll = 0.0
    filtered_pitch = 0.0
    gx_buf = deque(maxlen=GYRO_BUF_LEN)
    gy_buf = deque(maxlen=GYRO_BUF_LEN)
    gz_buf = deque(maxlen=GYRO_BUF_LEN)

    # 5) Loop variables
    step_idx = 0
    last_save = time.time()
    actuators_enabled = False
    enable_time = None
    scale_factor = 0.0

    # initial T_bf / T_b0
    T_bf = copy.deepcopy(spot.WorldToFoot)
    T_b0 = copy.deepcopy(spot.WorldToFoot)

    log.info("Entering main loop. Warmup steps: %d (normalizer.observe only). Dry run: %s", WARMUP_STEPS, DRY_RUN)

    while running:
        t0 = time.time()
        # read sensors
        a = mpu.get_accel_data()
        g = mpu.get_gyro_data()

        # gyro rad/s minus bias
        gx = math.radians(g['x']) - biases['gx']
        gy = math.radians(g['y']) - biases['gy']
        gz = math.radians(g['z']) - biases['gz']
        gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)
        gx_f = float(np.mean(gx_buf)); gy_f = float(np.mean(gy_buf)); gz_f = float(np.mean(gz_buf))

        # accel-based roll/pitch
        ax = a['x']; ay = a['y']; az = a['z']
        roll_a = math.atan2(ax, az) - biases['roll_acc']
        pitch_a = math.atan2(ay, az) - biases['pitch_acc']

        # complementary filter integration
        filtered_roll = roll_a * COMP_ALPHA + (1 - COMP_ALPHA) * (filtered_roll + gy_f * DT)
        filtered_pitch = pitch_a * COMP_ALPHA + (1 - COMP_ALPHA) * (filtered_pitch + gx_f * DT)
        true_roll = filtered_pitch
        true_pitch = -filtered_roll

        # smach state
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = smach.StateMachine()

        # replace orientation roll/pitch
        orn = (true_roll, true_pitch, orn[2])

        # generate T_bf and phases
        contacts = [0, 0, 0, 0]
        T_bf = TGP.GenerateTrajectory(StepLength, LateralFraction, YawRate, StepVelocity,
                                      T_b0, T_bf, ClearanceHeight, PenetrationDepth, contacts, DT)
        phases = np.array(TGP.Phases)

        # build 16-d state: [roll, pitch, gx, gy, gz, ax, ay, az, phases(4), contacts(4)]
        state = np.zeros(state_dim, dtype=np.float32)
        state[0:8] = [true_roll, true_pitch, gx, gy, gz, ax, ay, az]
        state[8:12] = phases
        state[12:16] = contacts

        # update normalizer (always observe, both during warmup and later)
        normalizer.observe(state)
        # normalized copy for policy usage (don't normalize contacts)
        state_norm = state.copy()
        state_norm[:-4] = normalizer.normalize(state)[:-4]

        # Count steps and manage warmup
        step_idx += 1
        if (not actuators_enabled) and step_idx >= WARMUP_STEPS:
            actuators_enabled = True
            enable_time = time.time()
            log.info("Warmup complete — enabling actuators (if not dry_run).")

        # compute action but apply to servos only after enabled
        raw_act = policy.evaluate(state_norm, None, None)
        act = np.tanh(raw_act)

        # exponential filter on first actions_to_filter
        if actions_to_filter > 0:
            # very simple damping using alpha and previous value stored in normalizer? we keep local old_act
            # We will store old_act across iterations
            if 'old_act' not in locals():
                old_act = np.zeros_like(act)
            act[:actions_to_filter] = alpha * old_act + (1.0 - alpha) * act[:actions_to_filter]
            old_act = act[:actions_to_filter].copy()

        # scale ramp
        if actuators_enabled:
            elapsed_since_enable = max(0.0, time.time() - enable_time)
            scale_factor = min(1.0, elapsed_since_enable / SCALE_RAMP_TIME)
        else:
            scale_factor = 0.0

        # final applied action scaled by scale_factor for safety
        applied_act = act * scale_factor

        # apply to trajectory / IK
        ClearanceHeight_mod = ClearanceHeight + applied_act[0] * CD_SCALE
        pos_local = list(pos)
        pos_local[2] += applied_act[1] * Z_SCALE

        res = applied_act.copy()
        # scale residuals indices 2..13
        res[2:14] *= RESIDUALS_SCALE

        T_bf_copy = copy.deepcopy(T_bf)
        # follow training code indexing: T_bf[key][3,:3] or [ :3, 3 ] depending on function used.
        # Many codes use T_bf[key][3, :3] or T_bf[key][:3,3] — check your SpotModel conventions.
        # Here use same as training earlier: T_bf["FL"][3,:3] += ...
        try:
            T_bf_copy["FL"][3, :3] += res[2:5]
            T_bf_copy["FR"][3, :3] += res[5:8]
            T_bf_copy["BL"][3, :3] += res[8:11]
            T_bf_copy["BR"][3, :3] += res[11:14]
        except Exception:
            # fallback if indexing different (common is [:3, 3])
            try:
                T_bf_copy["FL"][:3, 3] += res[2:5]
                T_bf_copy["FR"][:3, 3] += res[5:8]
                T_bf_copy["BL"][:3, 3] += res[8:11]
                T_bf_copy["BR"][:3, 3] += res[11:14]
            except Exception:
                log.exception("Failed to add residuals to T_bf_copy with both indexing styles.")

        # IK
        try:
            joint_angles = spot.IK(orn, pos_local, T_bf_copy)  # shape (4,3)
        except Exception:
            log.exception("IK failed")
            joint_angles = None

        # send to servos if enabled and not dry run
        if joint_angles is not None and actuators_enabled and not DRY_RUN:
            apply_joint_angles_to_servos(joint_angles, MAP_JOINT_TO_SERVO)
        else:
            # log what would be applied
            log.debug("DRY: step=%d enabled=%s scale=%.3f act[0]=%.4f act[1]=%.4f", step_idx, actuators_enabled, scale_factor, act[0], act[1])

        # periodic save normalizer
        if time.time() - last_save > SAVE_INTERVAL:
            save_normalizer(normalizer, NORMALIZER_SAVE)
            last_save = time.time()

        # loop timing
        elapsed = time.time() - t0
        to_sleep = DT - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    # end loop
    log.info("Main loop stopped, saving normalizer and moving servos to neutral")
    try:
        save_normalizer(normalizer, NORMALIZER_SAVE)
    except Exception:
        pass

    # move servos to neutral
    if not DRY_RUN:
        for _ in range(10):
            for _, (ch, _, _) in MAP_JOINT_TO_SERVO.items():
                try:
                    kit.servo[ch].angle = 90
                except Exception:
                    pass
            time.sleep(0.1)

    log.info("Shutdown complete.")

if __name__ == "__main__":
    main()