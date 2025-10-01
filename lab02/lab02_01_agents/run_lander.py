import gymnasium as gym
import numpy as np

class ReflexAgent:
    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.main_cd = 0

    def __call__(self, obs):
        x, y, vx, vy, theta, vtheta, left_leg, right_leg = obs
        p = self.p

        # cooldown
        if self.main_cd > 0:
            self.main_cd -= 1

        # 1) giữ thăng bằng
        if theta > p["angle_th"]: return 1
        if theta < -p["angle_th"]: return 3

        # pha theo độ cao
        high_alt = y > p["high_y"]
        mid_alt  = p["low_y"] < y <= p["high_y"]
        low_alt  = y <= p["low_y"]

        near_straight = abs(theta) < p["near_straight"]

        # 2) dùng ga chính có kiểm soát
        if high_alt and vy < p["fall_high"] and near_straight and self.main_cd == 0:
            self.main_cd = p["cd"]
            return 2
        if mid_alt and vy < p["fall_mid"] and near_straight and self.main_cd == 0:
            self.main_cd = p["cd"]
            return 2
        if low_alt and near_straight:
            if vy < p["fall_low"] and self.main_cd == 0:
                self.main_cd = p["cd"]
                return 2
            if (left_leg or right_leg) and vy < p["leg_fall"] and self.main_cd == 0:
                self.main_cd = p["cd"]
                return 2

        if self.main_cd > 0:
            return 2

        # 3) điều khiển ngang (tùy pha)
        if high_alt:
            if x > p["x_high"] or vx > p["vx_high"]: return 1
            if x < -p["x_high"] or vx < -p["vx_high"]: return 3
        elif mid_alt:
            if x > p["x_mid"] or vx > p["vx_mid"]: return 1
            if x < -p["x_mid"] or vx < -p["vx_mid"]: return 3
        else:  # low_alt
            if x > p["x_low"] or vx > p["vx_low"]: return 1
            if x < -p["x_low"] or vx < -p["vx_low"]: return 3
            if (left_leg or right_leg) and vy > -0.10 and near_straight:
                return 0

        return 0

def run_episode(agent, render=True, seed=42, max_steps=1000):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    obs, info = env.reset(seed=seed)
    agent.reset()  # quan trọng: reset state mỗi episode
    total = 0.0
    main_count = 0
    steps = 0

    for _ in range(max_steps):
        action = agent(obs)
        if action == 2:
            main_count += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()
    print(f"Episode return: {total:.2f}")
    print(f"Main engine usage: {main_count}/{steps} = { (main_count/steps if steps else 0):.2%}")
    return total, main_count, steps

soft_landing = {
    "angle_th": 0.07, "near_straight": 0.05, "cd": 1,
    "high_y": 0.60, "low_y": 0.25,
    "fall_high": -0.45, "fall_mid": -0.40, "fall_low": -0.22, "leg_fall": -0.10,
    "x_high": 0.25, "vx_high": 0.40,
    "x_mid": 0.18, "vx_mid": 0.30,
    "x_low": 0.12, "vx_low": 0.22,
}

fuel_saver = {
    "angle_th": 0.08, "near_straight": 0.05, "cd": 1,
    "high_y": 0.70, "low_y": 0.20,
    "fall_high": -0.50, "fall_mid": -0.45, "fall_low": -0.24, "leg_fall": -0.12,
    "x_high": 0.28, "vx_high": 0.45,
    "x_mid": 0.20, "vx_mid": 0.35,
    "x_low": 0.14, "vx_low": 0.25,
}
balanced_v1 = {
    # cân bằng
    "angle_th": 0.06,      # nhạy hơn vss 0.07
    "near_straight": 0.05,
    "cd": 1,

    # chia pha theo độ cao
    "high_y": 0.65,
    "low_y": 0.28,

    # ngưỡng hãm rơi (chặt ở cao, nới dần khi xuống thấp)
    "fall_high": -0.48,
    "fall_mid":  -0.42,
    "fall_low":  -0.20,    # mở hơn soft_landing (-0.22 -> -0.20)
    "leg_fall":  -0.08,    # nếu đã chạm chân mà vẫn rơi nhanh -> nhá ga

    # điều khiển ngang theo pha (siết nhẹ ở thấp để vào tâm)
    "x_high": 0.25, "vx_high": 0.40,
    "x_mid":  0.17, "vx_mid": 0.28,
    "x_low":  0.10, "vx_low": 0.20,
}
def evaluate(params, n_seeds=10):
    agent = ReflexAgent(params)
    scores = []
    mains = []
    steps = []
    for s in range(n_seeds):
        ret, main_cnt, n = run_episode(agent, render=False, seed=100+s)
        scores.append(ret); mains.append(main_cnt); steps.append(n)
    mean_score = float(np.mean(scores))
    mean_main_pct = float(np.mean([m/n for m, n in zip(mains, steps) if n]))
    print(f"\nMean return over {n_seeds} seeds: {mean_score:.2f}")
    print(f"Mean main usage: {mean_main_pct:.2%}")
    return mean_score, mean_main_pct

balanced_v2 = {
    # Giữ thăng bằng nhạy hơn chút
    "angle_th": 0.055, "near_straight": 0.05, "cd": 1,

    # Pha cao/trung/thấp
    "high_y": 0.60, "low_y": 0.26,

    # Hãm rơi (giữ chặt ở cao/trung, mở nhẹ ở thấp)
    "fall_high": -0.50,
    "fall_mid":  -0.43,
    "fall_low":  -0.18,  # mở hơn để tránh 0% ga ở sát đất
    "leg_fall":  -0.06,  # nếu chân đã chạm mà còn rơi nhanh thì nhá ga

    # Điều khiển ngang (chặt hơn chút ở thấp để vào tâm)
    "x_high": 0.25, "vx_high": 0.40,
    "x_mid":  0.17, "vx_mid": 0.28,
    "x_low":  0.09, "vx_low": 0.18,
}

# ví dụ chạy:
evaluate(balanced_v2, n_seeds=8)


