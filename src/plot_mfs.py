import os
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

# Ensure output directory exists
os.makedirs("fuzzy_plots", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.titlesize": 14})

# ==========================================
# PLOT 1: ACTION SMOOTHER MEMBERSHIP FUNCTIONS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Universe vectors
x_act = np.linspace(-1.0, 1.0, 200)
x_turb = np.linspace(0.0, 100.0, 200)  # Assuming a max historical turbulence of 100

# MFs for Raw Action
act_sell = fuzz.trimf(x_act, [-1.0, -1.0, 0.0])
act_hold = fuzz.trimf(x_act, [-0.2, 0.0, 0.2])
act_buy = fuzz.trimf(x_act, [0.0, 1.0, 1.0])

ax1.plot(x_act, act_sell, "r-", linewidth=2.5, label="Sell")
ax1.plot(x_act, act_hold, "g-", linewidth=2.5, label="Hold")
ax1.plot(x_act, act_buy, "b-", linewidth=2.5, label="Buy")
ax1.set_title("Input: Raw Agent Action ($a_i$)")
ax1.set_xlabel("Action Range")
ax1.set_ylabel("Membership Degree")
ax1.legend(loc="upper right")
ax1.set_ylim(-0.05, 1.05)

# MFs for Turbulence
turb_low = fuzz.trimf(x_turb, [0.0, 0.0, 30.0])
turb_med = fuzz.trimf(x_turb, [10.0, 50.0, 80.0])
turb_high = fuzz.smf(x_turb, 50.0, 100.0)

ax2.plot(x_turb, turb_low, "c-", linewidth=2.5, label="Low")
ax2.plot(x_turb, turb_med, "m-", linewidth=2.5, label="Medium")
ax2.plot(x_turb, turb_high, "k-", linewidth=2.5, label="High")
ax2.set_title("Input: Market Turbulence ($\\tau_t$)")
ax2.set_xlabel("Turbulence Index Value")
ax2.set_ylabel("Membership Degree")
ax2.legend(loc="upper right")
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("fuzzy_plots/action_smoother_mfs.png", dpi=300)
plt.close()

# ==========================================
# PLOT 2: REWARD SHAPER MEMBERSHIP FUNCTIONS
# ==========================================
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4.5))

x_exp = np.linspace(0.0, 1.0, 200)
x_vix = np.linspace(0.0, 60.0, 200)  # Standard Volatility Index spectrum

# MFs for Exposure
exp_low = fuzz.trimf(x_exp, [0.0, 0.0, 0.4])
exp_high = fuzz.trimf(x_exp, [0.3, 1.0, 1.0])

ax3.plot(x_exp, exp_low, "g-", linewidth=2.5, label="Low Exposure")
ax3.plot(x_exp, exp_high, "r-", linewidth=2.5, label="High Exposure")
ax3.set_title("Input: Portfolio Exposure ($e_t$)")
ax3.set_xlabel("Exposure Ratio")
ax3.set_ylabel("Membership Degree")
ax3.legend(loc="upper right")
ax3.set_ylim(-0.05, 1.05)

# MFs for VIX
vix_calm = fuzz.trimf(x_vix, [0.0, 0.0, 20.0])
vix_stressed = fuzz.trimf(x_vix, [15.0, 30.0, 45.0])
vix_panic = fuzz.smf(x_vix, 35.0, 60.0)

ax4.plot(x_vix, vix_calm, "b-", linewidth=2.5, label="Calm")
ax4.plot(x_vix, vix_stressed, "orange", linewidth=2.5, label="Stressed")
ax4.plot(x_vix, vix_panic, "r-", linewidth=2.5, label="Panic")
ax4.set_title("Input: Market Volatility Index ($v_t$)")
ax4.set_xlabel("VIX Level")
ax4.set_ylabel("Membership Degree")
ax4.legend(loc="upper right")
ax4.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("fuzzy_plots/reward_shaper_mfs.png", dpi=300)
plt.close()

print("Successfully saved publication-grade plots inside the 'fuzzy_plots/' folder.")