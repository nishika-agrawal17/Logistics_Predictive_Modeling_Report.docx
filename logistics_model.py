"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
------------------------------------------------------------------
Problem : Forecast last-mile delivery time (minutes) for a regional courier network
          and use the forecasts to optimise driver/vehicle assignment, dispatch timing
          and fleet cost.

Run     : python logistics_model.py
Outputs : results.json, data/deliveries.csv and PNG charts in ./figures
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10, "figure.dpi": 150,
})

# ----------------------------------------------------------------------------
# 1. DATA SIMULATION
# ----------------------------------------------------------------------------
VEHICLE_SPEED_KMH = {"bike": 16.0, "van": 32.0, "truck": 26.0}      # free-flow speed
VEHICLE_MAX_KG = {"bike": 10.0, "van": 120.0, "truck": 600.0}
COST_PER_MIN = {"bike": 0.18, "van": 0.42, "truck": 0.75}            # USD per minute
WEATHER_PENALTY = {"clear": 1.00, "rain": 1.18, "fog": 1.30}


def expected_time(df: pd.DataFrame) -> np.ndarray:
    """Noise-free ground-truth delivery time (min). Non-linear and with interactions,
    so that simple linear models are visibly inadequate."""
    speed = df["vehicle_type"].map(VEHICLE_SPEED_KMH).to_numpy()
    weather = df["weather"].map(WEATHER_PENALTY).to_numpy()
    congestion = 1 + 0.07 * df["traffic_index"] ** 1.35 / 3            # grows faster at high traffic
    peak = np.where(df["hour"].isin([8, 9, 17, 18, 19]), 1.22, 1.0)
    weekend = np.where(df["day_of_week"] >= 5, 0.92, 1.0)
    drive = df["distance_km"] / speed * 60 * congestion * weather * peak * weekend
    service = df["num_stops"] * (3.2 + 0.05 * df["package_kg"])        # per-stop handling
    heavy = np.where(df["package_kg"] > 25, 0.35 * (df["package_kg"] - 25), 0)
    exp_gain = -0.9 * np.log1p(df["driver_exp_yrs"]) * (drive + service) / 12
    queue = 0.12 * df["warehouse_queue"]                                # loading wait
    return np.asarray(6 + drive + service + heavy + exp_gain + queue)


def simulate(n: int) -> pd.DataFrame:
    df = pd.DataFrame({
        "distance_km": np.round(rng.gamma(2.6, 3.4, n).clip(0.8, 45), 2),
        "package_kg": np.round(rng.gamma(1.8, 6.0, n).clip(0.3, 80), 1),
        "num_stops": rng.integers(1, 13, n),
        "traffic_index": np.round(rng.uniform(0, 10, n), 1),
        "weather": rng.choice(["clear", "rain", "fog"], n, p=[0.68, 0.24, 0.08]),
        "hour": rng.choice(np.arange(7, 21), n),
        "day_of_week": rng.integers(0, 7, n),
        "driver_exp_yrs": np.round(rng.gamma(2.2, 2.0, n).clip(0, 20), 1),
        "warehouse_queue": rng.poisson(14, n),
    })
    # vehicle choice depends on package weight (what a dispatcher would do); every vehicle type
    # appears across a range of weights so the model can learn each one's behaviour
    kg = df["package_kg"].to_numpy()
    df["vehicle_type"] = np.where(
        kg <= 10, rng.choice(["bike", "van", "truck"], n, p=[0.45, 0.40, 0.15]),
        np.where(kg <= 35, rng.choice(["van", "truck"], n, p=[0.70, 0.30]),
                 rng.choice(["truck", "van"], n, p=[0.70, 0.30])))
    df.loc[(df["distance_km"] > 18) & (df["vehicle_type"] == "bike"), "vehicle_type"] = "van"
    mu = expected_time(df)
    noise = rng.normal(0, 0.09, n) * mu + rng.normal(0, 2.0, n)        # multiplicative + additive
    df["delivery_time_min"] = np.round((mu + noise).clip(8), 1)
    return df


df = simulate(6000)
df.to_csv("data/deliveries.csv", index=False)
print(df.describe().T[["mean", "std", "min", "max"]].round(2))

TARGET = "delivery_time_min"
NUM = ["distance_km", "package_kg", "num_stops", "traffic_index", "hour",
       "day_of_week", "driver_exp_yrs", "warehouse_queue"]
CAT = ["weather", "vehicle_type"]
X, y = df[NUM + CAT], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def make_pipe(model, scale=False):
    num_tf = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("num", num_tf, NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", pre), ("model", model)])


def metrics(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)}


# ----------------------------------------------------------------------------
# 2. MODEL TRAINING + 5-FOLD CROSS-VALIDATION
# ----------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    "Linear Regression": make_pipe(LinearRegression(), scale=True),
    "Decision Tree": make_pipe(DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=SEED)),
    "Random Forest": make_pipe(RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                      n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": make_pipe(GradientBoostingRegressor(random_state=SEED)),
}
results = {"baseline": {}, "cv": {}, "test_default": {}}
# naive baseline: predict the training mean
results["baseline"] = metrics(y_test, np.full(len(y_test), y_train.mean()))

for name, pipe in candidates.items():
    cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=cv,
                               scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    results["cv"][name] = {"RMSE_mean": float(cv_rmse.mean()), "RMSE_std": float(cv_rmse.std()),
                           "R2_mean": float(cv_r2.mean())}
    pipe.fit(X_train, y_train)
    results["test_default"][name] = metrics(y_test, pipe.predict(X_test))
    print(f"{name:20s} CV RMSE {cv_rmse.mean():6.2f} +/- {cv_rmse.std():.2f} | "
          f"test RMSE {results['test_default'][name]['RMSE']:.2f}")

# ----------------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (Randomised search, 5-fold CV) on Gradient Boosting
# ----------------------------------------------------------------------------
param_dist = {
    "model__n_estimators": [200, 300, 400, 600],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_samples_leaf": [3, 5, 10, 20],
    "model__subsample": [0.7, 0.85, 1.0],
}
search = RandomizedSearchCV(make_pipe(GradientBoostingRegressor(random_state=SEED)),
                            param_dist, n_iter=25, cv=cv,
                            scoring="neg_root_mean_squared_error",
                            random_state=SEED, n_jobs=-1)
t0 = time.time()
search.fit(X_train, y_train)
best = search.best_estimator_
results["tuning"] = {"best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                     "best_cv_rmse": float(-search.best_score_),
                     "seconds": round(time.time() - t0, 1)}
pred_test = best.predict(X_test)
results["test_tuned"] = metrics(y_test, pred_test)
print("Tuned GB:", results["tuning"], results["test_tuned"])

# Residual diagnostics
resid = y_test.to_numpy() - pred_test
results["residuals"] = {"mean": float(resid.mean()), "std": float(resid.std()),
                        "within_5min_pct": float((np.abs(resid) <= 5).mean() * 100),
                        "within_10min_pct": float((np.abs(resid) <= 10).mean() * 100)}

# Error by segment (is the model biased anywhere?)
seg = X_test.copy()
seg["abs_err"] = np.abs(resid)
seg["peak"] = np.where(seg["hour"].isin([8, 9, 17, 18, 19]), "peak", "off-peak")
results["mae_by_vehicle"] = seg.groupby("vehicle_type")["abs_err"].mean().round(2).to_dict()
results["mae_by_weather"] = seg.groupby("weather")["abs_err"].mean().round(2).to_dict()
results["mae_by_peak"] = seg.groupby("peak")["abs_err"].mean().round(2).to_dict()

# Permutation importance on held-out data
pi = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=SEED,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False)
results["importance"] = imp.round(3).to_dict()

# ----------------------------------------------------------------------------
# 4. FIGURES FOR THE REPORT
# ----------------------------------------------------------------------------
# 4a distribution of target
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.hist(df[TARGET], bins=45, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8)
ax.set_xlabel("Delivery time (min)"); ax.set_ylabel("Number of deliveries")
ax.set_title("Distribution of simulated delivery times", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig1_target_hist.png"); plt.close(fig)

# 4b model comparison (test RMSE) - one series, sorted
names = list(results["test_default"].keys()) + ["GB (tuned)"]
rm = [results["test_default"][n]["RMSE"] for n in results["test_default"]] + [results["test_tuned"]["RMSE"]]
order = np.argsort(rm)[::-1]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
bars = ax.barh([names[i] for i in order], [rm[i] for i in order], color=BLUE, height=0.55)
bars[-1].set_color(ORANGE)
for b, i in zip(bars, order):
    ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f"{rm[i]:.2f}", va="center", color="#0b0b0b")
ax.set_xlabel("Test RMSE (minutes) - lower is better"); ax.grid(axis="y", visible=False)
ax.set_title("Model comparison on held-out test set", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig2_model_comparison.png"); plt.close(fig)

# 4c predicted vs actual
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(y_test, pred_test, s=8, alpha=0.35, color=BLUE, edgecolors="none")
lim = [0, float(max(y_test.max(), pred_test.max())) + 5]
ax.plot(lim, lim, color=ORANGE, lw=1.6, label="Perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual delivery time (min)"); ax.set_ylabel("Predicted delivery time (min)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Tuned Gradient Boosting: predicted vs actual", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig3_pred_vs_actual.png"); plt.close(fig)

# 4d permutation importance
fig, ax = plt.subplots(figsize=(6.2, 3.4))
imp_plot = imp.sort_values()
ax.barh(imp_plot.index, imp_plot.values, color=BLUE, height=0.6)
ax.set_xlabel("Increase in RMSE when feature is shuffled (min)"); ax.grid(axis="y", visible=False)
ax.set_title("Permutation feature importance", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig4_importance.png"); plt.close(fig)

# 4e residuals
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.hist(resid, bins=40, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8)
ax.axvline(0, color=ORANGE, lw=1.6)
ax.set_xlabel("Residual = actual - predicted (min)"); ax.set_ylabel("Count")
ax.set_title("Residual distribution (test set)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig5_residuals.png"); plt.close(fig)

# ----------------------------------------------------------------------------
# 5. OPTIMISATION BUILT ON THE MODEL
# ----------------------------------------------------------------------------
# 5a. Order-to-driver assignment (linear assignment / Hungarian algorithm)
#     Each of 60 orders must be given to exactly one of 60 available drivers
#     (each driver has a vehicle and an experience level). Cost = predicted minutes * cost/min.
#     Capacity: a vehicle cannot carry a parcel above its max weight.
N = 60
orders = df.sample(N, random_state=7).reset_index(drop=True)
drivers = pd.DataFrame({
    "vehicle_type": rng.choice(["bike", "van", "truck"], N, p=[0.30, 0.50, 0.20]),
    "driver_exp_yrs": np.round(rng.gamma(2.2, 2.0, N).clip(0, 20), 1),
})
BIG = 1e6


def build_matrices():
    pred_cost = np.zeros((N, N)); true_cost = np.zeros((N, N)); pred_min = np.zeros((N, N))
    for j, d in drivers.iterrows():
        trial = orders.copy()
        trial["vehicle_type"] = d["vehicle_type"]
        trial["driver_exp_yrs"] = d["driver_exp_yrs"]
        p = best.predict(trial[NUM + CAT])
        t = expected_time(trial)
        c = COST_PER_MIN[d["vehicle_type"]]
        infeasible = trial["package_kg"].to_numpy() > VEHICLE_MAX_KG[d["vehicle_type"]]
        # bikes also limited to short trips
        if d["vehicle_type"] == "bike":
            infeasible |= trial["distance_km"].to_numpy() > 18
        pred_min[:, j] = p
        pred_cost[:, j] = np.where(infeasible, BIG, p * c)
        true_cost[:, j] = np.where(infeasible, BIG, t * c)
    return pred_cost, true_cost, pred_min


pred_cost, true_cost, pred_min = build_matrices()
r, c = linear_sum_assignment(pred_cost)
opt_true_cost = float(true_cost[r, c].sum())
per_min = np.array([COST_PER_MIN[v] for v in drivers["vehicle_type"]])
opt_true_min = float((true_cost[r, c] / per_min[c]).sum())

# Baseline = rule-of-thumb dispatcher (no ML): process orders in sequence and give each the
# smallest vehicle that can physically carry it. See the greedy loop below.
results["assignment"] = {"orders": N}

# Baseline "smallest sufficient vehicle" greedy: orders are processed in sequence and each gets
# the first remaining driver whose vehicle can carry it (smallest vehicle class first), ignoring ML.
rank = {"bike": 0, "van": 1, "truck": 2}
free = set(range(N)); greedy_cost = 0.0
for i in range(N):
    feas = [j for j in free if true_cost[i, j] < BIG]
    j = min(feas, key=lambda j: (rank[drivers.loc[j, "vehicle_type"]], j))   # smallest sufficient vehicle
    greedy_cost += true_cost[i, j]; free.remove(j)
results["assignment"].update({
    "greedy_cost": round(float(greedy_cost), 2),
    "optimised_cost": round(opt_true_cost, 2),
    "saving_pct": round((greedy_cost - opt_true_cost) / greedy_cost * 100, 1),
    "optimised_total_minutes": round(opt_true_min, 1),
})

# 5b. Dispatch-time shifting for flexible parcels (departure window 07:00-20:00)
#     For 1,000 non-urgent deliveries with a flexible slot we pick the hour with the lowest predicted time.
flex = df.sample(1000, random_state=11).reset_index(drop=True)
hours = np.arange(7, 21)
pred_by_hour = np.column_stack([best.predict(flex.assign(hour=h)[NUM + CAT]) for h in hours])
true_by_hour = np.column_stack([expected_time(flex.assign(hour=h)) for h in hours])
chosen = pred_by_hour.argmin(axis=1)
cur_true = expected_time(flex)
opt_true = true_by_hour[np.arange(len(flex)), chosen]
results["dispatch"] = {
    "n_flexible": 1000,
    "avg_minutes_current": round(float(cur_true.mean()), 2),
    "avg_minutes_shifted": round(float(opt_true.mean()), 2),
    "saving_pct": round(float((cur_true.mean() - opt_true.mean()) / cur_true.mean() * 100), 1),
    "share_moved_out_of_peak_pct": round(float(np.isin(hours[chosen], [8, 9, 17, 18, 19]).mean() * 100), 1),
}
hr_mean = pd.Series(pred_by_hour.mean(axis=0), index=hours)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
cols = [ORANGE if h in [8, 9, 17, 18, 19] else BLUE for h in hours]
ax.bar([f"{h}:00" for h in hours], hr_mean.values, color=cols, width=0.62)
ax.set_ylabel("Mean predicted time (min)"); ax.set_xlabel("Dispatch hour")
ax.set_ylim(hr_mean.min() - 5, hr_mean.max() + 2)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); ax.grid(axis="x", visible=False)
ax.set_title("Predicted delivery time by dispatch hour (orange = peak)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig6_dispatch_hours.png"); plt.close(fig)
results["dispatch"]["peak_vs_offpeak_pct"] = round(float(
    (hr_mean[[8, 9, 17, 18, 19]].mean() / hr_mean[[h for h in hours if h not in [8, 9, 17, 18, 19]]].mean() - 1) * 100), 1)

# 5c. Fleet mix: LP minimising daily cost for a day's demand
#     Demand classes: light (<=10 kg), medium (10-120 kg), heavy (>120 kg is not present; use 25+ as 'bulky').
#     Decision x[c, v] = deliveries of class c done by vehicle v. Cost uses model-predicted mean minutes.
day = df.sample(900, random_state=21).reset_index(drop=True)
day["cls"] = np.where(day["package_kg"] <= 10, "light", np.where(day["package_kg"] <= 40, "medium", "heavy"))
demand = day["cls"].value_counts().to_dict()
classes = ["light", "medium", "heavy"]; vehs = ["bike", "van", "truck"]
mean_min = {}
for cl in classes:
    sub = day[day["cls"] == cl]
    for v in vehs:
        mean_min[(cl, v)] = float(best.predict(sub.assign(vehicle_type=v)[NUM + CAT]).mean())
allowed = {("light", "bike"), ("light", "van"), ("medium", "van"), ("medium", "truck"),
           ("heavy", "van"), ("heavy", "truck")}
idx = [(cl, v) for cl in classes for v in vehs if (cl, v) in allowed]
cost_vec = [mean_min[k] * COST_PER_MIN[k[1]] for k in idx]
A_eq, b_eq = [], []
for cl in classes:
    A_eq.append([1 if k[0] == cl else 0 for k in idx]); b_eq.append(demand.get(cl, 0))
# shift capacity (minutes) available per vehicle type: 50 bikes, 50 vans, 20 trucks x 480 min
cap = {"bike": 50 * 480, "van": 50 * 480, "truck": 20 * 480}
A_ub = [[mean_min[k] if k[1] == v else 0 for k in idx] for v in vehs]; b_ub = [cap[v] for v in vehs]
lp = linprog(cost_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
# baseline: default dispatch rule (light and medium parcels on vans, heavy parcels on trucks),
# i.e. what happens when nobody optimises the vehicle mix
base_alloc = {("light", "van"): demand.get("light", 0), ("medium", "van"): demand.get("medium", 0),
              ("heavy", "truck"): demand.get("heavy", 0)}
base_cost = sum(n * mean_min[k] * COST_PER_MIN[k[1]] for k, n in base_alloc.items())
base_van_minutes = sum(n * mean_min[k] for k, n in base_alloc.items() if k[1] == "van")
results["fleet_lp"] = {
    "status": lp.message, "demand": demand,
    "mean_minutes": {f"{k[0]}/{k[1]}": round(v, 1) for k, v in mean_min.items()},
    "optimal_cost": round(float(lp.fun), 2),
    "default_rule_cost": round(float(base_cost), 2),
    "default_rule_van_minutes": round(float(base_van_minutes), 0),
    "van_minutes_available": cap["van"],
    "saving_pct": round(float((base_cost - lp.fun) / base_cost * 100), 1),
    "allocation": {f"{k[0]}->{k[1]}": round(float(x), 1) for k, x in zip(idx, lp.x) if x > 0.5},
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(json.dumps(results, indent=2, default=str))
