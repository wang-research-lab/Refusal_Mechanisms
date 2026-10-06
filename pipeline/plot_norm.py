import json
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

# Base path
base_path = "pipeline/runs"

name_map = {
    "gemma-2-2b-it":                          "Gemma2-2B",
    "Qwen3-4B":                               "Qwen3-4B",
    "Phi-4-mini-instruct":                    "Phi4-Mini",
    "Mistral-Small-3.2-24B-Instruct-2506":    "Mistral-3.2-Small",
}

baseline_asr = {
    "gemma-2-2b-it":                       0.884765625,
    "Qwen3-4B":                            0.94140625,
    "Phi-4-mini-instruct":                 0.953125,
    "Mistral-Small-3.2-24B-Instruct-2506": 0.857421875
}

def get_hidden_size(model_name):
    if "gemma" in model_name: return 2304
    if "Phi" in model_name: return 3072
    if "Qwen" in model_name: return 2560
    if "Mistral" in model_name: return 5120
    return None

model_data = defaultdict(lambda: defaultdict(list))
bot_table_data = [] # To store (Model, t, k/d %, Relative ASR)

# Patterns for both Top and Bot
top_pattern = r'harmful_NormTop_layer\d+_t(\d+)_k(\d+)_evaluations\.json'
bot_pattern = r'harmful_NormBot_layer\d+_t(\d+)_k(\d+)_evaluations\.json'

for folder_name, display_name in name_map.items():
    norm_path = os.path.join(base_path, folder_name, "completions", "norm_sweep")
    h_size = get_hidden_size(folder_name)
    base_val = baseline_asr.get(folder_name)
    
    if not os.path.exists(norm_path) or h_size is None or base_val is None:
        continue
    
    for filename in os.listdir(norm_path):
        # 1. Process NormTop for the Graph
        top_match = re.search(top_pattern, filename)
        if top_match:
            t_val, k_val = int(top_match.group(1)), int(top_match.group(2))
            try:
                with open(os.path.join(norm_path, filename), 'r') as f:
                    eval_data = json.load(f)
                if 'llamaguard3_success_rate' in eval_data:
                    rel_asr = (eval_data['llamaguard3_success_rate'] / base_val) * 100
                    model_data[display_name][t_val] = ( (k_val/h_size)*100, rel_asr )
            except Exception: pass

        # 2. Process NormBot for the Table
        bot_match = re.search(bot_pattern, filename)
        if bot_match:
            t_val, k_val = int(bot_match.group(1)), int(bot_match.group(2))
            try:
                with open(os.path.join(norm_path, filename), 'r') as f:
                    eval_data = json.load(f)
                if 'llamaguard3_success_rate' in eval_data:
                    rel_asr = (eval_data['llamaguard3_success_rate'] / base_val) * 100
                    kd_pct = (k_val / h_size) * 100
                    bot_table_data.append([display_name, t_val, kd_pct, rel_asr])
            except Exception: pass

# --- Plotting Code (Same as yours) ---
plt.figure(figsize=(20, 7))
model_colors = {"Gemma2-2B": "#e63946", "Qwen3-4B": "#1d3557", "Phi4-Mini": "#2a9d8f", "Mistral-3.2-Small": "#f4a261"}
t_markers = {50: 'o', 70: 's', 80: 'D', 90: '*', 95: 'X', 97: 'P', 99: 'H'}

all_x = []
for model_name, t_dict in sorted(model_data.items()):
    sorted_ts = sorted(t_dict.keys())
    budgets = [t_dict[t][0] for t in sorted_ts]
    rel_asrs = [t_dict[t][1] for t in sorted_ts]
    all_x.extend(budgets)
    color = model_colors.get(model_name, "gray")
    plt.plot(budgets, rel_asrs, color=color, linewidth=4, alpha=0.3, zorder=1)
    
    for t in sorted_ts:
        b, r_a = t_dict[t]
        plt.scatter(b, r_a, color=color, marker=t_markers.get(t, 'o'), s=1000, edgecolors='black', linewidth=1.5, zorder=3)
        if "gemma" in model_name.lower():
            if t in [50, 70, 80, 90, 95, 97]:
                plt.annotate(f"{t}%", (b, r_a), textcoords="offset points", xytext=(0,20), ha='center', fontsize=21, fontweight='bold', color=color)

plt.title("Relative ASR vs. Dimension Tradeoff at Highest Performing Layer", fontsize=32, fontweight='bold', pad=40)
plt.xlabel("% of Total Hidden Dimensions", fontsize=24, fontweight='bold', labelpad=20)
plt.ylabel("% of Dense Steering\nASR Recovered", fontsize=24, fontweight='bold', labelpad=20)
plt.axhline(y=100, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.xticks(fontsize=22); plt.yticks(fontsize=22)
model_handles = [Line2D([0], [0], color=c, lw=8, label=m) for m, c in model_colors.items() if m in model_data]
plt.legend(handles=model_handles, title="Models", fontsize=20, title_fontsize=22, loc='lower right', frameon=True)
plt.xlim(0, max(all_x) * 1.05 if all_x else 10); plt.ylim(0, 110); plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'normtop_relative.png'), dpi=300)
plt.show()

# --- NormBot Table Output ---
print("\n" + "="*60)
print(f"{'NORMBOT RESULTS SUMMARY':^60}")
print("="*60)
print(f"{'Model':<20} | {'explained norm %':<5} | {'hidden dim %':<10} | {'Rel ASR %':<10}")
print("-" * 60)

bot_table_data.sort(key=lambda x: (x[0], x[1]))

for row in bot_table_data:
    print(f"{row[0]:<20} | {row[1]:<5} | {row[2]:>9.2f}% | {row[3]:>9.2f}%")
print("="*60)


# --- Rotation Control Collation ---
# Pattern: harmful_RotCtrl_layer{L}_t{T}_trial{i}_evaluations.json
rot_pattern = r'harmful_RotCtrl_layer(\d+)_t(\d+)_trial(\d+)_evaluations\.json'

rotation_summary = {}

for folder_name, display_name in name_map.items():
    rot_path  = os.path.join(base_path, folder_name, "completions", "rotation_control")
    norm_path = os.path.join(base_path, folder_name, "completions", "norm_sweep")
    h_size    = get_hidden_size(folder_name)
    base_val  = baseline_asr.get(folder_name)

    if not os.path.exists(rot_path) or h_size is None or base_val is None:
        continue

    # (layer, t) -> list of raw ASR values across trials
    trials_by_key = defaultdict(list)

    for filename in os.listdir(rot_path):
        m = re.search(rot_pattern, filename)
        if not m:
            continue
        layer_idx, t_val, trial_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            with open(os.path.join(rot_path, filename)) as f:
                asr = json.load(f)['llamaguard3_success_rate']
            trials_by_key[(layer_idx, t_val)].append(asr)
        except Exception:
            pass

    if not trials_by_key:
        continue

    model_entry = {}
    for (layer_idx, t_val), trial_asrs in sorted(trials_by_key.items()):
        # Look up the matching NormTop result at the same (layer, t)
        # We need k to reconstruct the filename; scan norm_sweep for it
        orig_asr = None
        k_val    = None
        if os.path.exists(norm_path):
            for fname in os.listdir(norm_path):
                nm = re.search(
                    rf'harmful_NormTop_layer{layer_idx}_t{t_val}_k(\d+)_evaluations\.json',
                    fname
                )
                if nm:
                    k_val = int(nm.group(1))
                    try:
                        with open(os.path.join(norm_path, fname)) as f:
                            orig_asr = json.load(f)['llamaguard3_success_rate']
                    except Exception:
                        pass
                    break

        n          = len(trial_asrs)
        mean_asr   = sum(trial_asrs) / n
        variance   = sum((x - mean_asr) ** 2 for x in trial_asrs) / n
        std_asr    = variance ** 0.5
        rel_mean   = (mean_asr  / base_val) * 100
        rel_orig   = (orig_asr  / base_val) * 100 if orig_asr is not None else None
        delta_rel  = (rel_mean - rel_orig)         if rel_orig  is not None else None
        k_frac     = (k_val    / h_size)           if k_val     is not None else None

        model_entry[f"layer{layer_idx}_t{t_val}"] = {
            "layer":            layer_idx,
            "threshold_pct":    t_val,
            "k":                k_val,
            "k_frac":           round(k_frac, 4)      if k_frac    is not None else None,
            "n_trials":         n,
            "trial_asrs":       trial_asrs,
            "rot_mean_asr":     round(mean_asr,  4),
            "rot_std_asr":      round(std_asr,   4),
            "rot_rel_mean_pct": round(rel_mean,  2),
            "orig_asr":         round(orig_asr,  4)   if orig_asr  is not None else None,
            "orig_rel_pct":     round(rel_orig,  2)   if rel_orig  is not None else None,
            "delta_rel_pct":    round(delta_rel, 2)   if delta_rel is not None else None,
        }

    rotation_summary[display_name] = model_entry

# Save to pipeline/runs/rotation_control_summary.json
out_path = os.path.join(base_path, "rotation_control_summary.json")
with open(out_path, "w") as f:
    json.dump(rotation_summary, f, indent=2)
print(f"\nRotation control summary written to {out_path}")

# Also print a compact table to stdout
print("\n" + "="*75)
print(f"{'ROTATION CONTROL SUMMARY':^75}")
print("="*75)
print(f"{'Model':<20} {'Layer':>5} {'t%':>4} {'k%':>6} "
      f"{'Orig Rel%':>10} {'Rot Rel%':>10} {'Std':>6} {'Delta':>7} {'N':>3}")
print("-"*75)
for display_name, entries in sorted(rotation_summary.items()):
    for key, e in sorted(entries.items(), key=lambda x: (x[1]['layer'], x[1]['threshold_pct'])):
        k_pct   = f"{e['k_frac']*100:.1f}"   if e['k_frac']    is not None else "  -  "
        orig    = f"{e['orig_rel_pct']:.1f}"  if e['orig_rel_pct']  is not None else "  -  "
        delta   = f"{e['delta_rel_pct']:+.1f}" if e['delta_rel_pct'] is not None else "  -  "
        print(f"{display_name:<20} {e['layer']:>5} {e['threshold_pct']:>4} {k_pct:>6} "
              f"{orig:>10} {e['rot_rel_mean_pct']:>10.1f} "
              f"{e['rot_std_asr']:>6.3f} {delta:>7} {e['n_trials']:>3}")
print("="*75)