import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = r"D:\VED\Coding\ML\Data Science Project\Human Vs Robot\model\base_xgb_thresh_opt_model.pkl"

FEATURE_ORDER = [
    'num_bids', 'num_auct', 'num_merch_type', 'num_device_type',
    'num_time', 'num_ctry', 'num_ip', 'num_url', 'time_diff',
    'mean_diff', 'std_diff', 'min_diff', 'median_diff', 'max_diff',
    'iqr_diff', 'num_concurrent_bids', 'num_first_bid', 'num_last_bid',
    'num_firsthalf_bids', 'num_secondhalf_bids', 'percent_firsthalf_bids',
    'percent_secondhalf_bids', 'max_bids_in_auct', 'max_bids_per_device',
    'max_bids_per_device_per_auct', 'percent_concurrent_bids', 'bids_per_auct',
    'bids_per_device', 'bids_per_url', 'device_per_auct', 'ip_per_ctry',
    'percent_max_bids'
]

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
else:
    model = None

def run_prediction():
    if model is None:
        messagebox.showerror("Error", f"Model not found at:\n{MODEL_PATH}")
        return

    try:
        n_bids = float(entry_bids.get())
        n_auct = max(float(entry_auct.get()), 1.0)
        n_dev = max(float(entry_dev.get()), 1.0)
        n_ip = max(float(entry_ip.get()), 1.0)
        m_diff = float(entry_mdiff.get())
        c_ratio = float(entry_cratio.get())
    except ValueError:
        messagebox.showwarning("Input Error", "Please ensure all fields contain numbers.")
        return

    input_data = {feat: 0.0 for feat in FEATURE_ORDER}
    
    # Core variables
    input_data['num_bids'] = n_bids
    input_data['num_auct'] = n_auct
    input_data['num_device_type'] = n_dev
    input_data['num_ip'] = n_ip
    input_data['num_time'] = n_bids 
    input_data['num_merch_type'] = 1.0
    input_data['num_ctry'] = 1.0
    
    # Bots rotate URLs to avoid IP bans; scale URLs dynamically
    input_data['num_url'] = max(n_auct * 2.0, n_ip * 2.0) 
    
    # Temporal mapping
    input_data['num_firsthalf_bids'] = n_bids / 2.0
    input_data['num_secondhalf_bids'] = n_bids / 2.0
    input_data['percent_firsthalf_bids'] = 0.5
    input_data['percent_secondhalf_bids'] = 0.5
    input_data['num_first_bid'] = n_auct
    input_data['num_last_bid'] = n_auct

    input_data['mean_diff'] = m_diff
    input_data['median_diff'] = m_diff
    input_data['min_diff'] = 0.0 
    input_data['max_diff'] = m_diff * 3.0
    input_data['std_diff'] = m_diff * 0.5
    input_data['iqr_diff'] = m_diff * 0.5
    input_data['time_diff'] = m_diff * n_bids 
    
    input_data['num_concurrent_bids'] = n_bids * c_ratio
    input_data['percent_concurrent_bids'] = c_ratio
    
    # CLAMPING LOGIC: Cap extreme ratios so the model doesn't break on unrealistic inputs
    input_data['bids_per_auct'] = min(n_bids / n_auct, 15000)
    input_data['bids_per_device'] = min(n_bids / n_dev, 15000)
    input_data['bids_per_url'] = min(n_bids / input_data['num_url'], 15000)
    input_data['device_per_auct'] = n_dev / n_auct
    input_data['ip_per_ctry'] = n_ip / 1.0
    
    input_data['max_bids_in_auct'] = min(n_bids / n_auct, 15000)
    input_data['max_bids_per_device'] = min(n_bids / n_dev, 15000)
    input_data['max_bids_per_device_per_auct'] = min(input_data['max_bids_per_device'] / n_auct, 15000)
    input_data['percent_max_bids'] = input_data['max_bids_in_auct'] / n_bids if n_bids > 0 else 0.0

    df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
    prob = model.predict_proba(df)[0][1]
    
    threshold = 0.09
    is_bot = prob >= threshold

    result_text = "BOT DETECTED" if is_bot else "HUMAN VERIFIED"
    result_color = "red" if is_bot else "green"
    
    lbl_result_status.config(text=result_text, fg=result_color)
    lbl_result_prob.config(text=f"Confidence Score: {prob:.4f}")

# --- GUI Setup ---
root = tk.Tk()
root.title("PartiX: Bot Detection System")
root.geometry("380x450")
root.configure(padx=20, pady=20)

tk.Label(root, text="Behavioral Inputs", font=("Arial", 14, "bold")).pack(pady=(0, 10))

def create_input(label_text, default_val):
    frame = tk.Frame(root)
    frame.pack(fill="x", pady=4)
    tk.Label(frame, text=label_text, width=22, anchor="w").pack(side="left")
    entry = tk.Entry(frame, justify="center")
    entry.insert(0, str(default_val))
    entry.pack(side="right", fill="x", expand=True)
    return entry

entry_bids = create_input("Total Bids:", 5000)
entry_auct = create_input("Unique Auctions:", 10)
entry_dev = create_input("Unique Devices:", 1)
entry_ip = create_input("Unique IPs:", 15)
entry_mdiff = create_input("Mean Time Diff:", 0.01)
entry_cratio = create_input("Concurrent Bids Ratio (0-1):", 0.6)

tk.Button(root, text="Classify Bidder", command=run_prediction, bg="#0078D7", fg="white", font=("Arial", 11, "bold")).pack(pady=20, fill="x")

lbl_result_status = tk.Label(root, text="Awaiting Input...", font=("Arial", 16, "bold"))
lbl_result_status.pack(pady=(5, 0))
lbl_result_prob = tk.Label(root, text="", font=("Arial", 12))
lbl_result_prob.pack()
tk.Label(root, text="Decision Threshold: 0.09", font=("Arial", 9, "italic"), fg="gray").pack(side="bottom")

if __name__ == "__main__":
    if model is None:
        messagebox.showwarning("Model Warning", "Model could not be loaded. Please verify the path.")
    root.mainloop()



# Part 3: The Correct Test Inputs
# To show off your model successfully to your professor, use inputs that mathematically mimic real-world behavior inside the dataset's boundaries.
# Test Case 1: The Blatant Bot
# Bots use automated scripts, resulting in thousands of bids, high concurrency, and multiple rotating IP addresses to avoid platform bans.
# Total Bids: 8500
# Unique Auctions: 12
# Unique Devices: 1
# Unique IPs: 25 (Rotating IPs is a massive red flag for XGBoost)
# Mean Time Diff: 0.02
# Concurrent Bids Ratio: 0.75 (75% of bids placed at the exact same time)

# Test Case 2: The Normal Human
# Humans bid moderately. They only have one IP, they don't bid simultaneously, and they take time to think between bids.
# Total Bids: 14
# Unique Auctions: 3
# Unique Devices: 1
# Unique IPs: 1
# Mean Time Diff: 145.0 (Taking over two minutes between bids)
# Concurrent Bids Ratio: 0.0 (Zero simultaneous bids)