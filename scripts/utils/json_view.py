import json

fname = "./rl4lm_exps/rl4lm_experiment/epoch_0_val_split_predictions.json"
with open(fname, "r") as f:
    res = json.load(f)
print(res)