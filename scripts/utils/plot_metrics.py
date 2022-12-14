import json
from matplotlib import pyplot as plt

baseline_file = "/home/cxiao7/research/rl/RL4LMs/rl4lm_exps/rl4lm_experiment/val_split_metrics.jsonl"
nlpo_file = "/home/cxiao7/research/rl/RL4LMs/rl4lm_exps/t5_nlpo_on_supervised/val_split_metrics.jsonl"
ppo_file = "/home/cxiao7/research/rl/RL4LMs/rl4lm_exps/t5_ppo_on_supervised/val_split_metrics.jsonl"

def plot_metrics():

    # Plot baseline
    baseline_tag = "Baseline"
    with open(baseline_file, "r") as f:
        json_list = list(f)

    baseline_stats = []
    for json_str in json_list:
        res = json.loads(json_str)
        baseline_stats.append(res)

    # Plot ppo
    ppo_tag = "Supervised + PPO"
    with open(ppo_file, "r") as f:
        json_list = list(f)

    ppo_stats = []
    for json_str in json_list:
        res = json.loads(json_str)
        ppo_stats.append(res)

    # Plot nlpo
    nlpo_tag = "Supervised + NLPO"
    with open(nlpo_file, "r") as f:
        json_list = list(f)

    nlpo_stats = []
    for json_str in json_list:
        res = json.loads(json_str)
        nlpo_stats.append(res)

    epochs = [float(stat["epoch"]) for stat in baseline_stats[2:-1]]

    bleus_baseline = [float(stat["metrics"]["lexical/bleu"]) for stat in baseline_stats[2:-1]]
    bleus_ppo = [float(stat["metrics"]["lexical/bleu"]) for stat in ppo_stats[2:-1]]
    bleus_nlpo = [float(stat["metrics"]["lexical/bleu"]) for stat in nlpo_stats[2:-1]]
    meteors_baseline = [float(stat["metrics"]["lexical/meteor"]) for stat in baseline_stats[2:-1]]
    meteors_ppo = [float(stat["metrics"]["lexical/meteor"]) for stat in ppo_stats[2:-1]]
    meteors_nlpo = [float(stat["metrics"]["lexical/meteor"]) for stat in nlpo_stats[2:-1]]
    rogue1s_baseline = [float(stat["metrics"]["lexical/rouge_rouge1"]) for stat in baseline_stats[2:-1]]
    rogue1s_ppo = [float(stat["metrics"]["lexical/rouge_rouge1"]) for stat in ppo_stats[2:-1]]
    rogue1s_nlpo = [float(stat["metrics"]["lexical/rouge_rouge1"]) for stat in nlpo_stats[2:-1]]
    rogue2s_baseline = [float(stat["metrics"]["lexical/rouge_rouge2"]) for stat in baseline_stats[2:-1]]
    rogue2s_ppo = [float(stat["metrics"]["lexical/rouge_rouge2"]) for stat in ppo_stats[2:-1]]
    rogue2s_nlpo = [float(stat["metrics"]["lexical/rouge_rouge2"]) for stat in nlpo_stats[2:-1]]
    bertscores_baseline = [float(stat["metrics"]["semantic/bert_score"]) for stat in baseline_stats[2:-1]]
    bertscores_ppo = [float(stat["metrics"]["semantic/bert_score"]) for stat in ppo_stats[2:-1]]
    bertscores_nlpo = [float(stat["metrics"]["semantic/bert_score"]) for stat in nlpo_stats[2:-1]]
    entropies_baseline = [float(stat["metrics"]["diversity_metrics/entropy-3-nopunct"]) for stat in baseline_stats[2:-1]]
    entropies_ppo = [float(stat["metrics"]["diversity_metrics/entropy-3-nopunct"]) for stat in ppo_stats[2:-1]]
    entropies_nlpo = [float(stat["metrics"]["diversity_metrics/entropy-3-nopunct"]) for stat in nlpo_stats[2:-1]]

    fig, ax = plt.subplots(2, 3, figsize=(7 * 3, 6 * 2))
    ax[0][0].plot(epochs, bleus_baseline, label=baseline_tag)
    ax[0][0].plot(epochs, bleus_ppo, label=ppo_tag)
    ax[0][0].plot(epochs, bleus_nlpo, label=nlpo_tag)
    ax[0][0].set_xlabel("Epochs")
    ax[0][0].set_ylabel('BLEU Score')
    ax[0][0].set_title("BLEU Scores")
    ax[0][0].legend()

    ax[0][1].plot(epochs, meteors_baseline, label=baseline_tag)
    ax[0][1].plot(epochs, meteors_ppo, label=ppo_tag)
    ax[0][1].plot(epochs, meteors_nlpo, label=nlpo_tag)
    ax[0][1].set_xlabel("Epochs")
    ax[0][1].set_ylabel('METEOR Score')
    ax[0][1].set_title("METEOR Scores")
    ax[0][1].legend()

    ax[0][2].plot(epochs, rogue1s_baseline, label=baseline_tag)
    ax[0][2].plot(epochs, rogue1s_ppo, label=ppo_tag)
    ax[0][2].plot(epochs, rogue1s_nlpo, label=nlpo_tag)
    ax[0][2].set_xlabel("Epochs")
    ax[0][2].set_ylabel('ROGUE-1 Score')
    ax[0][2].set_title("ROGUE-1 Scores")
    ax[0][2].legend()

    ax[1][0].plot(epochs, rogue2s_baseline, label=baseline_tag)
    ax[1][0].plot(epochs, rogue2s_ppo, label=ppo_tag)
    ax[1][0].plot(epochs, rogue2s_nlpo, label=nlpo_tag)
    ax[1][0].set_xlabel("Epochs")
    ax[1][0].set_ylabel('ROGUE-2 Score')
    ax[1][0].set_title("ROGUE-2 Scores")
    ax[1][0].legend()

    ax[1][1].plot(epochs, bertscores_baseline, label=baseline_tag)
    ax[1][1].plot(epochs, bertscores_ppo, label=ppo_tag)
    ax[1][1].plot(epochs, bertscores_nlpo, label=nlpo_tag)
    ax[1][1].set_xlabel("Epochs")
    ax[1][1].set_ylabel('BERTScore')
    ax[1][1].set_title("BERTScores")
    ax[1][1].legend()

    ax[1][2].plot(epochs, entropies_baseline, label=baseline_tag)
    ax[1][2].plot(epochs, entropies_ppo, label=ppo_tag)
    ax[1][2].plot(epochs, entropies_nlpo, label=nlpo_tag)
    ax[1][2].set_xlabel("Epochs")
    ax[1][2].set_ylabel('Entropy')
    ax[1][2].set_title("Entropy")
    ax[1][2].legend()

    fig.savefig("/home/cxiao7/research/rl/RL4LMs/rl4lm_exps/rl4lm_experiment/stats/stats.png")

plot_metrics()