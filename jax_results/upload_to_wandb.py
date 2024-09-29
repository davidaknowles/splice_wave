from pathlib import Path
import pandas as pd

import json
import wandb
import os
os.environ["WANDB_SILENT"] = "true"

p = Path("context-RG_LM_small/")
resdirs = [ g for g in list(p.glob("*")) if g.is_dir() ]

res = []
for resdir in resdirs: 
    print(resdir)
    metrics_f = resdir / "metrics.tsv"
    if not metrics_f.exists():
        continue

    metrics = pd.read_csv(resdir / "metrics.tsv", sep = "\t")
    with open(resdir / 'config.json', 'r') as file:
        config = json.load(file)

    checkpoint_file = resdir / "checkpoint.pkl"
    if checkpoint_file.exists(): 
        wandb.init(project="context-RG_LM_small", name = resdir.name, config = config)
        for i in range(len(metrics)): 
            wandb.log({"train_loss": metrics.loc[i,"train_loss"], "test_loss": metrics.loc[i,"test_loss"]})
        wandb.save(checkpoint_file, base_path=resdir, policy="now")
        wandb.finish()
    
    config["test_loss"] = metrics["test_loss"].min()
    res.append(config)
