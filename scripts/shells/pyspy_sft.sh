P=$(pgrep -f "retryvla.train.sft" | head -1); echo PID=$P
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniforge3/envs/retryvla/bin/py-spy dump --pid $P 2>&1 | sed -n 4,16p
