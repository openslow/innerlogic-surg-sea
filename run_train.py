import runpy, sys
# Third continuation round: up to 15 more epochs (ep11-ep25) from the ep10
# best checkpoint, same recipe as the Modal cont rounds (fresh AdamW +
# fresh 15-epoch cosine, peak backbone 1e-5 / head 1e-4, early stop
# patience 3). Upload this file to the session, then launch detached:
# colab exec -s <session> -f launch_train.py
sys.argv = ["colab_stage_e_cont.py", "--mode", "train",
            "--epochs", "15", "--start-ep", "11", "--patience", "3",
            "--start-ckpt", "best_finetune_ep10.pt",
            "--start-epoch-expect", "10", "--start-best-val-expect", "0.9676",
            "--out-dir", "results_e_cont3"]
runpy.run_path("/content/ils/colab_stage_e_cont.py", run_name="__main__")
