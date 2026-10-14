# Все эксперименты подряд на Windows. Запускать из корня проекта с активированным venv.
# Запуск: powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
function Run { param([string]$cmd) Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "упало: $cmd" } }

Run "python -m unittest discover -s tests -t ."

# синтетика
Run "python -m src.synthetic --out data/synth --clients 3000"
Run "python -m src.prepare --raw data/synth --out prepared/synth"
Run "python -m src.gbm --data prepared/synth --name synth_gbm --runs runs_synth"
Run "python -m src.train finetune --data prepared/synth --model transformer --name synth_tr --runs runs_synth --epochs 8"
Run "python -m src.train finetune --data prepared/synth --model gru --name synth_gru --runs runs_synth --epochs 8"
Run "python -m src.report --runs runs_synth --out reports/synthetic.md"

# MBD-mini
Run "python -m src.inspect_mbd --download --root data/mbd"
Run "python -m src.prepare --raw data/mbd --out prepared/mbd --max-clients 40000 --last-months 6 --time-shift-hours 3"
Run "python -m src.gbm --data prepared/mbd --name gbm"
Run "python -m src.gbm --data prepared/mbd --name gbm_leak --leak"
foreach ($s in 0,1,2) {
  Run "python -m src.train finetune --data prepared/mbd --model gru --name gru_s$s --seed $s"
  Run "python -m src.train finetune --data prepared/mbd --model transformer --name tr_scratch_s$s --seed $s"
}
Run "python -m src.train pretrain --data prepared/mbd --name mlm --steps 6000"
foreach ($s in 0,1,2) {
  Run "python -m src.train finetune --data prepared/mbd --model transformer --init runs/mlm/encoder.pt --lr 1e-4 --name tr_mlm_s$s --seed $s"
}
foreach ($f in "0.05","0.2") {
  $tag = $f.Replace(".", "p")
  foreach ($s in 0,1) {
    Run "python -m src.train finetune --data prepared/mbd --model transformer --label-frac $f --name tr_scratch_f${tag}_s$s --seed $s"
    Run "python -m src.train finetune --data prepared/mbd --model transformer --init runs/mlm/encoder.pt --lr 1e-4 --label-frac $f --name tr_mlm_f${tag}_s$s --seed $s"
  }
}
Run "python -m src.train embed --data prepared/mbd --ckpt runs/mlm/encoder.pt --out runs/mlm/embeddings.npz"
Run "python -m src.gbm --data prepared/mbd --name gbm_emb_only --emb runs/mlm/embeddings.npz --emb-only"
Run "python -m src.gbm --data prepared/mbd --name gbm_plus_emb --emb runs/mlm/embeddings.npz"
Run "python -m src.report --runs runs --out reports/results.md"
Run "python -m src.report --compare runs/gbm runs/tr_scratch_s0 --data prepared/mbd"
Run "python -m src.report --compare runs/tr_scratch_s0 runs/tr_mlm_s0 --data prepared/mbd"
Run "python -m src.report --compare runs/gru_s0 runs/tr_mlm_s0 --data prepared/mbd"
Run "python -m src.report --compare runs/tr_scratch_f0p05_s0 runs/tr_mlm_f0p05_s0 --data prepared/mbd"
