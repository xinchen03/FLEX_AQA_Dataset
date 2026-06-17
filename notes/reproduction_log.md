# FLEX Reproduction Log

## 2026-06-18 — Corrections & Final Checks

### Corrections to 06-17 Notes

1. **Modality terminology**: The "4 modalities" are more accurately described as:
   - **3 modality types**: RGB (双视角 View-1 + View-2), 3D Skeleton, EMG
   - **4 input branches** in the model: I3D(View-1) + I3D(View-2) + STGCN(Skeleton) + EMG_Encoder(EMG)
   - When discussing lightweight models, clarify: "保留双模态 RGB+Skeleton" or "保留单视角 RGB+Skeleton"

2. **Seven_cls range**: `1-20` for FLEX (A01-A20), NOT `1-6`.
   - `0` = all 20 FLEX classes (full dataset)
   - `1-20` = single class (A01-A20 respectively)
   - The help text "1-6 for AQA-7" is a leftover from the older MTL-AQA benchmark
   - Verified in `utils/parser.py:13`: `choices=[0,1,...,20]`

3. **Lightweight Student plan is preliminary** — multiple options exist:
   - RGB-View1 + Skeleton (minimal, best for deployment)
   - RGB-View1 + RGB-View2 + Skeleton (drops only EMG, closer to baseline)
   - RGB-only + skeleton distill (most aggressive)
   - Skeleton-only + RGB teacher distill (ablative)
   - **Decision deferred until forward pass analysis and ablation experiments on real data.**

### Critical Check: model_rgb.pth — ✅ RESOLVED (2026-06-18)
- **Status: RESOLVED** — copied from `D:/projects/MTL_CoRe.pth` (179MB)
- Located at `./MTL-AQA/model_rgb.pth` as expected by config
- Structure: Full CoRe training checkpoint w/ keys `[base_model, regressor, optimizer, ...]`
  - `base_model` = I3D state_dict (344 keys), has `module.` prefix (from DataParallel)
- Source: CoRe project (yuxumin/CoRe)
- ⚠️ **Potential issue**: `load_pretrain()` in `models/Backbone.py:13` does `torch.load(ckpt)` then passes entire dict to `self.backbone.load_state_dict()`. But the checkpoint top-level is NOT an I3D state dict — it's `{base_model, regressor, ...}`. Need to extract `ckpt['base_model']` and strip `module.` prefix first. Compare with `resume_train()` in `builder.py:108-109` which correctly does this.
- ⚠️ Also: `load_pretrain` uses `weights_only=True` which fails on numpy scalars in older checkpoints. PyTorch 2.4.1 needs `weights_only=False` or `add_safe_globals`.
- **Fix needed before training**: patch `Backbone.py:load_pretrain` to extract `base_model` key and strip `module.`
- ⚠️ NOT tracked by git (covered by `*.pth` in .gitignore)

### Hardcoded Debug Paths Found
- `models/emg_encoder.py:77`: `/data/YH/FLEX-AQA/FLEX-AQA3/EMG/A01/199/EMG.csv`
- `models/stgcn_encoder.py:102`: `/data/YH/FLEX-AQA/FLEX-AQA3/Skeleton/Skeleton/A01/099/skeleton_points.csv`
- These appear to be debug/test artifacts; actual data loading uses dataset classes

### Environment Records Exported
- `notes/conda_list_py38.txt` — full conda package list
- `notes/pip_freeze_py38.txt` — pip package list
- Use to restore environment if broken later

---

## 2026-06-17 — Environment Setup + Pre-Data Inspection Complete

### Repository
- Local path: `D:/projects/FLEX_AQA_Dataset`
- Origin: `git@github.com:xinchen03/FLEX_AQA_Dataset.git`
- Upstream: `https://github.com/HaoYin116/FLEX_AQA_Dataset.git`
- Branch: `win-debug-env` (Windows debug fixes, committed & pushed)

### Environment
- Conda env: `py38`
- Python: 3.8.20
- PyTorch: 2.4.1 (CPU)
- OpenCV: 4.10.0 (conda)
- torchvideotransforms: installed from `hassony2/torch_videovision` GitHub (`--no-deps`)
- `python main.py -h`: ✅ SUCCESS

### Windows-Specific Fix
On Windows, cv2 has a DLL loading conflict when torchvision is imported before cv2.
Root cause: `torchvision` loads DLLs that conflict with conda `opencv`'s cv2 import.
Fix: import cv2 FIRST, before any torchvision-dependent modules.

**Modified in FLEX repo (committed to win-debug-env):**
- `tools/builder.py`: `torchvideotransforms` import moved BEFORE model imports (I3D, STGCN, etc.)

**Modified in site-packages (environment-only, not in repo):**
- `torchvideotransforms/functional.py`: `import cv2` before `import torch`
- `torchvideotransforms/video_transforms.py`: `from . import functional` before `import torchvision`

### Dataset Structure (from datasets/SevenPair.py)

**Directory layout expected by code:**
```
{FLEX-AQA}/
├── Split_4/
│   ├── split_4_train_list.mat    # training split annotations
│   └── split_4_test_list.mat     # test split annotations
├── A01/ ... A20/                 # 20 action classes
│   └── 001/ ... NNN/
│       ├── View-1/img_00001.JPG ... img_00103.JPG  # front view
│       └── View-2/img_00001.JPG ... img_00103.JPG  # side view
├── Skeleton/A01...A20/001...NNN/skeleton_points.csv  # 21 joints → NTU 25
└── EMG/A01...A20/001...NNN/EMG.csv                   # 4-ch sEMG → L/R ratio
```

**Key details:**
- 20 action classes: A01–A20 (mapped from Seven_cls arg)
- 4 modalities: RGB-View1 + RGB-View2 + 3D Skeleton + EMG
- 103 frames per sample (fixed)
- 4-fold cross-validation (Split_4)
- ALL modalities are ALWAYS loaded — no modality selection flag in dataset
- Skeleton: 21 joints from CSV → remapped to 25 joints (NTU format), normalized
- EMG: 4-channel CSV → per-class muscle selection → L/R contribution ratio (2 values)
- Pairwise ranking: compares sample_1 vs sample_2, predicts relative quality

### Training Logic

**Command:**
```bash
bash ./scripts/train.sh <GPU_ID> Seven <exp_name> --Seven_cls <1-6>
```
Where: `CUDA_VISIBLE_DEVICES=$1 python3 -u main.py --benchmark $2 --exp_name $3 ${@:4}`

**Training flow (from tools/runner.py):**
- Pair-based: input = (sample_A, sample_B), predict which is better
- Voter mechanism: during test, compares against `voter_number=3` training samples → ensemble
- Optimizer: Adam, base_lr=0.001, lr_factor=0.1 (I3D/STGCN lr = base_lr × lr_factor)
- Checkpoints saved to: `experiments/CoRe_RT/{benchmark}/{exp_name}/best.pth` (on best ρ)

### Model Architecture (from tools/builder.py)

```
Input: (video_View1, video_View2, Skeleton, EMG)
  │
  ├── I3D_backbone(video)        → video features (pretrained on Kinetics-400)
  ├── I3D_backbone(video2)       → video2 features
  ├── STGCNFeatureExtractor()    → skeleton features
  ├── EMG_Spectrogram_ResNet18() → EMG features
  │
  ├── FeatureCompressor(1536→1024) → fused features
  │
  └── RegressTree(in_channel=2×2306+1, hidden=256, depth=5)
      → pairwise score difference prediction
```

### Evaluation Metrics (from tools/runner.py)

| Metric | Code | Description | Target |
|--------|------|-------------|--------|
| Spearman's ρ | `stats.spearmanr(pred, true)` | Rank correlation | higher is better |
| L2 | `∑(pred-true)² / N` | Mean squared error | lower is better |
| RL2 | `∑((pred-true)/range)² / N` | Normalized L2 | lower is better |

Best checkpoint saved when ρ improves.

### Config Parameters (Seven_CoRe.yaml)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| data_root | `/root/FLEX-AQA/` | **MUST CHANGE** to actual data path |
| bs_train/bs_test | 1 | Batch size (very small) |
| workers | 4 | DataLoader workers (reduce for Windows) |
| max_epoch | 200 | Training epochs |
| frame_length | 103 | Frames per video |
| score_range | 100 | Max score for normalization |
| voter_number | 3 | Test-time ensemble voters |
| RT_depth | 5 | RegressTree depth |
| step_per_update | 2 | Gradient accumulation steps |
| pretrained_i3d_weight | `./MTL-AQA/model_rgb.pth` | I3D pretrained weights |

### Lightweight Model Plan — Key Insight

The current model ALWAYS loads all 4 modalities. To make a 2-modal student:
1. Remove `EMG_Spectrogram_ResNet18_Encoder` from model_builder
2. Adjust `FeatureCompressor` input dim (1536 → ?)
3. Adjust `RegressTree` in_channel (2×2306+1 → ?)
4. For distillation: train full 4-modal as teacher → distill to RGB+Pose student
5. For missing modality robustness: random modality dropout during training

### Pending
- [x] Fork & clone repo
- [x] Set up conda environment (py38)
- [x] Fix Windows torchvision/cv2 DLL conflict
- [x] Create .gitignore
- [x] Read configs and training scripts
- [x] Identify evaluation metrics (Spearman ρ, L2, RL2)
- [x] Analyze dataset structure (datasets/SevenPair.py)
- [x] Analyze model architecture (tools/builder.py)
- [x] Analyze training logic (tools/runner.py)
- [x] Apply for FLEX dataset access (all 4 subsets)
- [ ] Wait for dataset access approval
- [ ] Change data_root in Seven_CoRe.yaml to actual path
- [ ] Get GPU server access (lab server or AutoDL)
- [ ] Run single-class baseline (Seven_cls=1)
- [ ] Reproduce paper results on all 20 classes
- [ ] Design 2-modal lightweight student architecture
