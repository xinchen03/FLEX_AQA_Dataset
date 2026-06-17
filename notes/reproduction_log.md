# FLEX Reproduction Log

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
