# FLEX Reproduction Log

## 2026-06-17 — Environment Setup Complete

### Repository
- Local path: `D:/projects/FLEX_AQA_Dataset`
- Origin: `git@github.com:xinchen03/FLEX_AQA_Dataset.git`
- Upstream: `https://github.com/HaoYin116/FLEX_AQA_Dataset.git`
- Branch: `win-debug-env` (Windows debug fixes)

### Environment
- Conda env: `py38`
- Python: 3.8.20
- PyTorch: 2.4.1 (CPU)
- OpenCV: 4.10.0 (conda)
- torchvideotransforms: installed from `hassony2/torch_videovision` GitHub
- `python main.py -h`: ✅ SUCCESS

### Windows-Specific Fix
On Windows, cv2 has a DLL loading conflict when torchvision is imported before cv2.
The import order was adjusted:

**Modified in FLEX repo (committed):**
- `tools/builder.py`: torchvideotransforms import moved before model imports

**Modified in site-packages (environment-only, not in repo):**
- `torchvideotransforms/functional.py`: cv2 imported before torch
- `torchvideotransforms/video_transforms.py`: functional imported before torchvision

### Pre-Data Inspection

**Config (Seven_CoRe.yaml):**
- data_root: `/root/FLEX-AQA/` (MUST CHANGE for local/server path)
- bs_train: 1, bs_test: 1
- workers: 4 (may need reduction on Windows)
- max_epoch: 200
- frame_length: 103
- pretrained_i3d_weight: `./MTL-AQA/model_rgb.pth`
- score_range: 100, voter_number: 3

**Training command:**
```bash
bash ./scripts/train.sh <GPU_ID> Seven <exp_name> --Seven_cls <1-6>
```

**Metrics:**
- Spearman's Rho (ρ) — primary, higher is better
- L2 (MSE / N) — lower is better
- RL2 (normalized L2) — lower is better

**Modalities:**
- TODO: identify from datasets/SevenPair.py

### Pending
- [x] Fork & clone repo
- [x] Set up conda environment
- [x] Fix Windows import issues
- [x] Create .gitignore
- [x] Read configs and training scripts
- [x] Identify evaluation metrics
- [ ] Read datasets/SevenPair.py for modality selection
- [ ] Apply for FLEX dataset access
- [ ] Identify data path to change when data arrives
- [ ] Test single-class training (Seven_cls=1)
