Longitudinal Strain Estimation
---

Install
---

To download model's checkpoints, using `pip` and install `gdown` then run:
```bash
cd ckpts
sh download.sh
```

Serving
---
- Run server to have api: `PYTHONPATH='.' python echols/server.py`
- Run worker, which process the dicom file: `PYTHONPATH='.' python echols/worker.py`
- You can spawn as many workers as possible
- `./scripts/test.sh` for testing all pipeline

Using Docker
---

- `make compose-up` to build and run whole service
