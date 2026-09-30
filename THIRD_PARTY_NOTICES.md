# Third-party notices

`redimnet2_mlx/` is a port of code from the projects below, and `mlx_models/` holds weights
converted from the ReDimNet2 v1.0.0 release.

- **ReDimNet2** (https://github.com/PalabraAI/redimnet2): model code and pretrained weights.
  MIT License, Copyright (c) 2026 Palabra.ai. Full text in `LICENSE-redimnet2`.
- **ReDimNet** layers (https://github.com/IDRnD/redimnet), which ReDimNet2 builds on:
  MIT License, Copyright (c) 2024 ID R&D, Inc. The license text is the same as in
  `LICENSE-redimnet2`, with this copyright line.
- **WeSpeaker** ASTP pooling (https://github.com/wenet-e2e/wespeaker): Apache License 2.0,
  Copyright (c) 2021 Shuai Wang. The `ASTP` class in `redimnet2_mlx/model.py` is a modified
  MLX reimplementation. License: http://www.apache.org/licenses/LICENSE-2.0
- The transformer layer follows Hugging Face Transformers (Apache License 2.0), as noted upstream.
