# HEAR-SAVI
This module extracts an audio subnetwork from the upstream semantic audio-visual navigation
([savi](https://github.com/marl/sound-spaces/tree/main/ss_baselines/savi)) model.

The embeddings implement the [common API](https://hearbenchmark.com/hear-api.html)
from HEAR Benchmark.

## Requirements
1. Install habitat-lab v0.2.1 and habitat-sim from commit 80f8e31140.
2. `pip install hear-savi`

````
@inproceedings{chen20soundspaces,
  title     =     {SoundSpaces: Audio-Visual Navigaton in 3D Environments},
  author    =     {Changan Chen and Unnat Jain and Carl Schissler and Sebastia Vicenc Amengual Gari and Ziad Al-Halah and Vamsi Krishna Ithapu and Philip Robinson and Kristen Grauman},
  booktitle =     {ECCV},
  year      =     {2020}
}
```