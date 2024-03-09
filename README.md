## Code for "Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector Bandits", (AISTATS 2024)

### Setup
```setup
conda env create --name paveba --file environment.yml
```

### Run as:
```bash
python simulations.py --experiment_file EXP_FILE_PATH
```
where `EXP_FILE_PATH` is a YAML file prepared according to the samples in `experiments/` folder.

## You can cite PaVeBa as below:
```
@inproceedings{
  karagozlu2024paveba,
  title={Learning the Pareto Set Under Incomplete Preferences: Pure Exploration in Vector Bandits},
  author={Karagözlü, Efe Mert and Yıldırım, Yaşar Cahit and Ararat, Çağın and Tekin, Cem},
  booktitle={Proc. 27th International Conference on Artificial Intelligence and Statistics},
  year={2024}
}
```
