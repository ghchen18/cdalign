> Code for AAAI 2021 paper "Lexically Constrained Neural Machine Translation with Explicit Alignment Guidance". [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17496)]

### Install and Data preprocess

The code is implemented on fairseq [v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0), follow the same steps to install and prepare the processed fairseq dataset with script [here](https://github.com/pytorch/fairseq/blob/v0.9.0/examples/translation/prepare-wmt14en2de.sh). You may need to process other datasets similarly. The python package `fastbpe` is also needed. 

```bash
git clone https://github.com/ghchen18/cdalign.git
cd cdalign
pip install --editable ./
```

### Step 1: Train vanilla transformer
See `scripts/run_train.sh`

### Step 2: Extract alignment using Att-Input method and process alignment data
See `scripts/extract_alignment.sh`

### Step 3: Train with EAM-Output method
See `scripts/run_train.sh`

### Step 4: Test on lexically constrained NMT task
See `scripts/run_test.sh`


### Citation

```
@inproceedings{chen2021lexically,
  title={Lexically Constrained Neural Machine Translation with Explicit Alignment Guidance},
  author={Guanhua, Chen and Yun, Chen and Victor O.K., Li},
  booktitle = {Proceedings of AAAI},
  year      = {2021},
  pages  = {12630--12638},
  volume={35},
  number={14},
}
```
