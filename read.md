#### Get started
`clone the repository`

create a new conda env 

`conda create --name 'env_name' python=3.9`

`conda activate 'env_name' `

`pip install -r requirements.txt`


#### prepare training data

download data from `GSE262931`

To prepare training data for enformer or borzoi finutuning, run the following script:

`sh ./data/process_enformer_training_data.sh`  or

`sh ./data/process_borzoi_training_data.sh`


## options for preparing training data



| Option                     | Description                                  | Default    |
| -------------------------- | -------------------------------------------- | ---------- |
| `<fasta_file>`             | Input genome FASTA file                      | -          |
| `<targets_file>`           | Input targets / coverage file                | -          |
| `-b`                       | Set blacklist nucleotides to baseline value  | None       |
| `-c, --crop`               | Crop base pairs from each end                | 0          |
| `-d`                       | Round values to given decimals               | None       |
| `-f`                       | Generate cross fold split                    | None       |
| `-g`                       | Genome assembly gaps BED file                | None       |
| `-l`                       | Sequence length                              | 196608     |
| `--limit`                  | Limit to segments overlapping BED file       | None       |
| `--local`                  | Run jobs locally instead of SLURM            | False      |
| `-o`                       | Output directory                             | `data_out` |
| `-p`                       | Number of parallel processes                 | None       |
| `--peaks`                  | Create contigs only from peaks               | False      |
| `--restart`                | Continue progress from midpoint              | False      |
| `-s`                       | Down-sample the segments                     | 1.0        |
| `--st, --split_test`       | Exit after split                             | False      |
| `--stride, --stride_train` | Stride to advance train sequences            | 1.0        |
| `--stride_test`            | Stride to advance valid/test sequences       | 1.0        |
| `-t`                       | Proportion of data for testing               | 0.05       |
| `-w`                       | Sum pool width                               | 128        |
| `-v`                       | Proportion of data for validation            | 0.05       |



This will generate the train test and val data

To train the model, enformer or borzoi: run the follwoing script

` python ./model/utils/train_kidformer.py` or `python ./model/utils/train_kidzoi.py`

##### to get the scores from the model

create a directory ` mkdir -p resources/pretrained`
and   `mkdir -p resources/genome`
download the model first from the follwoing link  and save the pretrained models in `resources/pretrained`
save the genome in resources/genome

run the following script to get the scores from enformer finetuned


```
python ./model/utils/score_enformer_ft.py \                                
  --vcf_file ./test.vcf \
  --output_dir ./out \
  --target_length 16 \
  --sad_stats SAD,logSAD \
  --shifts="-1,0,1"
```

run the following script to get the scores from borzoi finetuned

```
python ./model/utils/score_borzoi_ft.py \                                
  --vcf_file ./test.vcf \
  --output_dir ./out \
  --target_length 16 \
  --sad_stats SAD,logSAD \
  --shifts="-1,0,1"
```

`target_length` is the number of bins to compute score from

