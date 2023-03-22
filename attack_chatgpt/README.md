## Downloading GLUE

We provide a convenience python script for downloading all GLUE data and standard splits.

```
python download_glue_data.py --data_dir glue_data --tasks all
```

After downloading GLUE, point ``PATH_PREFIX`` in  ``src/preprocess.py`` to the directory containing the data.
## Usage

All things can be used by running `main.py`:

For classification tasks:
- Use GPT API: `python main.py --train_tasks sst2  --eval_tasks sst2 --attack True  --begin_num 0 --test_num 2`


