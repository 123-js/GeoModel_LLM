# GeoModel-LLM: training and 3D inference

This minimal release contains one English notebook and the four supplied JSONL files. It retains model training, validation-loss monitoring during training, and complete 3D inference. Visualization, accuracy calculation, other evaluation metrics, standalone evaluation scripts, tests, result tables, and unused placeholder folders have been removed.

## Pretrained adapter and inference without retraining

The fine-tuned GeoModel-LLM **LoRA adapter** and tokenizer are available on Hugging Face:

- **Fine-tuned adapter:** [js123js/Qwen3_14_finetune](https://huggingface.co/js123js/Qwen3_14_finetune)
- **Base model used by this notebook:** [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)

The adapter contains the fine-tuned LoRA parameters, not a standalone copy of all base-model weights. Inference requires both the base model and adapter. The notebook explicitly loads the base model, attaches the adapter with PEFT, and loads the tokenizer from the adapter repository. Readers can use the published adapter without running the training section.

For synthetic test-set inference, use the following settings in the notebook's configuration cell:

```python
BASE_MODEL = "Qwen/Qwen3-14B"
ADAPTER_SOURCE = "js123js/Qwen3_14_finetune"

RUN_TRAINING = False
RUN_SYNTHETIC_INFERENCE = True
RUN_ENGINEERING_INFERENCE = False
```

Run Section 1 (configuration and prompt), Section 3 (inference model and helper functions), and Section 4 (synthetic 3D inference). Skip Section 2 (training). For the engineering example instead, set `RUN_SYNTHETIC_INFERENCE=False` and `RUN_ENGINEERING_INFERENCE=True`, then run Sections 1, 3, and 5. Predictions are saved as NPY files under `outputs/inference`.

These instructions require a suitable CUDA environment, the dependencies listed in `requirements.txt`, access to the Hugging Face repositories, and the corresponding local JSONL input file. Model loading and GPU inference were not executed during preparation of this release; the historical base-model revision and exact dependency versions remain to be confirmed.

## Files

| File | JSONL rows | Ordered complete models | Size (MiB) |
|---|---:|---:|---:|
| `finetune_data_3d_train.jsonl` | 18,000 | 450 | 46.36 |
| `finetune_data_3d_validation.jsonl` | 2,000 | 50 | 5.16 |
| `finetune_data_3d_test.jsonl` | 4,000 | 100 | 10.29 |
| `instruction_input_largerscale.jsonl` | 60 | 1 | 0.07 |

The first three files are the author's supplied training/validation/test splits. They are loaded directly, without another random split. Engineering input has instruction/input fields only. Model counts above describe complete ordered slice blocks; they do not prove cross-split statistical independence or recover historical IDs.

## Use

1. Open `3D_try_new_3.ipynb` in this folder on a suitable CUDA machine.
2. Run the configuration and prompt cells. All operation flags default to False.
3. For training, enable `RUN_TRAINING` and run Section 2. The local adapter is saved to `outputs/adapter`.
4. For synthetic 3D inference, enable `RUN_SYNTHETIC_INFERENCE`, run the inference setup/helpers, then Section 4.
5. For the engineering case, enable `RUN_ENGINEERING_INFERENCE`, run the inference setup/helpers, then Section 5.

The model identifiers can also be overridden through `GEOMODEL_BASE_MODEL` and `GEOMODEL_ADAPTER`. The base is loaded explicitly so a legacy private base path in adapter metadata is not used to discover it. After training, the same notebook session selects the newly saved local adapter.

`GEOMODEL_ROOT` overrides the repository root; `GEOMODEL_DATA_DIR` overrides the directory containing the four JSONL files. Outputs are saved under `outputs/inference`. Synthetic output numbering starts at 001 in the order of this supplied test file and does not imply the old model IDs. No reference cubes or metrics are written.

The main training settings are retained: 4-bit base loading, LoRA r=64/alpha=128/dropout=0, batch size 8, accumulation 8, two epochs, learning rate 2e-4, seed 3407, and response-only loss. Standard inference uses greedy decoding with batches of 40 and max_new_tokens=4000; engineering inference uses batches of 20 and max_new_tokens=8000. Both retain the 2048-token input cap and the original 4000-token model loading setting; their compatibility with the actual model/runtime must be checked on the target GPU environment.

RLE decoding retains the original repair behavior: short columns are padded with their last label, long columns are truncated, and empty/missing columns can remain zero. No smoothing or visualization is performed.

## Environment and validation status

`requirements.txt` lists dependencies without claiming recovered historical versions. Install CUDA-compatible PyTorch and Unsloth according to their official guides and pin the versions from the actual training environment for exact reproduction.

This edit only inspected JSONL contents and parsed Python syntax. The notebook, model loading, training, validation, and inference were not executed. All saved notebook outputs and execution counts are cleared. No API keys, experiment-tracker login, or private server paths are included. The original project notebook and the four supplied JSONL files are unchanged.

## Uploading the data

The training JSONL is 46.36 MiB, above GitHub's 25 MiB browser-upload limit but below its 100 MiB ordinary-Git per-file limit. To keep all four files in this GitHub repository, push with Git/GitHub Desktop rather than uploading the training file through the browser. Alternatively put the full JSONL data in a Hugging Face Dataset repository and add its actual URL here; download the files into this folder before use. JSONL files are not excluded by this release's .gitignore. No data have been uploaded by this edit.

Official limit reference: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

Code/data licenses and the original base-model revision remain for the author to confirm. This is the training/inference subset of the workflow, not a claim that the full synthetic generator or complete application toolset is included.
