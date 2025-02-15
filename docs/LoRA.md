# LLaVA (LoRA, Preview)

NOTE: This is a technical preview, and is not yet ready for production use. We are still running hyperparameter search for the LoRA model, and will release the final model soon.  If you'd like to contribute to this, please contact us.

You need latest code base for LoRA support (instructions [here](https://github.com/haotian-liu/LLaVA#upgrade-to-latest-code-base))

## Demo (Web UI)

Please execute each of the command below one by one (after the previous one has finished).  The commands are the same as launching other demos except for an additional `--model-base` flag to specify the base model to use. Please make sure the base model corresponds to the LoRA checkpoint that you are using.  For this technical preview, you need Vicuna v1.1 (7B) checkpoint (if you do not have that already, follow the instructions [here](https://github.com/lm-sys/FastChat#vicuna-weights)).

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-vicuna-7b-v1.1-lcs_558k-instruct_80k_3e-lora-preview-alpha --model-base /path/to/vicuna-v1.1
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000
```

## Training

Please see sample training scripts at [`./scripts/deepspeed`](./scripts/deepspeed).

We provide two sample DeepSpeed configs, [`zero3.json`](./scripts/deepspeed/zero3.json) is more like PyTorch FSDP, and [`zero3_offload.json`](./scripts/deepspeed/zero3_offload.json) can further save memory consumption by offloading parameters to CPU. `zero3.json` is usually faster than `zero3_offload.json` but requires more GPU memory, therefore, we recommend trying `zero3.json` first, and if you run out of GPU memory, try `zero3_offload.json`. You can also tweek the `per_device_train_batch_size` and `gradient_accumulation_steps` in the config to save memory, and just to make sure that `per_device_train_batch_size` and `gradient_accumulation_steps` remains the same.
