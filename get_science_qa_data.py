from datasets import load_dataset_builder

ds_builder = load_dataset_builder("derek-thomas/ScienceQA")

ds_builder.info.description