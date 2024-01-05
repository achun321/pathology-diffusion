from transformers import CLIPProcessor, TrainingArguments, Trainer
from datasets import load_dataset


dataset = load_dataset(
            "imagefolder",
            data_dir="train", 
            cache_dir=None,
        )
tokenizer = CLIPProcessor.from_pretrained("vinid/plip").tokenizer
tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)
training_args = TrainingArguments(
    output_dir="./plip_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=0,
    weight_decay=0.01,
    logging_dir='./plip_logs',
    logging_steps=10,
)

