from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer
from pointer_t5 import T5Pointer

# dataset = load_dataset("cnn_dailymail", "3.0.0")

# train_dataset = Dataset.from_dict(dataset["train"][:20000])
# validation_dataset = Dataset.from_dict(dataset["validation"][:2000])
# dataset = DatasetDict({"train":train_dataset, "validation":validation_dataset})

dataset = load_dataset("csv", data_files={"train":["../Dataset/Data/inshort_news_data-1.csv", "../Dataset/Data/inshort_news_data-2.csv", "../Dataset/Data/inshort_news_data-3.csv", "../Dataset/Data/inshort_news_data-4.csv", "../Dataset/Data/inshort_news_data-6.csv", "../Dataset/Data/inshort_news_data-7.csv"], "validation":["../Dataset/Data/inshort_news_data-5.csv"]}, delimiter=",", column_names=["news_headline", "news_article"], skiprows=1)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

max_input_length = 256
max_target_length = 128

prefix = "Summarize: "
def preprocess_examples(examples):
    articles = examples["news_article"]
    highlights = examples["news_headline"]
    
    inputs = [prefix + article for article in articles]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    labels = tokenizer(highlights, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    labels_with_ignore_index = []
    for labels_examples in labels:
        labels_examples = [label if label != tokenizer.pad_token_id else -100 for label in labels_examples]
        labels_with_ignore_index.append(labels_examples)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

dataset = dataset.map(preprocess_examples, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = T5Pointer.from_pretrained("t5-small")
args = TrainingArguments(output_dir="checkpoints/pointer_T5_cnn_loss", evaluation_strategy="steps", eval_steps=500, per_device_train_batch_size=8, per_device_eval_batch_size=8, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, num_train_epochs=3, learning_rate=5e-4, remove_unused_columns=False, label_names=["labels"])
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()

trainer.save_model(output_dir="checkpoints/pointer_T5_cnn_loss/best")
