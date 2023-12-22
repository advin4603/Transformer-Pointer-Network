from pointer_t5 import T5Pointer
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore

model = T5Pointer.from_pretrained("checkpoints/pointer_t5_cnn_loss/best")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_dataset = Dataset.from_dict(dataset["test"][:100])
dataset = DatasetDict({"test": test_dataset})

rouge = ROUGEScore()
rougelp, rougelr, rougelf = 0, 0, 0
for i in tqdm(range(len(dataset["test"]))):
    article = dataset["test"][i]["article"][:512]
    highlights = dataset["test"][i]["highlights"]
    
    input_ids = tokenizer("Summarize: " + article, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    rouge_score = rouge(summary, highlights)
    rougelp += rouge_score["rougeL_precision"]
    rougelr += rouge_score["rougeL_recall"]
    rougelf += rouge_score["rougeL_fmeasure"]
    
    print(f"Article: {article}")
    print(f"Summary: {summary}")
    print(f"Highlights: {highlights}")
    
    break
    
rougelp /= len(dataset["test"])
rougelr /= len(dataset["test"])
rougelf /= len(dataset["test"])

print(f"ROUGE-L Precision: {rougelp:.2f}")
print(f"ROUGE-L Recall: {rougelr:.2f}")
print(f"ROUGE-L F1: {rougelf:.2f}")