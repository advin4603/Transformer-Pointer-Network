from pointer_t5 import T5Pointer
from transformers import AutoTokenizer

model = T5Pointer.from_pretrained("checkpoints/pointer_T5_cnn_loss/best")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

input_ids = tokenizer("""Summarize: The Boring Company shared a short clip on Twitter showing one of the underground stations that the company is building as part of its Las Vegas Convention Center (LVCC) loop. In September, Founder Elon Musk said the first operational tunnel under Vegas was almost complete. ""Tunnels under cities with self-driving electric cars will feel like warp drive,"" he had added.""",return_tensors="pt").input_ids

model.pointer.pgen_list = []
outputs = model.generate(input_ids=input_ids, max_length=150)
for pgen, token in zip(model.pointer.pgen_list, outputs[0]):
    print(f"{pgen:.2f} {tokenizer.decode(token)}")
    
print()
print("Article:")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
print()    
print("Summary:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
