from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("K024/mt5-zh-ja-en-trimmed")

model = AutoModelForSeq2SeqLM.from_pretrained("K024/mt5-zh-ja-en-trimmed")