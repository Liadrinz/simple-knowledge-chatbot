from transformers import BertTokenizer, BartForConditionalGeneration
from knowledge import Knowledge


class Chatbot:
    
    def __init__(self, use_kg=True, device="cpu"):
        if use_kg:
            self.knowledge = Knowledge(device)
        self.tokenizer = BertTokenizer.from_pretrained("HIT-TMG/dialogue-bart-large-chinese")
        self.model = BartForConditionalGeneration.from_pretrained("HIT-TMG/dialogue-bart-large-chinese")
        self.model.to(device)
        self.use_kg = use_kg
        self.device = device

    def chat(self, s):
        if self.use_kg:
            k = self.knowledge.get_knowledge_text(s)
            input_ids = self.tokenizer("对话历史：" + s, "知识：" + k, return_tensors="pt", truncation=True, max_length=512).input_ids
        else:
            input_ids = self.tokenizer("对话历史：" + s, return_tensors="pt", truncation=True, max_length=512).input_ids
        outputs = self.model.generate(input_ids.to(self.device), max_length=32, top_p=0.5, do_sample=True)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text.replace(" ", "")
        return output_text
