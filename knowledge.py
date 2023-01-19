import requests
from transformers import pipeline


class Knowledge:
    
    def __init__(self, device="cpu") -> None:
        self.ner = pipeline(
            model="ckiplab/bert-base-chinese-ner", aggregation_strategy="simple", framework="pt", device=device)
        
    def get_entities(self, s):
        results = []
        prev_end = None
        prev_word = ""
        for entity in self.ner(s):
            start = entity["start"]
            end = entity["end"]
            word = entity["word"]
            if prev_end is not None and start != prev_end:
                results.append(prev_word.replace(" ", ""))
                prev_word = ""
            prev_word += word
            prev_end = end
        if len(prev_word):
            results.append(prev_word.replace(" ", ""))
        return results

    def get_knowledge_text(self, s, topk=1):
        knowledges = []
        entities = self.get_entities(s)
        for entity in entities:
            resp = requests.get("http://shuyantech.com/api/cndbpedia/ment2ent", params={"q": entity})
            knowledges.extend(resp.json()["ret"][:topk])
        return " ".join(knowledges)
