import pickle
import re
import json
import os
from openai import OpenAI

class QueryHelper:
    prompt_dict = {
        # medium
        'cora': 'opening text of machine learning papers',
        'citeseer': 'description or opening text of scientific publications',
        # large
        'wikics': 'entry and content of wikipedia',
        'bookhis': 'description or title of the book',
        'bookchild': 'description or title of the child literature',
        'sportsfit': 'the title of a good in sports & fitness',
        # small
        'cornell': 'webpage text',
        'texas': 'webpage text',
        'wisconsin': 'webpage text',
        'washington': 'webpage text',
    }
    network_dict = {'cora': 'citation network', 'citeseer': 'citation network', 
                    'wikics': 'knowledge graph', 'bookhis': 'e-commerce network',
                    'bookchild': 'e-commerce network', 'sportsfit': 'e-commerce network',
                    'cornell': 'university webpage network', 'texas': 'university webpage network', 'wisconsin': 'university webpage network', 'washington': 'university webpage network'}
    item_name = {'cora': 'Paper', 'citeseer': 'Paper', 'wikics': 'Entry', 'bookhis': 'Book', 'bookchild': 'Book', 'sportsfit': 'Product', 'cornell': 'Webpage', 'texas': 'Webpage', 'wisconsin': 'Webpage', 'washington': 'Webpage'}
    item_name_ploral = {'cora': 'papers', 'citeseer': 'papers', 'pubmed': 'papers',
                'ogbn-arxiv': 'papers', 'wikics': 'entries', 'bookhis': 'books',
                'bookchild': 'books', 'sportsfit': 'products', 'cornell': 'webpages',
                'texas': 'webpages', 'wisconsin': 'webpages', 'washington': 'webpages'}

    def __init__(self, args, dataset, llm_name, labels):
        self.args = args
        self.dataset = dataset
        self.dataset_description = self.prompt_dict[dataset]
        self.item_name_text = self.item_name[dataset]
        self.item_ploral_name_text = self.item_name_ploral[dataset]
        self.network_text = self.network_dict[dataset]
        self.labels = labels
        self.llm_name = llm_name
        if os.path.exists(args.cache_file) and not args.disable_cache:
            with open(args.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}
        self.query_log = []

    def query(self, text_list):
        prompt = self.prepare_prompt(text_list)
        response = self.generate(prompt)

        final_answer_match = re.findall(r'Final Answer:\s*(.+)', response)
        if final_answer_match:
            content = final_answer_match[-1].strip()
        else:
            content = response
        answer = self.catch_answer(content)
        return answer
    
    def prepare_prompt(self, text_list):
        init_instruct_1 = f'We have {self.dataset_description} in a {self.network_text} from the following {len(self.labels)} categories: {self.labels}'
        init_instruct_2 = f'Below are texts from {len(text_list)} {self.item_ploral_name_text}.'
        content = []
        for i, text in enumerate(text_list):
            content.append(f'[{self.item_name_text} {i + 1}]\n{text}')
        content = '\n\n'.join(content)
        init_instruct_3 = f'Please tell me the main category that most of the {self.item_ploral_name_text} belong to. Think carefully and then provide the final answer in the last line, using the format: "Final Answer: <category>".'
        messages = [
            {"role": "system",
            "content": "You are a chatbot who is an expert in text classification",},
            {"role": "user", "content": init_instruct_1+'\n'+init_instruct_2+'\n'+content+'\n'+init_instruct_3},
        ]
        return messages
    
    def generate(self, prompt):
        prompt_key = json.dumps(prompt, sort_keys=True)
        if prompt_key in self.cache:
            return self.cache[prompt_key]
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.llm_name,
            messages=prompt
        )
        response = completion.choices[0].message.content
        self.cache[prompt_key] = response
        return response
    
    def catch_answer(self, content):
        answer = -1
        for label_idx, label in enumerate(self.labels):
            label = re.sub(r'\(.*?\)', '', label)
            is_present = bool(re.search(label, content, re.IGNORECASE))
            if is_present:
                if answer == -1:
                    answer = label_idx
                else:
                    answer = -2
                    break
        return answer

