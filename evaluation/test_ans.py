import json
from openai import OpenAI
import time
import concurrent.futures
import os

class LLMTesterAndSorter:
    def __init__(self, year, model_year):
        self.year = year
        self.model_year = model_year
        self.client = OpenAI(
            api_key='EMPTY',
            base_url=f'http://127.0.0.1:9002/v1',
        )
        self.model = self.client.models.list().data[0].id


    def process_conversation_group(self, conversation_group):
        group_results = []
        # messages = [{"role": "system", "content": self.system_promt}]
        messages = []
        for i, message in enumerate(conversation_group['messages']):
            if message['role'] == 'user':
                messages.append({"role": "user", "content": message['content']})
                try:
                    resp = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1024, temperature=0)
                    response = resp.choices[0].message.content
                    group_results.append({
                        "question": message['content'],
                        "answer": response
                    })
                    messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    print(f"Error occurred while processing question: {message['content']}, Error: {e}")
        return group_results

    def run_inference(self, datasets_ori):
        data = datasets_ori
        test_results = []
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {executor.submit(self.process_conversation_group, conversation_group): index for index, conversation_group in enumerate(data)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                test_results.insert(index, result)

        return test_results

    def reorder_results(self, datasets_ori, test_results):
        question_order = [messages[0]['content'] for messages in [item['messages'] for item in datasets_ori]]
        reordered_results = []
        for question in question_order:
            for section in test_results:
                if section[0]['question'] == question:
                    reordered_results.append(section)
                    break
        return reordered_results


    def run(self):
        ori_path = 'datasets/test_data/'  # labels
        ori_file_name = f'{self.year}-test.json'
        with open(ori_path + ori_file_name, 'r', encoding='utf-8') as file:
            datasets_ori = json.load(file)
            
        base_save_path = '/LLM_test/output_qwen2_5_lwf2/'  # 存储回答的路径
        os.makedirs(base_save_path, exist_ok=True)
        save_name = f'{self.model_year}_test_{self.year}.json'
        save_path = base_save_path + save_name
        
        test_results = self.run_inference(datasets_ori)
        reordered_results = self.reorder_results(datasets_ori, test_results)
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(reordered_results, file, indent=4, ensure_ascii=False)
        print("重排序后的结果已保存:", save_path)

if __name__ == "__main__":
    t1 = time.time()
    model_year = '2025'
    year = '2022'
    tester_and_sorter = LLMTesterAndSorter(year, model_year)
    tester_and_sorter.run()
    print(time.time() - t1)