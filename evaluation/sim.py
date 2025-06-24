import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_assistant_responses(data):
    responses = []
    for item in data:
        if isinstance(item, dict) and 'messages' in item:
            for message in item['messages']:
                if message['role'] == 'assistant':
                    responses.append(message['content'])
    return responses

def extract_assistant_responses2(data):
    responses = []
    for item in data:
        for message in item:
            responses.append(message['answer'])
    return responses

def compute_similarities(responses1, responses2, model, batch_size=32):
    similarities = []
    min_length = min(len(responses1), len(responses2))
    for i in range(0, min_length, batch_size):
        batch_responses1 = responses1[i:i+batch_size]
        batch_responses2 = responses2[i:i+batch_size]

        batch_emb1 = model.encode(batch_responses1, convert_to_tensor=True)
        batch_emb2 = model.encode(batch_responses2, convert_to_tensor=True)

        cos_sim_matrix = cosine_similarity(batch_emb1.cpu().numpy(), batch_emb2.cpu().numpy())
        batch_similarities = np.diag(cos_sim_matrix)
        
        similarities.extend(batch_similarities)

    return similarities

def main(file1, file2):
    data1 = load_json(file1)
    data2 = load_json(file2)

    resp_map1 = extract_assistant_responses(data1)
    resp_map2 = extract_assistant_responses2(data2)
    
    cache_folder = './MiniLM-L12-v2'
    print(cache_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=cache_folder，device=device)

    sims = compute_similarities(resp_map1, resp_map2, model)

    if sims:
        avg_sim = sum(sims) / len(sims)
        print(f"\nAverage similarity: {avg_sim:.4f}")
    else:
        print("No comparable assistant replies were found.")

if __name__ == "__main__":
    labels = 'datasets/test_data/2022-test.json'
    test_res = 'LLM_test/output2/2025_test_2022.json'
    main(labels, test_res)