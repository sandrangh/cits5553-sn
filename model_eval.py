import csv
import string
import nltk
import pandas as pd
from gensim.downloader import load
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from opt_rag import MapAssistant

# Load pre-trained word vectors from Gensim (GloVe vectors)
word_vectors = load('glove-wiki-gigaword-100')

# Load pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def download_nltk_resources():
    """Ensure necessary NLTK resources are available."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading WordNet...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def preprocess(text):
    """Preprocess text for WMD by tokenizing, lowering the text, and removing punctuation."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [word for word in tokens if word in word_vectors]

def calculate_wmd(response, ground_truth):
    """Calculate Word Mover's Distance (WMD) between response and ground truth."""
    preprocessed_response = preprocess(response)
    preprocessed_ground_truth = preprocess(ground_truth)

    if not preprocessed_response or not preprocessed_ground_truth:
        return 2.0 # Assign a large distance for incomparable sentences

    try:
        distance = word_vectors.wmdistance(preprocessed_response, preprocessed_ground_truth)
        return distance if distance != float('inf') else 2.0  # Handle infinite distance
    except KeyError:
        return 2.0

def calculate_metrics(expected, generated):
    """Calculate precision, recall, and F1 score."""
    expected_set, generated_set = set(expected), set(generated)
    true_positives = len(expected_set & generated_set)
    false_positives = len(generated_set - expected_set)
    false_negatives = len(expected_set - generated_set)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1

def calculate_rouge_l_score(expected, generated):
    """Calculate ROUGE-L score."""
    rouge = Rouge()
    scores = rouge.get_scores(generated, expected, avg=True)
    return scores['rouge-l']['f']

def calculate_sentence_similarity(response, ground_truth):
    """Calculate cosine similarity between sentence embeddings."""
    response_embedding = sbert_model.encode([response])
    ground_truth_embedding = sbert_model.encode([ground_truth])
    return cosine_similarity(response_embedding, ground_truth_embedding)[0][0]

def validate_chatbot(validation_file, chatbot):
    """Validate the chatbot responses using multiple metrics."""
    results = []
    response_times = [] # Store all response times

    with open(validation_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                print(f"Skipping invalid row at line {row_num}: {row}")
                continue

            question, expected_response = row[0].strip(), row[1].strip()

            # Handle case where `process_chat()` returns response and response_time
            response_result = chatbot.process_chat(question)
            if isinstance(response_result, tuple):
                generated_response = response_result[0]
                response_time = response_result[1]
            else:
                generated_response = response_result
                response_time = None

            if response_time is not None:
                response_times.append(response_time)  # Collect response times

            # Use entire generated response for metrics
            meteor_score_value = meteor_score([expected_response.split()], generated_response.split())
            bleu_score_value = sentence_bleu([expected_response.split()], generated_response.split())
            precision, recall, f1 = calculate_metrics(expected_response.split(), generated_response.split())
            rouge_l = calculate_rouge_l_score(expected_response, generated_response)
            wmd_score = calculate_wmd(generated_response, expected_response)
            sbert_similarity = calculate_sentence_similarity(generated_response, expected_response)

            results.append({
                'question': question,
                'ground_truth': expected_response,
                'generated_response': generated_response,
                'meteor_score': round(meteor_score_value, 2),
                'bleu_score': round(bleu_score_value, 2),
                'rouge_l': round(rouge_l, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1_score': round(f1, 2),
                'wmd_score': round(wmd_score, 2),
                'sbert_similarity': round(sbert_similarity, 2),
                'response_time': round(response_time, 2) if response_time else None
            })

    # Calculate average response time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
    return results

def main():
    """Main function to run chatbot validation."""
    download_nltk_resources()

    validation_file = 'reduce_val_data.csv' # Path to validation CSV
    chatbot = MapAssistant() # Initialise chatbot

    results = validate_chatbot(validation_file, chatbot)

    # Save results to dataframe and export to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('opt_rag.csv', index=False)

    # Print average scores
    if not results_df.empty:
        avg_scores = {
            'avg_meteor': results_df['meteor_score'].mean(),
            'avg_bleu': results_df['bleu_score'].mean(),
            'avg_rouge_l': results_df['rouge_l'].mean(),
            'avg_precision': results_df['precision'].mean(),
            'avg_recall': results_df['recall'].mean(),
            'avg_f1': results_df['f1_score'].mean(),
            'avg_wmd': results_df['wmd_score'].mean(),
            'avg_sbert_similarity': results_df['sbert_similarity'].mean(),
            'avg_response_time': results_df['response_time'].mean()  # Add average response time
        }

        print("Average Scores:")
        for metric, avg in avg_scores.items():
            print(f"{metric.replace('avg_', 'Average ').title()}: {avg:.2f}")

if __name__ == "__main__":
    main()