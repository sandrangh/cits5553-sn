import pandas as pd
import os

def calculate_averages(input_files):
    averages = []

    for file in input_files:
        try:
            # Read CSV file
            df = pd.read_csv(file, sep=",")
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Check for required columns
            required_columns = ['meteor_score', 'bleu_score', 'rouge_l', 'precision', 'recall', 'f1_score', 'wmd_score', 'sbert_similarity', 'response_time']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Error: Missing columns in {file}: {', '.join(missing_columns)}")
                continue
            
            # Calculate average of specified columns
            avg_scores = {
                'meteor_score': round(df['meteor_score'].mean(), 2),
                'bleu_score': round(df['bleu_score'].mean(), 2),
                'rouge_l': round(df['rouge_l'].mean(), 2),
                'precision': round(df['precision'].mean(), 2),
                'recall': round(df['recall'].mean(), 2),
                'f1_score': round(df['f1_score'].mean(), 2),
                'wmd_score': round(df['wmd_score'].mean(), 2),
                'sbert_similarity': round(df['sbert_similarity'].mean(), 2),
                'response_time': round(df['response_time'].mean(), 2)
            }
            
            # Append averages along with filename
            avg_scores['file'] = os.path.basename(file)
            averages.append(avg_scores)

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Create dataframe from list of averages
    avg_df = pd.DataFrame(averages)

    return avg_df

def main():
    input_files = [
        'rag_base_val.csv',
        'rag_instruct_val.csv',
        'rag_val.csv'
    ]

    # Calculate averages
    averages_df = calculate_averages(input_files)

    # Output averages to a new CSV file
    averages_df.to_csv("model_comparisons.csv", index=False)

    print("Averages have been calculated and saved to 'model_comparisons.csv'.")

if __name__ == "__main__":
    main()