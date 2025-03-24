import pandas as pd
import os
from openai import OpenAI
import time
from tqdm import tqdm



# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def create_evaluation_prompt(question, ground_truth, answer):
    """Create a structured evaluation prompt."""
    prompt = f"""
    You are an evaluator tasked with assessing a student's answer to a question based on the following criteria:
    
    - Relevance: Does the response directly address the query? Does it stay on-topic and relate well to the question?
    - Accuracy: Is the information provided factually correct? Does the answer align with the provided ground truth?
    - Completeness: Does the response cover all necessary aspects of the question? Are there any important details left out?
    - Clarity: Is the response easy to understand? Is the language clear, and is the meaning conveyed effectively?
    - Conciseness: Is the response free from unnecessary information? Does it avoid verbosity while answering thoroughly?
    
    Here is the task:
    
    Question: {question}
    Ground Truth: {ground_truth}
    Answer provided by the student: {answer}
    
    For each of the above criteria, please provide a score from 0 to 10, where 0 means very poor, and 10 means excellent. Then, provide a final list of the scores in the same order as the criteria: [Relevance, Accuracy, Completeness, Clarity, Conciseness].
    
    The scores should be based on the following scale:
    
    - 0-3: Poor
    - 4-6: Average
    - 7-9: Good
    - 10: Excellent
    
    Please output the scores as a list of floating-point numbers, like this example: [8.5, 9.2, 7.8, 8.0, 6.5]. Be sure not to change the order or format of the output.
    
    Example output: [9.2, 6.5, 7.3, 9.2, 3.7]
    
    ### Evaluation:
    """
    return prompt

def evaluate_response(question, ground_truth, answer, client, max_retries=3, retry_delay=2):
    """Evaluate a single response with error handling and retries."""
    prompt = create_evaluation_prompt(question, ground_truth, answer)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an Evaluator who evaluates student answers based on the provided criteria."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Extract and parse the model's response
            evaluation_result = response.choices[0].message.content.strip()
            try:
                # Look for a list pattern in the response
                import re
                list_pattern = r'\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]'
                match = re.search(list_pattern, evaluation_result)
                if match:
                    scores = eval(match.group(0))
                    if len(scores) == 5:
                        return scores
                else:
                    raise ValueError("Incorrect number of scores")
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response: {evaluation_result}")
            if attempt == max_retries - 1:
                return [None, None, None, None, None]
        except Exception as e:
            print(f"API call error: {e}")
        if attempt == max_retries - 1:
            return [None, None, None, None, None]
        time.sleep(retry_delay)
    return [None, None, None, None, None]

def process_excel_file(file_path, client, batch_size=10):
    """Process the Excel file with batching and progress tracking."""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        required_columns = ['Question', 'Ground Truth', 'Model Generated']

        # Verify all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns in Excel file: {missing_columns}")
            return None

        # Initialize result columns
        result_columns = ['Relevancy', 'Accuracy', 'Completeness', 'Clarity', 'Conciseness']
        for col in result_columns:
            df[col] = None

        # Process in batches with progress bar
        total_rows = len(df)
        with tqdm(total=total_rows, desc="Evaluating responses") as pbar:
            for i in range(0, total_rows, batch_size):
                batch_end = min(i + batch_size, total_rows)
                for j in range(i, batch_end):
                    question = df.loc[j, 'Question']
                    ground_truth = df.loc[j, 'Ground Truth']
                    answer = df.loc[j, 'Model Generated']

                    # Handle NaN values
                    if pd.isna(question) or pd.isna(ground_truth) or pd.isna(answer):
                        df.loc[j, result_columns] = [None] * len(result_columns)
                        pbar.update(1)
                        continue

                    # Get evaluation scores
                    scores = evaluate_response(question, ground_truth, answer, client)

                    # Update the dataframe
                    df.loc[j, result_columns] = scores
                    pbar.update(1)

        # Save final results
        output_file = "evaluation_results_final.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Evaluation complete. Results saved to {output_file}")
        return df

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    # Input file path
    input_file = "question.xlsx"

    # Check if API key is available
    if not api_key:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    # Process the file
    result_df = process_excel_file(input_file, client)

    if result_df is not None:
        # Calculate and display average scores
        score_columns = ['Relevancy', 'Accuracy', 'Completeness', 'Clarity', 'Conciseness']
        avg_scores = {col: result_df[col].mean() for col in score_columns}

        print("\nAverage Scores:")
        for criterion, score in avg_scores.items():
            print(f"{criterion}: {score:.2f}")

        # Calculate overall average
        overall_avg = sum(avg_scores.values()) / len(avg_scores)
        print(f"\nOverall Average Score: {overall_avg:.2f}")

        threshold = 8.0
        if overall_avg >= threshold:
            print("Validation passed successfully!")
            exit(0)  # Exit with success code
        else:
            print("Validation failed.")
            exit(1)  # Exit with failure code

if __name__ == "__main__":
    main()
