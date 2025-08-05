import json
import requests
from openai import OpenAI
import re
import time
import tiktoken
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import List, Dict
def extract_first_uppercase(s):
            match = re.search(r'[A-D]', s)
            return match.group(0) if match else None
global True_count
True_count = 0
API_KEY="" 
BASE_URL=""
MODEL_NAME = ""

class ModelEvaluator:
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.token_statistics = pd.DataFrame(
            columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
            dtype=object  
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def make_request_with_retries(self, function, max_retries=3, wait_time=10, *args, **kwargs):
        for attempt in range(max_retries):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                print(f"Error encountered: {e}. Retrying {attempt + 1}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
        print("Max retries reached. Returning failure.")
        return None
    
    def save_statistics(self, filename: str = ""):
        """Save token statistics to a CSV file. Append if the file exists."""
        if os.path.exists(filename):
            self.token_statistics.to_csv(filename, mode='a', header=False, index=False)
        else:
            self.token_statistics.to_csv(filename, mode='w', header=True, index=False)
        print(f"Token statistics saved to {filename}")


    def generate_feedback_prompt(self, question, generated_answer):
        return f"""
You are a verification assistant, focusing on helping to check the answers to elementary school math problems. Your goal is to quickly determine whether the answer is correct through simple logic and intuition, without over-analyzing the details in the problem.
Tasks:
    Start from the final answer provided in the generated answer.
    Trace back step by step, quickly verifying whether each reasoning step conforms to the main conditions of the problem.
    Focus on numerical calculations and basic logic, without over-analyzing possible inaccuracies in the problem.
    For each step A→B, simply verify the following:
        Conflict with problem conditions:
        Check whether B obviously violates the main conditions of the problem. If B contradicts the main information described in the problem, classify this step as incorrect.
        Basic logical consistency:
        Ensure that B is reasonably deduced from A, without obvious logical jumps or errors.
        Numerical calculation correctness:
        Quickly check whether the numerical calculations are correct, without having to delve too deeply into the details.
    If all steps and conditions are basically met, output: "The answer is correct."
    If obvious numerical errors, logical errors, or conflicts with the main conditions of the problem are found, classify the answer as incorrect.
    Clearly point out which step has a problem.
Guidelines:
    Trust your intuition: If the intermediate result seems reasonable, there is no need to question it excessively.
    End with "The answer is correct." or "The answer is incorrect."

Question:
{question}
Generated Answer:
{generated_answer}
"""

    def verify_answer(self, question: str, answer: str, origin_answer) -> str:
        """Verify the generated answer."""
        feedback_prompt = self.generate_feedback_prompt(question, answer)
        input_token_count = len(self.tokenizer.encode(feedback_prompt))
        try:
            data = {
                'model': self.model_name,
                'messages': [
                    {'role': 'user', 'content': feedback_prompt}
                ],
                'temperature': 0.5,
                'max_tokens': 4000
            }
            response = self.make_request_with_retries(self.client.chat.completions.create, 3, 60, **data)
            forcast = "true"
            feedback = response.choices[0].message.content
            model_number = self.extract_answer(text=answer)
            if model_number:
                model_number = model_number
            else:
                model_number = self.extract_final_answer(generated_answer=answer)
            expected_number = origin_answer
            evaluation_result = self.evaluate_answer_match(model_number, expected_number)
            is_correct = ""
            if "The answer is correct." in evaluation_result:
                is_correct = True
            else:
                is_correct = False
            if "The answer is correct" in feedback or "the answer is correct" in feedback or ("correct" in feedback and "incorrect" not in feedback):
                if is_correct:
                    forcast = f"Feedback indicates the generated answer, {model_number}, is correct. The standard answer is {expected_number}, which matches the generated answer. Prediction: Correct."
                else:
                    forcast = f"Feedback indicates the generated answer, {model_number}, is correct. The standard answer is {expected_number}, which differs from the generated answer. Prediction: Incorrect."
            else:
                if is_correct:
                    forcast = f"Feedback indicates the generated answer, {model_number}, is incorrect. The standard answer is {expected_number}, which matches the generated answer. Prediction: Incorrect."
                else:
                    forcast = f"Feedback indicates the generated answer, {model_number}, is incorrect. The standard answer is {expected_number}, which differs from the generated answer. Prediction: Correct."

            output_token_count = len(self.tokenizer.encode(feedback))
            self.token_statistics = self.token_statistics.append({
                "Question": str(question),
                "Input": str(feedback_prompt),
                "Input Tokens": input_token_count,
                "Output Tokens": output_token_count,
                "Output": str(feedback),
                "Type": "Answer->Feedback",
                "Experiment": "Reverse",
                "Forecast": forcast

            }, ignore_index=True)
            return feedback
        except Exception as e:
            print(f"Verification failed: {e}")
            return ""

    def extract_answer(self, text, prefixes=("The answer is", "the answer is")):
        """
        Extract the content following the specified prefixes in the text.
        Handles both 'The answer is' and 'the answer is', including newlines.
        """
        try:
            # Create a regex pattern to match the prefixes (case-insensitive) and capture everything after them
            pattern = rf"(?i)\b(?:{'|'.join(map(re.escape, prefixes))})\b\s*[:：]?\s*(.*)"
            match = re.search(pattern, text, re.DOTALL)  # Use re.DOTALL to include newlines in the match
            if match:
                return match.group(1).strip()  # Return the content after the prefix, stripped of extra spaces
            else:
                return None
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None
    
    def extract_final_answer(self, generated_answer, max_tokens=4000):
        # Add the fixed prompt to guide the model in extracting the final answer
        messages = [
            {"role": "user",
             "content": "Here is a generated answer. Please extract the final answer mentioned in the text . If no such statement exists, please respond with 'No answer found.'."},
            {"role": "assistant",
             "content": "Sure! Please provide the generated answer, and I will extract the final answer for you."},
            {"role": "user",
             "content": f"Generated Answer:\n{generated_answer}"}
        ]

        # Prepare the API request payload
        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens
        }

        # Make the API call
        response = self.client.chat.completions.create(**data)
        extracted_answer = response.choices[0].message.content
        return extracted_answer
    
    def generate_new_answer(self, question: str, history, generated_answer="", origin_answer="") -> str:
        history_prompt = "\n\n".join([
            f"Answer {i + 1}: {entry['answer']}\nFeedback {i + 1}: {entry['feedback']}"
            for i, entry in enumerate(history)
        ])

        correction_prompt = f"""
        You are an expert problem solver with a strong ability to refine and correct answers based on external feedback. Your task is to determine the most accurate and corrected answer for the following question by strictly following the feedback provided in the history.
Question:

{question}
Answer History & Feedback:

{history_prompt}
### Step 1: Root Cause Analysis (RCA)
Based on the provided feedback, first, perform a diagnosis of the original error.
1.  **Summarize Failure:** Briefly summarize why the previous answer was wrong according to the feedback.
2.  **Classify Flaw Type:** Based on the failure, classify the core flaw. Was it:
    -   **(A) Calculation Error:** The overall strategy and logic were correct, but a mathematical or execution step was performed incorrectly.
    -   **(B) Reasoning Flaw:** The calculations were correct, but the entire strategy was based on a misunderstanding of the question, a logical fallacy, or a flawed assumption.

### Step 2: Formulate a Corrective Action Plan
Based on your diagnosis in Step 1, you must now create a plan.
-   **If the flaw was (A) Calculation Error,** your plan is: "I will re-execute the original strategy, paying close attention to the specific step that was calculated incorrectly."
-   **If the flaw was (B) Reasoning Flaw,** your plan must be: "The original approach was fundamentally flawed. I will now adopt an alternative strategy. My new strategy is to [Describe your new approach here. For example: 're-read the question to identify overlooked constraints,' or 'break the problem down differently,' or 'work backward from a known fact']."

### Step 3: Execute the Plan
Now, execute the corrective action plan you just formulated. Provide a clear, step-by-step derivation of the new, corrected answer.

### Step 4: Final Answer
Conclude with your final, corrected answer.
Please end with ' the answer is X', Where 'X' is the correct option you think from A，B，C，D.
                                """

        input_token_count = len(self.tokenizer.encode(correction_prompt))
        try:
            data = {
                'model': self.model_name,
                'messages': [
                    {'role': 'user', 'content': correction_prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 4000
            }
            response = self.make_request_with_retries(self.client.chat.completions.create, 3, 60, **data)
            correction_answer = response.choices[0].message.content
            input_token_count = len(self.tokenizer.encode(correction_prompt))
            output_token_count = len(self.tokenizer.encode(correction_answer))
            model_number = self.extract_answer(correction_answer)
            if model_number:
                model_number = model_number
            else:
                model_number = self.extract_final_answer(correction_answer)
            expected_number = origin_answer
            evaluation_result = self.evaluate_answer_match(model_number, expected_number)
            is_correct = ""
            forecast = ""

            expected_number2 = model_number
            model_number2 = self.extract_answer(generated_answer)
            if model_number2:
                model_number2 = model_number2
            else:
                model_number2 = self.extract_final_answer(generated_answer)
            genggai_result = self.evaluate_answer_match(model_number, model_number2)
            genggaiqian_result = self.evaluate_answer_match(model_number2, expected_number)

            if "The answer is correct." in evaluation_result:
                # This block handles cases where the final answer is considered correct.
                if "The answer is correct." in genggai_result:
                    # Case: The answer after modification is correct.
                    if "The answer is correct." in genggaiqian_result:
                        # Case: The answer was correct before and after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: No change detected, Before: Correct, After: Correct")
                    else:
                        # Case: The answer was incorrect before but correct after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: No change detected, Before: Incorrect, After: Correct")
                else:
                    # Case: The answer after modification is considered incorrect.
                    if "The answer is correct." in genggaiqian_result:
                        # Case: The answer was correct before but incorrect after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: Change detected, Before: Correct, After: Correct")
                    else:
                        # Case: The answer was incorrect before and after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: Change detected, Before: Incorrect, After: Correct")
            else:
                # This block handles cases where the final answer is considered incorrect.
                if "The answer is correct." in genggai_result:
                    # Case: The answer after modification is correct.
                    if "The answer is correct." in genggaiqian_result:
                        # Case: The answer was correct before and after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: No change detected, Before: Correct, After: Incorrect")
                    else:
                        # Case: The answer was incorrect before but correct after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: No change detected, Before: Incorrect, After: Incorrect")
                else:
                    # Case: The answer after modification is considered incorrect.
                    if "The answer is correct." in genggaiqian_result:
                        # Case: The answer was correct before but incorrect after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: Change detected, Before: Correct, After: Incorrect")
                    else:
                        # Case: The answer was incorrect before and after modification.
                        summary = (f"Original Answer: {generated_answer}, "
                                   f"Modified Answer: {model_number}, "
                                   f"Standard Answer: {origin_answer}, "
                                   f"Status: Change detected, Before: Incorrect, After: Incorrect")
            forecast = summary

            self.token_statistics = self.token_statistics.append({
                "Question": str(question),
                "Input": str(correction_prompt),
                "Input Tokens": input_token_count,
                "Output Tokens": output_token_count,
                "Output": str(correction_answer),
                "Type": "Feedback->Answer",
                "Experiment": "Reverse",
                "Forecast": forecast
            }, ignore_index=True)
            return correction_answer
        except Exception as e:
            print(f"Correction generation failed: {e}")
            return ""

    def evaluate_answer_match(self, generated_answer, standard_answer, max_tokens=500):
        generated_label = extract_first_uppercase(generated_answer)
        standard_label = extract_first_uppercase(standard_answer)
    
        if generated_label == standard_label:
            return "The answer is correct."
        else:
            return "The answer is incorrect."
class AdvancedSelfConsistency:
    def __init__(self, client, model: str, num_samples: int = 5, similarity_threshold: float = 0.3):  
        self.client = client
        self.model = model
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
        self.self_consistency_completion_tokens = 0

    def generate_responses(self, evaluator, message) -> List[str]:
        responses = []
        input_content = "\n".join([msg['content'] for msg in message])

        input_token_count = len(evaluator.tokenizer.encode(input_content))

        for _ in range(self.num_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                max_tokens=4096
            )
            correction_answer = response.choices[0].message.content
            self.self_consistency_completion_tokens += response.usage.completion_tokens
            responses.append(response.choices[0].message.content)
        return responses

    def extract_final_option(self, text: str) -> str:
        pattern = r"option\s*([A-D])"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def calculate_similarity(self, a: str, b: str,evaluator:ModelEvaluator) -> float:
        option_a = evaluator.extract_answer(a)
        option_b = evaluator.extract_answer(b)
        if not option_a :
            option_a = evaluator.extract_final_answer(a)
        if not option_b:
            option_b = evaluator.extract_final_answer(b)
        feedback=evaluator.evaluate_answer_match(option_a, option_b)
        print(f"option_a:{option_a},option_b:{option_b},feedback:{feedback}")
        if option_a and option_b:
            if "The answer is correct." in feedback or "the answer is correct." in feedback:
                return 1.0 
            else:
                return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def cluster_similar_responses(self, responses: List[str],evaluator) -> List[List[str]]:
        clusters = []
        for response in responses:
            added_to_cluster = False
            for cluster in clusters:
                if self.calculate_similarity(response, cluster[0],evaluator) >= self.similarity_threshold:
                    cluster.append(response)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
        return clusters

    def aggregate_results(self, responses: List[str],evaluator) -> Dict[str, any]:
        final_answers = responses
        clusters = self.cluster_similar_responses(final_answers,evaluator)

       
        cluster_info = []
        for cluster in clusters:
            cluster_info.append({
                "answer": cluster[0],  
                "frequency": len(cluster),
                "variants": cluster
            })

        cluster_info.sort(key=lambda x: x['frequency'], reverse=True)

        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def evaluate(self, evaluator, messages) -> Dict[str, any]:
        responses = self.generate_responses(evaluator, messages)
        aggregated_result = self.aggregate_results(responses,evaluator)

    
        return {
            "individual_responses": responses,
            "aggregated_result": aggregated_result
        }

def process_question_with_self_consistency_and_nicot(entry, evaluator, model):
    global True_count
    try:
        self_consistency = AdvancedSelfConsistency(evaluator.client, model)
        
        # Extract question, options and answer\
        context = entry.get("context", "")
        query = entry.get("query", "")
        options = entry.get("options", [])
        correct_option_idx = entry.get("correct_option", 0)  
    
        options_str = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
        question = f"Context:\n{context}\n\nQuestion:\n{query}\nOptions:\n{options_str}"
    
        correct_answer_letter = chr(65 + correct_option_idx)  # 0->A, 1->B, etc.
        correct_answer_text = options[correct_option_idx]
        answer = f"{correct_answer_letter}. {correct_answer_text}"

        
        # Build full question (including options)
        full_question = question
        
        # Build messages for initial generation
        messages = [
            {"role": "user",
             "content": f"""
                    Here is a logical reasoning question. Please analyze the context carefully, 
                    apply the given constraints step by step, and conclude with 
                    The answer is X' where X is the letter (A, B, C, or D).\n\n
                    {question}\n\nLet's think step by step:
"""}
        ]

        # Step 1: Use self-consistency to generate multiple responses and aggregate results
        result = self_consistency.evaluate(evaluator, messages)
        clusters = result["aggregated_result"]["clusters"]
        most_frequent_answer = clusters[0]["answer"] if clusters else "No consistent answer found."
        
        # Record initial self-consistency results
        initial_model_number = evaluator.extract_answer(most_frequent_answer)
        if not initial_model_number:
            initial_model_number = evaluator.extract_final_answer(most_frequent_answer)
        
        # Format expected answer
        expected_number = answer
        
        # Step 2: Apply NiCOT verification and refinement
        max_try = 2
        try_num = 0
        history = []
        current_answer = most_frequent_answer
        final_answer = current_answer
        is_correct = False
        
        while try_num < max_try:
            if try_num == 0:
                # Record initial answer statistics
                evaluator.token_statistics = evaluator.token_statistics.append({
                    "Question": full_question,
                    "Input": str(messages[0]['content']),
                    "Input Tokens": len(evaluator.tokenizer.encode(messages[0]['content'])),
                    "Output Tokens": len(evaluator.tokenizer.encode(current_answer)),
                    "Output": current_answer,
                    "Type": "Question->OriginAnswer",
                    "Experiment": "Self-Consistency+Reverse",
                    "Forecast": "Initial answer before NiCOT refinement"
                }, ignore_index=True)
            
            try_num += 1
            feedback = evaluator.verify_answer(full_question, current_answer, answer)
            history.append({
                'answer': current_answer,
                'feedback': feedback
            })
            
            model_number = evaluator.extract_answer(current_answer)
            if not model_number:
                model_number = evaluator.extract_final_answer(current_answer)
            
            # Check if answer is correct
            evaluation_result = evaluator.evaluate_answer_match(model_number, expected_number)
            if ("The answer is correct." in evaluation_result or 
                "the answer is correct." in evaluation_result or 
                ("correct" in feedback and "incorrect" not in feedback)):
                is_correct = True
                final_answer = current_answer
                break
            
            # Generate refined answer if not correct
            current_answer = evaluator.generate_new_answer(full_question, history, model_number, answer)
            final_answer = current_answer
        
        # Final evaluation
        final_model_number = evaluator.extract_answer(final_answer)
        if not final_model_number:
            final_model_number = evaluator.extract_final_answer(final_answer)
        
        final_evaluation = evaluator.evaluate_answer_match(final_model_number, expected_number)
        if "The answer is correct." in final_evaluation or "the answer is correct." in final_evaluation:
            is_correct_msg = f"Generated answer is {final_model_number}, expected answer is {expected_number}, correct."
            True_count += 1
        else:
            is_correct_msg = f"Generated answer is {final_model_number}, expected answer is {expected_number}, incorrect."
        
        # Save final statistics
        evaluator.token_statistics = evaluator.token_statistics.append({
            "Question": full_question,
            "Input": str(messages[0]['content']),
            "Input Tokens": len(evaluator.tokenizer.encode(messages[0]['content'])),
            "Output Tokens": len(evaluator.tokenizer.encode(final_answer)),
            "Output": final_answer,
            "Type": "Question->FinalAnswer",
            "Experiment": "Self-Consistency+Reverse",
            "Forecast": is_correct_msg
        }, ignore_index=True)

        return {
            "question": full_question,
            "initial_model_response": most_frequent_answer,
            "final_model_response": final_answer,
            "expected_answer": expected_number,
            "initial_model_number": initial_model_number,
            "final_model_number": final_model_number,
            "is_correct": is_correct_msg,
            "clusters": result["aggregated_result"]["clusters"],
            "refinement_history": history
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": entry.get("question"),
            "error": str(e)
        }

# Configuration

evaluator = ModelEvaluator(API_KEY, BASE_URL, MODEL_NAME)

directories = [ "logiqa"]

def process_directory(directory):
    global True_count
    input_file = f"{directory}/logiqa_test.jsonl"
    
    with open(input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    
    start_time = time.time()
    n = 0 
    count = 0 
    while count < 1:  
        for i in range(14): 
            m = n + 50  
            print(f"Processing {directory}: Attempt {count + 1}, Index {n} to {m}")
            
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda entry: process_question_with_self_consistency_and_nicot(entry, evaluator, MODEL_NAME), data[n:m]))
            
            
            output_file = f"self_consistency+Reverse_{directory}_{MODEL_NAME}_{n}-{m}.json"
            accuracy = True_count / len(results) * 100
            True_count = 0
            output_data = {
                "results": results,
                "accuracy": accuracy,
                "time_elapsed": time.time() - start_time
            }
            
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)
            
            
            evaluator.save_statistics(filename=f"self_consistency+Reverse_{directory}_{MODEL_NAME}_statistics_{n}-{m}.csv")
            evaluator.token_statistics = pd.DataFrame(
                columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
                dtype=object  
            )
            n = m  
        count += 1
        n = 0 


for directory in directories:
    process_directory(directory)

print("Self-consistency evaluation for logiqa dataset complete.")