import json
import requests
from openai import OpenAI
import re
import time
import tiktoken
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
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
        # target_dir = r"F:\llama-models\independent_evaluation\grade-school-math\result"
        # file_path = os.path.join(target_dir, filename)
        if os.path.exists(filename):
            self.token_statistics.to_csv(filename, mode='a', header=False, index=False)
        else:
            self.token_statistics.to_csv(filename, mode='w', header=True, index=False)
        print(f"Token statistics saved to {filename}")

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

    def extract_answer(self,text):
        if text is None:
            return None
        try:
            pattern = r"(?i)\bthe answer is\b\s*[:：]?\s*(.*)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            boxed_start_tag = "\\boxed{"
            start_index = text.find(boxed_start_tag)
            if start_index != -1:
                open_braces = 1 
                content_start_index = start_index + len(boxed_start_tag)
                for i in range(content_start_index, len(text)):
                    if text[i] == '{':
                        open_braces += 1
                    elif text[i] == '}':
                        open_braces -= 1
                    if open_braces == 0:
                        return text[content_start_index:i].strip()

            match = re.search(r"\\boxed\{(.+?)\}", text)
            if match:
                return match.group(1).strip()

            return None
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def evaluate(self, question, max_tokens=4000):
        # Add the fixed prompts to the messages
        messages = [
            {"role": "user",
             "content": f"Question: {question}\nLet's think step by step\nPlease end with 'The answer is'\nAnswer:"}
        ]

        # Prepare the API request payload
        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens
        }

        # Make the API call
        response = self.make_request_with_retries(self.client.chat.completions.create, 3, 60, **data)
        current_answer = response.choices[0].message.content
        return current_answer

    def generate_feedback_prompt(self, question, generated_answer):
        return f"""
    Instruction:
# The following generated answer may contain errors. Please check step by step to ensure that each part is correct in order to identify the error.
# If all conditions are satisfied, output: "The answer is correct."
# If there is a contradiction, output: "The answer is incorrect." Explain where the contradiction occurs.
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
            expected_number = self.extract_answer(text=origin_answer)
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
                "Experiment": "Forward",
                "Forecast": forcast

            }, ignore_index=True)
            return feedback
        except Exception as e:
            print(f"Verification failed: {e}")
            return ""

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
End your response with 'The answer is'
                                """

        input_token_count = len(self.tokenizer.encode(correction_prompt))
        try:
            data = {
                'model': self.model_name,
                'messages': [
                    {'role': 'user', 'content': correction_prompt}
                ],
                'temperature': 0.7,
                
                'max_tokens': 4096
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
            expected_number = self.extract_answer(origin_answer)
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
                "Experiment": "Forward",
                "Forecast": forecast
            }, ignore_index=True)
            return correction_answer
        except Exception as e:
            print(f"Correction generation failed: {e}")
            return ""

    def evaluate_answer_match(self, generated_answer, standard_answer, max_tokens=500):
        # Prepare the prompt to instruct the model
        messages = [
            {"role": "user",
             "content": f"""
        You are an evaluator. Your task is to determine if the generated answer matches the standard answer.
        ### Task:
        1. Compare the generated answer with the standard answer.
        2. Ignore all formatting issues (like extra characters, spaces, or LaTeX symbols) and only evaluate whether the two answers are logically equivalent and.
        3. If they are identical or logically equivalent, respond with: "The answer is correct."
        4. If they do not match in terms of logical equivalence, respond with: "The answer is incorrect." and explain why.
        ### Input:
        - Generated Answer: {generated_answer}
        - Standard Answer: {standard_answer}
        ### Output:
        Provide your evaluation here:
        """}
        ]

        # Prepare the API request payload
        data = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': max_tokens
        }

        # Make the API call
        response = self.client.chat.completions.create(**data)
        evaluation = response.choices[0].message.content.strip()
        return evaluation


# Initialize the evaluator
evaluator = ModelEvaluator(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)

# Load the dataset
input_file = "test.jsonl.txt"
with open(input_file, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]


def process_question(entry):
    question = entry.get("problem")
    answer = entry.get("answer")
    max_try = 2
    try_num = 0
    history = []
    current_answer = ''
    try:
        # Call the API to get the model's response
        while try_num < max_try:
            if try_num == 0:
                current_answer = evaluator.evaluate(question)
                model_input = f"Question: {question}\nLet's think step by step\nPlease end with 'The answer is'\nAnswer:"
                output_token_count = len(evaluator.tokenizer.encode(current_answer))
                input_token_count = len(evaluator.tokenizer.encode(model_input))
                new_row = pd.DataFrame([{
                    "Question": question,
                    "Input": model_input,
                    "Input Tokens": input_token_count,
                    "Output Tokens": output_token_count,
                    "Output": str(current_answer),
                    "Type": "Question->OriginAnswer",
                    "Experiment": "Forward",
                    "Forecast": "null"
                }])
                evaluator.token_statistics = pd.concat([evaluator.token_statistics, new_row], ignore_index=True)
            try_num += 1
            feedback = evaluator.verify_answer(question, current_answer, answer)
            history.append({
                'answer': current_answer,
                'feedback': feedback
            })
            model_number = evaluator.extract_answer(current_answer)
            if model_number:
                model_number = model_number
            else:
                model_number = evaluator.extract_final_answer(current_answer)
            expected_number = answer
            if "The answer is correct" in feedback or "the answer is correct" in feedback or ("correct" in  feedback and "incorrect" not in feedback):
                history = []
                print(
                    ".............................................................................the next answer..........................................................................")
                evaluation_result = evaluator.evaluate_answer_match(model_number, expected_number)
                if "The answer is correct." in evaluation_result:
                    is_correct = True
                else:
                    is_correct = False
                return {
                    "question": question,
                    "model_response": current_answer.strip(),
                    "expected_answer": answer,
                    "model_number": model_number,
                    "expected_number": expected_number,
                    "is_correct": is_correct
                }   
            current_answer = evaluator.generate_new_answer(question, history, model_number, answer)

        model_number = evaluator.extract_answer(text=current_answer)
        expected_number = answer
        if model_number:
            model_number = model_number
        else:
            model_number = evaluator.extract_final_answer(current_answer)
        evaluation_result = evaluator.evaluate_answer_match(model_number, expected_number)
        if "The answer is correct." in evaluation_result:
            is_correct = True
        else:
            is_correct = False
        history = []
        return {
            "question": question,
            "model_response": current_answer.strip(),
            "expected_answer": answer,
            "model_number": model_number,
            "expected_number": expected_number,
            "is_correct": is_correct
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": question,
            "error": str(e)
        }


# Evaluate the questions with multithreading


start_time = time.time()
n = 0 # 
count = 0
while count < 1:
    for i in range(10):
        m = n + 50
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_question, data[n:m]))

        # Save results to a file
        output_file = f"math500_{MODEL_NAME}_Forward_{n}-{m}.json"
        if len(results) > 0:
            output_data = {
                "results": results,
                "accuracy": sum(1 for result in results if result.get("is_correct")) / len(results) * 100,
                "time_elapsed": time.time() - start_time
            }
        else:
            output_data = {
                "results": results,
                "accuracy": sum(1 for result in results if result.get("is_correct")) / 50 * 100,
                "time_elapsed": time.time() - start_time
            }

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output_data, file, ensure_ascii=False, indent=4)

        evaluator.save_statistics(filename = f"math500_{MODEL_NAME}_Forward_{n}-{m}.csv")
        evaluator.token_statistics = pd.DataFrame(
            columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
            dtype=object  
        )

        n = m  
    count += 1
    n=0

