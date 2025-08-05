import json
import requests
from openai import OpenAI
import re
import time
import tiktoken
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
# Configuration,you need to fill in your API key, base URL, and model name
API_KEY="" 
BASE_URL=""
MODEL_NAME = ""


def extract_first_uppercase(s):
            match = re.search(r'[A-E]', s)
            return match.group(0) if match else None


def clean_for_gbk(text):
    if pd.isna(text):
        return ""
    try:
        return str(text).encode('gbk', errors='ignore').decode('gbk')  # 删除无法编码的字符
    except:
        return str(text)

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
            api_key= api_key,  
            base_url=base_url,
        )

    def make_request_with_retries(self, function, max_retries=3, wait_time=10, *args, **kwargs):
        """
        发送请求，遇到错误时重试。
        """
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

    def extract_final_answer(self, generated_answer, max_tokens=4000):
        """
        Extract the final answer from the provided generated answer using the model API.
        """
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

    def extract_answer(self, text, prefixes=("The answer is", "the answer is")):
        """
        Extract the content following the specified prefixes in the text.
        Handles both 'The answer is' and 'the answer is', including newlines.
        """
        try:
            pattern = rf"(?i)\b(?:{'|'.join(map(re.escape, prefixes))})\b\s*[:：]?\s*(.*)"
            match = re.search(pattern, text, re.DOTALL)  # Use re.DOTALL to include newlines in the match
            if match:
                return match.group(1).strip()  # Return the content after the prefix, stripped of extra spaces
            else:
                return None
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None

    def evaluate(self, question, max_tokens=4000):
        """Evaluate a single question using the model API."""
        # Add the fixed prompts to the messages
        messages = [
            {"role": "user",
             "content": f"Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, provide your answer option with 'the answer is  X', Where 'X' is the correct option you think from A,B,C,D. Here is the question you need to answer:\n{question}\nLet's think step by step:"}
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

    # 定义反馈器的 Prompt 模板
    def generate_feedback_prompt(self, question, generated_answer):
        return f"""
        Instruction:
You are a highly meticulous verification assistant specializing in reverse chain-of-thought (COT) reasoning. Your goal is to identify any possible inconsistencies or errors, even subtle ones, in the generated answer.

Task:
    Start from the final answer provided in the generated answer.
    Work backward step-by-step, verifying each reasoning step against the conditions described in the question.
    Be particularly vigilant about checking numerical calculations, logical consistency, and implicit assumptions.
    For each step A→B, rigorously verify:
        Conflict with Problem Conditions:
            Check if B conflicts with the conditions described in the question. Ensure that B does not violate any constraints or assumptions stated in the problem.
            If B contradicts the problem conditions, classify the step as incorrect.
        Equivalence and Reversibility (if A and B are equivalent):
            If A and B are logically equivalent (i.e., AB), verify whether B→A also holds.
            Ensure that the reasoning is bidirectional and that no additional assumptions are introduced when moving from B back to A.
        Inverse Consistency: Check if B logically leads to A. Ensure that the negation of B would indeed invalidate A, and that no edge cases are overlooked.
    If all steps and conditions are satisfied, output: "The answer is correct."
    If there is any inconsistency, numerical mistake, or questionable assumption at any step, classify the answer as incorrect.
    Clearly state which step contains the contradiction.
Guidelines:
    Clearly identify and state each step of the reverse reasoning process.
    Err on the side of caution: If an intermediate result is even slightly questionable, classify the answer as incorrect.
    Ensure that all numerical values, logical connections, and intermediate results are verified with precision.
    Be precise and concise in your validation.
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
                "Experiment": "reverse",
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

**Question:**
{question}

**Answer History & Feedback:**
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
Please end with 'the answer is X', where 'X' is the correct option you think from A,B,C,D.
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
                "Experiment": "reverse",
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



# Initialize the evaluator
evaluator = ModelEvaluator(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)




def process_question(entry):
    question = entry.get("question")
    answer = entry.get("answer")
    opinions = entry.get("options")
    options_str = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(opinions)])
    question = f"{question}\nOptions:\n{options_str}"
    index = ord(answer.upper()) - ord('A')
    selected_option = opinions[index]
    answer = f"{answer.upper()}.{selected_option}"
    max_try = 2
    try_num = 0
    history = []
    current_answer = ''
    try:
        # Call the API to get the model's response
        while try_num < max_try:
            if try_num == 0:
                current_answer = evaluator.evaluate(question)
                model_input = f"Here is a multiple-choice question about mathematics. Please reason through it step by step, and at the end, provide your answer option with 'Therefore, the correct answer is option X', Where 'X' is the correct option you think from A,B,C,D. Here is the question you need to answer:\n{question}\nLet's think step by step:"
                output_token_count = len(evaluator.tokenizer.encode(current_answer))
                input_token_count = len(evaluator.tokenizer.encode(model_input))
                new_row = pd.DataFrame([{
                    "Question": question,
                    "Input": model_input,
                    "Input Tokens": input_token_count,
                    "Output Tokens": output_token_count,
                    "Output": str(current_answer),
                    "Type": "Question->OriginAnswer",
                    "Experiment": "Reverse",
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

directories = ["college","high","middle"]

def process_directory(directory):
    input_file = f"{directory}/single_choice_en.jsonl"
    
    with open(input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    
    start_time = time.time()
    n = 0
    count = 0
    while count < 1:
        for i in range(3):
            m = n + 50
            print(f"Processing {directory}: Attempt {count+1}, Index {n} to {m}")

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_question, data[n:m]))
            
            output_file = f"Reverse_{directory}_{MODEL_NAME}_mathbench_{n}-{m}.json"
            accuracy = sum(1 for result in results if result.get("is_correct")) / len(results) * 100 if results else 0
            output_data = {"results": results, "accuracy": accuracy, "time_elapsed": time.time() - start_time}
            
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)

            evaluator.save_statistics(filename=f"Reverse_{directory}_{MODEL_NAME}_mathbench_statistics_{n}-{m}.csv")
            evaluator.token_statistics = pd.DataFrame(
            columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
            dtype=object  
            )
            n = m
        count += 1
        n = 0  

for directory in directories:
    process_directory(directory)
