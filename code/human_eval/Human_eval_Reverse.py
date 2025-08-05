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


def extract_first_uppercase(s):
            match = re.search(r'[A-E]', s)
            return match.group(0) if match else None

def clean_for_gbk(text):
    if pd.isna(text):
        return ""
    try:
        return str(text).encode('gbk', errors='ignore').decode('gbk')  
    except:
        return str(text)

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
        self.token_statistics = self.token_statistics.applymap(clean_for_gbk)
        if os.path.exists(filename):
            self.token_statistics.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            self.token_statistics.to_csv(filename, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"Token statistics saved to {filename}")


    def extract_final_answer(self, generated_answer, max_tokens=4000):
        # Add the fixed prompt to guide the model in extracting the final answer
        messages = [
    {
        "role": "user",
        "content": (
            "Here is a generated answer that may contain explanations and code.\n"
            "Please extract only the Python function implementation code from it.\n"
            "If no valid Python code is found, respond with 'No code found.'"
        )
    },
    {
        "role": "assistant",
        "content": "Sure! Please provide the generated answer, and I will extract the code for you."
    },
    {
        "role": "user",
        "content": f"Generated Answer:\n{generated_answer}"
    }
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

    def extract_answer(self, text, prefixes=("The code is:", "the code is:")):
        """
        Extract the content following the specified prefixes in the text.
        Handles both 'The answer is' and 'the answer is', including newlines.
        """
        try:
            if not isinstance(text, str):
                return None
            pattern = rf"(?i)\b(?:{'|'.join(map(re.escape, prefixes))})\b\s*[:：]?\s*(.*)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return text
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return None


    def evaluate(self, question, max_tokens=4000):
        # Add the fixed prompts to the messages
        messages = [
            {"role": "user",
             "content": f"Here is a multiple-choice question about commonsense reasoning. Please reason through it step by step, and at the end, provide your answer option with 'the answer is X', where 'X' is the letter of the correct option from the choices provided. Here is the question you need to answer:\n{question}\nLet's think step by step:"}
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

    def generate_reverse_feedback_prompt(self, question, generated_answer):
        return f"""
You are an expert in program semantics and code logic validation
Your task is to perform reverse validation on the code implementation only generated by a large language model for a HumanEval programming task

Evaluation procedure
Apply three reverse validation techniques on code

Validation one Example consistency check
Task
Run or mentally simulate the provided code on example inputs from the question
Check whether the output of the code matches the expected results
Verdict
If the output is incorrect or mismatched with the expected behavior this step fails

Validation two Code to specification recovery
Task
Try to infer the problem requirements based solely on the code implementation
Check whether the inferred specification aligns with the actual task description
Verdict
If the inferred behavior does not match the original problem’s requirement this step fails

Validation three Code mutation robustness
Task
Change one or two lines of code such as modifying key logic conditions or iterations
Test the modified version on the same example inputs
Check whether the behavior still aligns with the problem specification
Verdict
If the new behavior is still correct the original code may not be logically necessary indicating weak correctness This step fails

Final verdict
If any validation fails write The answer is incorrect
If all validations pass write The answer is correct
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
                "Experiment": "Reverse",
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
        You are an expert programming assistant with strong reasoning and debugging capabilities. Your task is to identify and correct mistakes in code generation for programming problems, using external feedback to guide the correction process. You are working with code generated for the HumanEval benchmark.
Your Task:
You are given a programming task along with a model's previous answer and corresponding feedback. Your goal is to produce a corrected solution by carefully analyzing and incorporating the feedback.
Question:

{question}
Previous Answer & Feedback:

{history_prompt}
Step 1: Identify Mistakes

Carefully review the feedback. Summarize the mistakes in the previous answer. Be specific and mention whether the issue lies in logic, incorrect implementation, missing edge cases, or misunderstanding of the problem.
Step 2: Apply Feedback

    First, evaluate whether the feedback itself is logically valid and technically correct.

    Then, strictly follow the valid feedback to rederive the solution from scratch.

    Walk through the reasoning step by step. Ensure correctness and handle all relevant edge cases.

Final Output Rules:
    You must rewrite the complete corrected function.
    Finally, please output the final implementation below.
    # The code is:
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

            self.token_statistics = self.token_statistics.append({
                "Question": str(question),
                "Input": str(correction_prompt),
                "Input Tokens": input_token_count,
                "Output Tokens": output_token_count,
                "Output": str(correction_answer),
                "Type": "Feedback->Answer",
                "Experiment": "Reverse",
                "Forecast": None
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

def clean_model_answer(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() 


def inject_cot_plan(prompt: str, entry_point: str) -> str:
    lines = prompt.strip().split("\n")
    new_lines = []
    inserted = False
    post_marker_added = False

    for line in lines:
        new_lines.append(line)
        if not inserted and line.strip().startswith(f"def {entry_point}"):
            cot_plan = [
                "    # Let's think step by step:",
                "    # Finally, please output the final implementation below.",
                "    # The code is:"
            ]
            new_lines.extend(cot_plan)
            inserted = True

    return "\n".join(new_lines)

def process_question(entry):
    prompt = entry.get("prompt")               
    task_id = entry.get("task_id")             
    answer = entry.get("canonical_solution") 
    history = []
    current_answer = ''
    model_answer = ''
    prompt_raw = entry.get("prompt")
    entry_point = entry.get("entry_point")
    question = prompt_raw 

    max_try = 1
    try_num = 0
    history = []
    current_answer = ''
    try:
        # Call the API to get the model's response
        while try_num < max_try:
            if try_num == 0:
                current_answer = evaluator.evaluate(question)
                model_input = f"Here is a multiple-choice question about commonsense reasoning. Please reason through it step by step, and at the end, provide your answer option with 'the answer is X', where 'X' is the letter of the correct option from the choices provided. Here is the question you need to answer:\n{question}\nLet's think step by step:"
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
            if not model_answer:
                model_answer = evaluator.extract_final_answer(current_answer)
            expected_number = answer
            if "The answer is correct" in feedback or "the answer is correct" in feedback or ("correct" in  feedback and "incorrect" not in feedback):
                history = []
                print(
                    ".............................................................................the next answer..........................................................................")
                model_answer = clean_model_answer(model_number)
                return {
                    "task_id": task_id,
                    "prompt": prompt,
                    "model_response": current_answer,
                    "model_answer": model_answer,
                    "canonical_solution": answer,
                    "is_correct": None 
                }
            current_answer = evaluator.generate_new_answer(question, history, model_number, answer)

        model_number = evaluator.extract_answer(text=current_answer)
        expected_number = answer
        if not model_answer:
            model_answer = evaluator.extract_final_answer(current_answer)
        evaluation_result = evaluator.evaluate_answer_match(model_number, expected_number)
        if "The answer is correct." in evaluation_result:
            is_correct = True
        else:
            is_correct = False
        history = []
        model_answer = clean_model_answer(model_number)
        return {
            "task_id": task_id,
            "prompt": prompt,
            "model_response": current_answer,
            "model_answer": model_answer,
            "canonical_solution": answer,
            "is_correct": None  
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": question,
            "error": str(e)
        }


# Evaluate the questions with multithreading

directories = ["human_eval"]

def process_directory(directory):
    input_file = f"test.jsonl"
    
    with open(input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    
    start_time = time.time()
    n = 0
    count = 0
    while count < 1:
        for i in range(4):
            m = n + 50
            print(f"Processing {directory}: Attempt {count+1}, Index {n} to {m}")
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_question, data[n:m]))

            output_file = f"Human_eval_Reverse_{MODEL_NAME}_{n}-{m}.json"
            accuracy = sum(1 for result in results if result.get("is_correct")) / len(results) * 100 if results else 0
            output_data = {"results": results, "accuracy": accuracy, "time_elapsed": time.time() - start_time}
            
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)

            evaluator.save_statistics(filename=f"Human_eval_Reverse_{MODEL_NAME}_{n}-{m}.csv")
            evaluator.token_statistics = pd.DataFrame(
            columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
            dtype=object  
            )
            n = m
        count += 1
        n = 0  

# 依次处理各个文件夹
for directory in directories:
    process_directory(directory)
