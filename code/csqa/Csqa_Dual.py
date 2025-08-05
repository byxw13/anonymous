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
            api_key= api_key,  
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

    def generate_feedback_prompt(self, question, generated_answer):
        return f"""
    Instruction:
You are an expert in commonsense reasoning and logical validation. Your task is to perform a reverse validation of the provided answer to a commonsense question. Your goal is to uncover any inconsistencies, weak justifications, or commonsense fallacies.

Task:
1.Start from the final answer provided in the generated answer.
2.Trace back step by step, quickly verifying whether each reasoning step conforms to the main conditions of the problem.
3.For each step Premise→Answer, simply verify the following:

    Validation one 
    Task: Check if there is a direct factual or logical conflict between the Answer and the Premise .
    Verdict: If the combination of Answer and Premise results in a contradiction, the validation fails.

    Validation two
    If Premise and Answer are logically equivalent (i.e., Premise↔Answer), verify whether Answer→Premise also holds.
    If it is a "One-Way Implication": This validation is not applicable; skip it directly.

    Validation three
    Check When the answer is not true (¬Answer), if the premise could possibly be true or could possibly be false, the validation passes.Only When the answer is not true (¬Answer), the premise must be true, the validation fails.

4. Trust your intuition: If the intermediate result seems reasonable, there is no need to question it excessively, without over-analyzing possible inaccuracies in the problem.
5. If any of the above applicable validation steps results in a "failure" verdict, the final verdict is:The answer is incorrect.If all applicable validation steps pass (or are marked as "not applicable" and skipped), the final verdict is:The answer is correct.

Question:
{question}

Generated Answer:
{generated_answer}
"""
    def generate_forward_feedback_prompt(self, question, generated_answer):
        return f"""
    Instruction:
    # The following generated answer may contain errors. Please check step by step to ensure that each part is correct in order to identify the error.
    # Trust your intuition: If the intermediate result seems reasonable, there is no need to question it excessively, without over-analyzing possible inaccuracies in the problem.
    # If all conditions are satisfied, output: "The answer is correct."
    # If there is a contradiction, output: "The answer is incorrect." Explain where the contradiction occurs.
    Question:
    {question}

    Generated Answer:
    {generated_answer}
    """


    def verify_answer(self, question: str, answer: str, origin_answer) -> str:
        forward_prompt = self.generate_forward_feedback_prompt(question, answer)  
        reverse_prompt = self.generate_feedback_prompt(question, answer)

        input_tokens_forward = len(self.tokenizer.encode(forward_prompt))
        input_tokens_reverse = len(self.tokenizer.encode(reverse_prompt))

        try:
            forward_resp = self.make_request_with_retries(
                self.client.chat.completions.create, 3, 60,
                model=self.model_name,
                messages=[{'role': 'user', 'content': forward_prompt}],
                temperature=0.5,
                max_tokens=4000
            ).choices[0].message.content

            reverse_resp = self.make_request_with_retries(
                self.client.chat.completions.create, 3, 60,
                model=self.model_name,
                messages=[{'role': 'user', 'content': reverse_prompt}],
                temperature=0.5,
                max_tokens=4000
            ).choices[0].message.content

            forward_wrong = True
            reverse_wrong = True

            if "The answer is correct." in forward_resp or "the answer is correct." in forward_resp:
                forward_wrong = False
            if "The answer is correct." in reverse_resp or "the answer is correct." in reverse_resp:
                reverse_wrong = False

            if forward_wrong != reverse_wrong:
                review_prompt = f"""
You are a Fact-Checking Adjudicator. Your task is to resolve a dispute about a 'Generated Answer' by performing targeted, independent verifications on the most critical points of disagreement. **You must pinpoint the exact point of failure without generating a full, new step-by-step solution.**

**Question:**
{question}

**Generated Answer:**
{answer}

**Forward View Feedback:**
{forward_resp}

**Reverse View Feedback:**
{reverse_resp}

### Instructions:

**1. Identify the Crux of the Disagreement:**
-   Read all provided inputs (`Generated Answer` and the two `Feedbacks`).
-   What is the single most critical calculation, logical step, or interpretation of a premise where the views conflict? State this crux clearly.

**2. Perform Targeted ("Surgical") Verification:**
-   Focus **only** on the critical point of disagreement you identified in Step 1.
-   **Perform a focused, independent calculation or logical check** to resolve this specific point. You MUST use only the original `Question` as your source of truth for this check.
-   State the result of your targeted check clearly and concisely.
**3. Render the Final Verdict:**
-   Based **only on the outcome of your targeted verification in Step 2**, determine if the `Generated Answer`'s final conclusion is correct or incorrect.
-   Provide a **brief (1-2 sentences) explanation** that hinges directly on the result of your verification.
-   **Crucially, do NOT write out a new, full solution.**

Your entire response must end with one of the two following exact statements:
"The answer is correct."
or
"The answer is incorrect."
    """
                review_resp = self.make_request_with_retries(
                    self.client.chat.completions.create, 3, 60,
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': review_prompt}],
                    temperature=0.5,
                    max_tokens=4000
                ).choices[0].message.content

                final_feedback = review_resp
            else:
                final_feedback = reverse_resp

            model_number = self.extract_answer(answer) 
            if model_number:
                model_number = model_number
            else:
                model_number = self.extract_final_answer(answer)
            expected_number = origin_answer
            evaluation_result = self.evaluate_answer_match(model_number, expected_number)

            is_correct = "The answer is correct." in evaluation_result or "the answer is correct." in evaluation_result
            is_predicted_correct = "the answer is correct" in final_feedback or "The answer is correct" in final_feedback

            if is_predicted_correct and is_correct:
                forecast = f"Feedback indicates the generated answer, {model_number}, is correct. The standard answer is {expected_number}, which matches the generated answer. Prediction: Correct."
            elif is_predicted_correct and not is_correct:
                forecast = f"Feedback indicates the generated answer, {model_number}, is correct. However, the standard answer is {expected_number}, which does not match the generated answer. Prediction: Incorrect."
            elif not is_predicted_correct and is_correct:
                forecast = f"Feedback indicates the generated answer, {model_number}, is incorrect. However, the standard answer is {expected_number}, which matches the generated answer. Prediction: Incorrect."
            else:
                forecast = f"Feedback indicates the generated answer, {model_number}, is incorrect. The standard answer is {expected_number}, which does not match the generated answer. Prediction: Incorrect."

            output_tokens = len(self.tokenizer.encode(final_feedback))
            self.token_statistics = self.token_statistics.append({
                "Question": str(question),
                "Input": str(reverse_prompt),
                "Input Tokens": input_tokens_reverse + input_tokens_forward,
                "Output Tokens": output_tokens,
                "Output": str(final_feedback),
                "Type": "Answer->Feedback",
                "Experiment": "Dual",
                "Forecast": forecast
            }, ignore_index=True)

            return final_feedback

        except Exception as e:
            print(f"Combined verify failed: {e}")
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
Task 1: Identify Mistakes
Summarize the specific mistakes in the previous answers based on the given feedback. Ensure that you highlight where the previous answers went wrong, paying close attention to feedback that indicates miscalculations or incorrect assumptions.
Task 2: Apply Feedback
Correct the mistakes identified in Task 1 strictly following the feedback.
    Rederive the answer from scratch, ensuring that each step is clearly explained and follows the correct reasoning.
    Pay careful attention to the feedback to avoid the same mistakes, and ensure the corrected answer aligns with the feedback provided.

Rules (You Must Follow These)
    Rederive the answer from scratch.
    First, evaluate whether the feedback is correct. Before applying the feedback, carefully assess the logic and accuracy of the feedback to ensure it is free from errors.

Once all mistakes are corrected, provide the final answer.
Please end with ' the answer is X', Where 'X' is the correct option you think from A，B，C，D，E.
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
                "Experiment": "Dual",
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
    choices_dict = entry.get("choices", {}) 

    option_labels = choices_dict.get("label", [])  
    option_texts = choices_dict.get("text", [])    

    options_list = [f"{label}. {text}" for label, text in zip(option_labels, option_texts)]
    options_str = "\n".join(options_list)


    full_question_with_options = f"{question}\n{options_str}"
    question = full_question_with_options

    correct_answer_key = entry.get("answerKey")  

    index = ord(correct_answer_key.upper()) - ord('A')

    selected_option_text = option_texts[index]

    answer = correct_answer_key

    max_try = 2
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
                    "Experiment": "Dual",
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

directories = ["csqa"]

def process_directory(directory):
    input_file = f"test.json"
    
    with open(input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    
    start_time = time.time()
    n = 0
    count = 0
    while count < 1:
        for i in range(13):
            m = n + 100
            print(f"Processing {directory}: Attempt {count+1}, Index {n} to {m}")
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_question, data[n:m]))
            
            output_file = f"Csqa_Dual_{MODEL_NAME}_results_{n}-{m}.json"
            accuracy = sum(1 for result in results if result.get("is_correct")) / len(results) * 100 if results else 0
            output_data = {"results": results, "accuracy": accuracy, "time_elapsed": time.time() - start_time}
            
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, ensure_ascii=False, indent=4)

            evaluator.save_statistics(filename=f"Csqa_Dual_{MODEL_NAME}_results_{n}-{m}.csv")
            evaluator.token_statistics = pd.DataFrame(
            columns=["Question", "Input", "Input Tokens", "Output Tokens", "Output", "Type", "Experiment", "Forecast"],
            dtype=object 
            )
            n = m
        count += 1
        n = 0  

for directory in directories:
    process_directory(directory)
