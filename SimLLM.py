
import os  
import shutil  
import random  
import pandas as pd  
import numpy as np  
import nltk  
import google.generativeai as genai  
import csv  
from transformers import (  
    AutoTokenizer,  
    DataCollatorWithPadding,  
    AutoModelForSequenceClassification,  
    EarlyStoppingCallback,  
    TrainerCallback,  
    TrainingArguments,  
    Trainer  
)  
from openai import OpenAI  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import roc_auc_score, accuracy_score  
from os.path import join  
from langchain.chat_models import ChatOpenAI  
from datasets import load_metric, load_dataset, Dataset  
from copy import deepcopy  
from bart_score import BARTScorer  
import argparse  
  
# Constants  
TOGETHER_API_KEY = "your_together_api_key"  
OPENAI_API_KEY = "your_openai_key"  
GEMINI_API_KEY = "your_gemini_key"  
LOG_FILE = "data/99_log.txt"  
OUTPUT_FILE = "data/result.txt"  
METRIC_NAME = "roc_auc"  

TRAIN_RATIO = 0.8  
VAL_RATIO = 0.1  
NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING = 10  
PATIENCE = 3  
BATCH_SIZE = 64  
OPTIMIZED_METRIC = "roc_auc"  
SEED = 0  
TEMPERATURE = 0.0  
IS_OUTPUT_NORMALIZATION = False  
RATIO = 0.9  
HUMAN_LABEL = 0
MACHINE_LABEL = 1
BART = "bart"

MULTIMODEL = "multimodel"
SINGLE_FROM_MULTIMODEL = "single_from_multimodel"

# Environment setup  
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY  
os.environ['CURL_CA_BUNDLE'] = ''  
os.environ['REQUESTS_CA_BUNDLE'] = ''  
  
# Download necessary NLTK data  
nltk.download('punkt')  
  
# Chat model configurations  
chat_model = ChatOpenAI(temperature=TEMPERATURE, model_name="gpt-3.5-turbo-0125")  
  
# API Models and Paths  
CHATGPT = "ChatGPT"  
GEMINI = "Gemini"  
# LLAMA_2_70_CHAT_TEMP_0 = "LLaMa"  
API_ERROR = "API_ERROR"  
IGNORE_BY_API_ERROR = "IGNORE_BY_API_ERROR"  
  
# Initialize BARTScorer  
bart_scorer = BARTScorer(device='cuda:0', checkpoint="facebook/bart-large-cnn")  
  
# Generative AI configuration  
genai.configure(api_key=GEMINI_API_KEY, transport='rest')  
generation_config = {  
    "temperature": TEMPERATURE,  
}  
GEMINI_MODEL = genai.GenerativeModel('gemini-pro', generation_config=generation_config)  
  
# Model paths  
MODEL_PATHS = {  
    "LLaMa": "meta-llama/Llama-2-70b-chat-hf",  
    "QWEN": "Qwen/Qwen1.5-72B-Chat",  
    "Yi": "NousResearch/Nous-Hermes-2-Yi-34B",  
    "Mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",  
    "OLMo": "allenai/OLMo-7B-Instruct",  
    "Phi": "microsoft/phi-2",  
    "OpenChat": "openchat/openchat-3.5-1210",  
    "WizardLM": "WizardLM/WizardLM-13B-V1.2",  
    "Vicuna": "lmsys/vicuna-13b-v1.5"  
}  

TOGETHER_PATH ='https://api.together.xyz'
  
# Roberta model configurations  
ROBERTA_BASE = "roberta-base"  
ROBERTA_LARGE = "roberta-large"  
ROBERTA_MODEL_PATHS = {  
    ROBERTA_BASE: "roberta-base",  
    ROBERTA_LARGE: "roberta-large"  
}  
LEARNING_RATES = {  
    ROBERTA_BASE: 2e-5,  
    ROBERTA_LARGE: 8e-6  
}  
MODEL_NAME = ROBERTA_BASE  


  
# Tokenizer initialization  
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATHS[MODEL_NAME])  
  
# Custom callback for Trainer  
class CustomCallback(TrainerCallback):  
    """  
    Custom callback to evaluate the training dataset at the end of each epoch.  
    """  
    def __init__(self, trainer) -> None:  
        super().__init__()  
        self._trainer = trainer  
  
    def on_epoch_end(self, args, state, control, **kwargs):  
        """  
        At the end of each epoch, evaluate the training dataset.  
        """  
        if control.should_evaluate:  
            control_copy = deepcopy(control)  
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")  
            return control_copy  
  
# Metric loading  
metric = load_metric(METRIC_NAME)  

def compute_metrics(evaluation_predictions):
    """
    Function to compute evaluation metrics for model predictions.
    
    Parameters:
    evaluation_predictions (tuple): A tuple containing two elements:
        - predictions (array-like): The raw prediction scores from the model.
        - labels (array-like): The true labels for the evaluation data.
    
    Returns:
    dict: A dictionary containing the computed evaluation metrics.
    """
    # Unpack predictions and labels from the input tuple
    raw_predictions, true_labels = evaluation_predictions

    # Convert raw prediction scores to predicted class labels
    predicted_labels = np.argmax(raw_predictions, axis=1)

    # Compute and return the evaluation metrics
    return metric.compute(prediction_scores=predicted_labels, references=true_labels, average="macro")


def abstract_proofread(model_path, temperature, base_url, api_key, prompt):
    """
    Function to proofread an abstract using an AI language model.
    
    Parameters:
    model_path (str): The path or identifier of the AI model to use.
    temperature (float): Sampling temperature for the model's output.
    base_url (str): The base URL for the API endpoint.
    api_key (str): The API key for authentication.
    prompt (str): The text prompt to provide to the AI for proofreading.
    
    Returns:
    str: The proofread abstract generated by the AI model.
    """
    # Initialize the AI client with the provided API key and base URL
    ai_client = OpenAI(api_key=api_key, base_url=base_url)

    # Create a chat completion request with the system message and user prompt
    chat_completion = ai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_path,
        max_tokens=1024,
        temperature=temperature,
    )

    # Return the content of the first choice's message
    return chat_completion.choices[0].message.content



def proofread_by_model_name(model_name, input_text, normalize_output):  
    """  
    Proofreads the given input text using the specified model.  
  
    Args:  
        model_name (str): The name of the model to use for proofreading.  
        input_text (str): The text to be proofread.  
        normalize_output (bool): Whether to normalize the output or not.  
  
    Returns:  
        str: The proofread text.  
    """  
    # Constants for API access  
    base_url = TOGETHER_PATH  
    api_key = TOGETHER_API_KEY  
    temperature = TEMPERATURE
      
    # Retrieve the model path from the dictionary  
    if model_name in MODEL_PATHS:  
        model_path = MODEL_PATHS[model_name]  
    else:  
        raise ValueError("Model name not found in the dictionary.")  
      
    # Formulate the prompt for the model  
    prompt = f"Proofreading for the text: ```{input_text}```"  
      
    # Apply output normalization if required  
    if normalize_output:  
        prompt = output_normalization(prompt)  
      
    # Debugging: Print the prompt  
    print(f"Prompt: {prompt}")  
      
    # Call the abstract proofreading function with the prepared parameters  
    return abstract_proofread(model_path, temperature, base_url, api_key, prompt)  


def gemini_proofread(input_text, normalize_output):  
    """  
    Proofreads the given text using the GEMINI_MODEL.  
      
    Parameters:  
    input_text (str): The text to be proofread.  
    normalize_output (bool): Flag indicating whether to normalize the output.  
  
    Returns:  
    str: The proofread text.  
    """  
    prompt = f"Proofreading for the text: ```{input_text}```"  
    if normalize_output:  
        prompt = output_normalization(prompt)  
    response = GEMINI_MODEL.generate_content(prompt)  
    return response.text  
  
def print_and_log(message):  
    """  
    Prints and logs the given message to a log file.  
      
    Parameters:  
    message (str): The message to be printed and logged.  
    """  
    print(message)  
    with open(LOG_FILE, "a+", encoding='utf-8') as log_file:  
        log_file.write(message + "\n")  
  
def write_to_file(filename, content):  
    """  
    Writes the given content to a specified file.  
      
    Parameters:  
    filename (str): The name of the file to write to.  
    content (str): The content to be written.  
    """  
    print(content)  
    with open(filename, "a+", encoding='utf-8') as file:  
        file.write(content)  
  
def output_normalization(prompt):  
    """  
    Normalizes the output by appending a specific instruction to the prompt.  
      
    Parameters:  
    prompt (str): The initial prompt.  
  
    Returns:  
    str: The modified prompt.  
    """  
    return prompt + " Please only output the proofread text without any explanation."  
  
def chatGPT_proofread(input_text, normalize_output):  
    """  
    Proofreads the given text using the chat_model.  
      
    Parameters:  
    input_text (str): The text to be proofread.  
    normalize_output (bool): Flag indicating whether to normalize the output.  
  
    Returns:  
    str: The proofread text.  
    """  
    prompt = f"Proofreading for the text: ```{input_text}```"  
    if normalize_output:  
        prompt = output_normalization(prompt)  
      
    print(f"Starting API call with prompt: {prompt}")  
    result = chat_model.predict(prompt)  
    print(f"Ending API call with prompt: {prompt}")  
      
    return result  
  
def normalize_text(input_text):  
    """  
    Normalizes the given text by removing certain characters and extra spaces.  
      
    Parameters:  
    input_text (str): The text to be normalized.  
  
    Returns:  
    str: The normalized text.  
    """  
    result = input_text.strip()  
    result = result.replace("**", "")  
    result = result.replace("\n", " ")  
    result = result.replace("  ", " ")  # Remove extra spaces  
    return result  
  
def write_to_csv(filename, row_data):  
    """  
    Writes a row of data to a specified CSV file.  
      
    Parameters:  
    filename (str): The name of the CSV file.  
    row_data (list): The row data to be written.  
    """  
    with open(filename, 'a+', encoding='UTF8', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(row_data)  
  
def number_of_csv_lines(filename):  
    """  
    Returns the number of lines in a specified CSV file.  
      
    Parameters:  
    filename (str): The name of the CSV file.  
  
    Returns:  
    int: The number of lines in the CSV file.  
    """  
    file_data = pd.read_csv(filename, sep=',').values  
    return len(file_data)  
  
def read_csv_data(input_file):  
    """  
    Reads data from a specified CSV file.  
      
    Parameters:  
    input_file (str): The name of the CSV file.  
  
    Returns:  
    numpy.ndarray: The data read from the CSV file.  
    """  
    file_data = pd.read_csv(input_file, dtype='string', keep_default_na=False, sep=',').values  
    return file_data  
  
def bart_score(text_1, text_2):  
    """  
    Computes the BART score between two texts.  
      
    Parameters:  
    text_1 (str): The first text.  
    text_2 (str): The second text.  
  
    Returns:  
    float: The BART score.  
    """  
    score = bart_scorer.score([text_1], [text_2])  
    return score  
  
def check_bart_score(input_text, raw_text):  
    """  
    Checks if the BART score between input_text and raw_text is above a threshold.  
      
    Parameters:  
    input_text (str): The input text.  
    raw_text (str): The raw text to compare against.  
  
    Returns:  
    bool: True if the score is above the threshold, False otherwise.  
    """  
    THRESHOLD = -2.459  
    normalized_text = normalize_text(raw_text)  
    score = bart_score(input_text, normalized_text)[0]  
    return score >= THRESHOLD  
  
def get_column(input_file, column_name):  
    """  
    Retrieves a specific column from a CSV file.  
      
    Parameters:  
    input_file (str): The name of the CSV file.  
    column_name (str): The name of the column to retrieve.  
  
    Returns:  
    numpy.ndarray: The values from the specified column.  
    """  
    df = pd.read_csv(input_file, dtype='string', keep_default_na=False, sep=',')  
    column_data = df[column_name]  
    return column_data.values  
  
def generate_column_names(categories):  
    """  
    Generates a list of column names based on given categories.  
      
    Parameters:  
    categories (list): The list of categories.  
  
    Returns:  
    list: The generated list of column names.  
    """  
    column_names = ['human']  
    for name in categories:  
        column_names.append(name)  
    for first in categories:  
        for second in categories:  
            column_names.append(f"{first}_{second}")  
    return column_names  
  
def write_new_data(output_file, current_data, column_names):  
    """  
    Writes new data to a CSV file based on current data and column names.  
      
    Parameters:  
    output_file (str): The name of the output CSV file.  
    current_data (dict): The current data to be written.  
    column_names (list): The list of column names.  
    """  
    data_row = [current_data[column] for column in column_names]  
    write_to_csv(output_file, data_row)  

def refine(input_text, candidate):  
    """  
    Refines the candidate string by removing specific surrounding marks if they are present  
    in the input_text with a count difference of exactly 2.  
      
    Args:  
        input_text (str): The original text.  
        candidate (str): The candidate text to be refined.  
  
    Returns:  
        str: The refined candidate text.  
    """  
  
    # Create a copy of the candidate string and strip whitespace  
    refined_candidate = candidate.strip()  
  
    # List of marks to check and potentially remove  
    marks = ["```", "'", '"']  
  
    # Iterate through each mark  
    for mark in marks:  
        # Count occurrences of the mark in input_text and refined_candidate  
        count_input_text = input_text.count(mark)  
        count_refined_candidate = refined_candidate.count(mark)  
  
        # Check if the mark should be stripped  
        if (count_refined_candidate == count_input_text + 2 and  
                refined_candidate.startswith(mark) and  
                refined_candidate.endswith(mark)):  
            # Strip the mark from both ends of the refined_candidate  
            refined_candidate = refined_candidate.strip(mark)  
  
    return refined_candidate  


def extract_by_best_similarity(input_text, raw_text):  
    """  
    Extracts the best candidate string from the raw text based on the highest similarity score  
    compared to the input text. The similarity score is calculated using the BART score.  
  
    Args:  
        input_text (str): The original text.  
        raw_text (str): The raw text containing multiple candidate strings.  
  
    Returns:  
        str: The best candidate string with the highest similarity score.   
             Returns the input text if no suitable candidate is found.  
    """  
  
    # Refine the raw text  
    refined_raw_text = refine(input_text, raw_text)  
  
    # Tokenize the refined raw text into sentences  
    raw_candidates = nltk.sent_tokenize(refined_raw_text)  
  
    # Split sentences further by newlines to get individual candidates  
    candidate_list = []  
    for sentence in raw_candidates:  
        candidate_list.extend(sentence.split("\n"))  
  
    # Initialize variables to track the best similarity score and the best candidate  
    best_similarity = -9999  
    best_candidate = ""  
  
    # Iterate over each candidate to find the best one based on the BART score  
    for candidate in candidate_list:  
        refined_candidate = refine(input_text, candidate)  
        if check_bart_score(input_text, refined_candidate):  
            score = bart_score(input_text, refined_candidate)[0]  
            if score > best_similarity:  
                best_similarity = score  
                best_candidate = refined_candidate  
  
    # Print the best candidate found  
    print(f"best_candidate = {best_candidate}")  
  
    # Return the best candidate if found, otherwise return the input text  
    if best_candidate == "":  
        return input_text  
    return best_candidate  

def proofread_with_best_similarity(input_text, model_kind):  
    """  
    Proofreads the input text using the specified model and extracts the best-corrected text based on similarity.  
  
    Args:  
        input_text (str): The original text to be proofread.  
        model_kind (str): The kind of model to use for proofreading (e.g., CHATGPT, GEMINI).  
  
    Returns:  
        tuple: A tuple containing the raw proofread text and the best-corrected text.  
    """  
      
    # Normalize the input text  
    normalized_input_text = normalize_text(input_text)  
    print_and_log(f"INPUT = {normalized_input_text}")  
      
    result_text = ""  
    raw_text = ""  
      
    for i in range(1):  # Loop is redundant as it runs only once; consider removing if unnecessary  
        # Select the proofreading model based on model_kind  
        if model_kind == CHATGPT:  
            raw_text = chatGPT_proofread(normalized_input_text, normalize_output=IS_OUTPUT_NORMALIZATION)  
        elif model_kind == GEMINI:  
            raw_text = gemini_proofread(normalized_input_text, normalize_output=IS_OUTPUT_NORMALIZATION)  
        else:  
            raw_text = proofread_by_model_name(model_kind, normalized_input_text, normalize_output=IS_OUTPUT_NORMALIZATION)  
          
        # Extract the best candidate text based on similarity  
        result_text = extract_by_best_similarity(normalized_input_text, raw_text)  
          
        # Log the raw and result texts  
        print_and_log(f"RAW_{i} = {raw_text}")  
        print_and_log(f"RESULT_{i} = {result_text}")  
          
        # Normalize the result text  
        result_text = normalize_text(result_text)  
          
        # If a valid result is obtained, return it  
        if result_text != "":  
            return raw_text, result_text  
      
    # Return the raw and result texts  
    return raw_text, result_text  

def generate_file_name(existing_data_file, existing_kinds, new_kinds):  
    """  
    Generates a new file name based on the path of an existing data file and a combination of existing and new kinds.  
  
    Args:  
        existing_data_file (str): The path to the existing data file.  
        existing_kinds (list): A list of existing kinds.  
        new_kinds (list): A list of new kinds.  
  
    Returns:  
        str: The generated file name with the full path.  
    """  
      
    # Combine existing and new kinds into a single list  
    combined_kinds = existing_kinds + new_kinds  
      
    # Get the directory path of the existing data file  
    directory_path = os.path.dirname(existing_data_file)  
      
    # Create a new file name by joining the kinds with underscores and adding a suffix  
    new_file_name = "_".join(combined_kinds) + "_with_best_similarity.csv"  
      
    # Combine the directory path with the new file name to get the full output file path  
    output_file_path = os.path.join(directory_path, new_file_name)  
      
    return output_file_path  


  
def generate_new_data_with_best_similarity(existing_data_file, existing_kinds, new_kinds):  
    """  
    Generates new data with the best similarity based on existing and new kinds, and writes the results to a CSV file.  
  
    Args:  
        existing_data_file (str): The path to the existing data file.  
        existing_kinds (list): A list of existing kinds.  
        new_kinds (list): A list of new kinds.  
  
    Returns:  
        None  
    """  
      
    # Combine existing and new kinds into a single list  
    all_kinds = existing_kinds + new_kinds  
      
    # Generate column names for the CSV file  
    column_names = generate_column_names(all_kinds)  
      
    # Generate column names for existing kinds  
    existing_column_names = generate_column_names(existing_kinds)  
      
    # Generate the output file name  
    output_file = generate_file_name(existing_data_file, existing_kinds, new_kinds)  
      
    # Create the output file with column names if it doesn't exist  
    if not os.path.exists(output_file):  
        write_to_csv(output_file, column_names)  
      
    # Read existing data from the file  
    existing_data = {kind: get_column(existing_data_file, kind) for kind in existing_column_names}  
      
    # Read input data from the output file  
    input_data = read_csv_data(output_file)  
    start_index = len(input_data)  
    print(f"start_index = {start_index}")  
      
    num_rows = len(existing_data["human"])  
    global_generate_set = []  
    global_reuse = []  
      
    for index in range(start_index, num_rows):  
        # Initialize generation and reuse sets  
        generate_set = []  
        reuse_set = []  
          
        # Prepare the current generation dictionary  
        current_generation = {kind: existing_data[kind][index] for kind in existing_column_names}  
        print(f"current_generation before generation = {current_generation}")  
          
        human_text = current_generation["human"]  
          
        # Generate new kinds based on human text  
        for kind in new_kinds:  
            _, generated_text = proofread_with_best_similarity(human_text, kind)  
            current_generation[kind] = generated_text  
            generate_set.append(kind)  
          
        print(f"current_generation after generate one = {current_generation}")  
          
        # Generate combinations of kinds  
        for first_kind in all_kinds:  
            for second_kind in all_kinds:  
                combination_name = f"{first_kind}_{second_kind}"  
                  
                if combination_name not in current_generation:  
                    if first_kind in current_generation and current_generation[first_kind] == human_text:  
                        generated_text = current_generation[second_kind]  
                        reuse_set.append(f"{combination_name} from {second_kind}")  
                    else:  
                        is_need_generation = True  
                        for first_kind_2 in all_kinds:  
                            if first_kind != first_kind_2 and current_generation[first_kind] == current_generation[first_kind_2]:  
                                combination_name_2 = f"{first_kind_2}_{second_kind}"  
                                if combination_name_2 in current_generation:  
                                    generated_text = current_generation[combination_name_2]  
                                    reuse_set.append(f"{combination_name} from {combination_name_2}")  
                                    is_need_generation = False  
                                    break  
                        if is_need_generation:  
                            _, generated_text = proofread_with_best_similarity(current_generation[first_kind], second_kind)  
                            generate_set.append(f"{first_kind}_{second_kind}")  
                      
                    current_generation[combination_name] = generated_text  
          
        # Write the current generation to the output file  
        write_new_data(output_file, current_generation, column_names)  
          
        # Update global sets  
        global_generate_set.append(generate_set)  
        global_reuse

def shuffle(array, seed):  
    """  
    Shuffles the elements of each sublist in the given array using the specified seed.  
  
    Args:  
        array (list of lists): The array containing sublists to shuffle.  
        seed (int): The seed value for the random number generator.  
  
    Returns:  
        None  
    """  
    for sublist in array:  
        random.Random(seed).shuffle(sublist)  
  
def generate_human_with_shuffle(dataset_name, column_name, num_samples, output_file):  
    """  
    Generates a shuffled list of sentences from the dataset and writes them to a CSV file.  
  
    Args:  
        dataset_name (str): The name of the dataset to load.  
        column_name (str): The column name to extract sentences from.  
        num_samples (int): The number of samples to process.  
        output_file (str): The path to the output CSV file.  
  
    Returns:  
        None  
    """  
    # Load the dataset  
    dataset = load_dataset(dataset_name)  
    data = dataset['train']  
      
    lines = []  
    # Tokenize sentences and add to the lines list  
    for sample in data:  
        nltk_tokens = nltk.sent_tokenize(sample[column_name])  
        lines.extend(nltk_tokens)  
      
    # Filter out empty lines  
    filtered_lines = [line for line in lines if line != ""]  
    lines = filtered_lines  
      
    # Shuffle the lines  
    shuffle([lines], seed=SEED)  
      
    # Ensure the output file exists and write the header if it doesn't  
    if not os.path.exists(output_file):  
        header = ["human"]  
        write_to_csv(output_file, header)  
      
    # Get the number of lines already processed in the output file  
    number_of_processed_lines = number_of_csv_lines(output_file)  
      
    # Print the initial lines to be processed  
    print(f"Lines before processing: {lines[:num_samples]}")  
      
    # Slice the lines list to get the unprocessed lines  
    lines = lines[number_of_processed_lines:num_samples]  
      
    # Print the lines after slicing  
    print(f"Lines after slicing: {lines}")  
      
    # Process each line and write to the output file  
    for index, human in enumerate(lines):  
        normalized_text = normalize_text(human)  
        output_data = [normalized_text]  
        write_to_csv(output_file, output_data)  
        print(f"Processed {index + 1} / {len(lines)}; Total processed: {number_of_processed_lines + index + 1} / {num_samples}")  


def split(data, ratio):  
    """  
    Splits the data into training and testing sets based on the given ratio.  
  
    Args:  
        data (list): The dataset to split.  
        ratio (float): The ratio for splitting the data into training and testing sets.  
  
    Returns:  
        tuple: A tuple containing the training data and the testing data.  
    """  
    train_size = int(len(data) * ratio)  
    train_data = data[:train_size]  
    test_data = data[train_size:]  
    return train_data, test_data  
  
def bart_score_in_batch(text_1, text_2):  
    """  
    Calculates the BART score for pairs of texts in batches.  
  
    Args:  
        text_1 (list of str): The first list of texts.  
        text_2 (list of str): The second list of texts.  
  
    Returns:  
        list: A list of BART scores for each pair of texts.  
    """  
    return bart_scorer.score(text_1, text_2, batch_size=BATCH_SIZE)  
  
def extract_feature_in_batch(text_1, text_2, feature_kind):  
    """  
    Extracts features for pairs of texts using BART scores.  
  
    Args:  
        text_1 (list of str): The first list of texts.  
        text_2 (list of str): The second list of texts.  
        feature_kind (str): The type of feature to extract.  
  
    Returns:  
        list: A list of extracted features.  
    """  
    features = bart_score_in_batch(text_1, text_2)  
    return features  
  
def abstract_train(features, labels):  
    """  
    Trains a model using the given features and labels.  
  
    Args:  
        features (list): The input features for training.  
        labels (list): The target labels for training.  
  
    Returns:  
        object: The trained model.  
    """  
    model = MLPClassifier()  
    model.fit(features, labels)  
    return model  
  
def evaluate_model(model, features, labels):  
    """  
    Evaluates the model's performance using accuracy and ROC AUC scores.  
  
    Args:  
        model (object): The trained model to evaluate.  
        features (list): The input features for evaluation.  
        labels (list): The target labels for evaluation.  
  
    Returns:  
        None  
    """  
    predictions = model.predict(features)  
    rounded_predictions = [round(value) for value in predictions]  
      
    accuracy = accuracy_score(labels, rounded_predictions)  
    write_to_file(OUTPUT_FILE, f"Accuracy: {accuracy * 100.0:.1f}%\n")  
      
    roc_auc = roc_auc_score(labels, rounded_predictions)  
    write_to_file(OUTPUT_FILE, f"ROC AUC: {roc_auc * 100.0:.1f}%\n")  
  
def combine_text_with_BERT_format(text_list):  
    """  
    Combines a list of texts into a single string formatted for BERT input.  
  
    Args:  
        text_list (list of str): The list of texts to combine.  
  
    Returns:  
        str: The combined text string formatted for BERT input.  
    """  
    combined_text = f"<s>{text_list[0]}</s>"  
    for i in range(1, len(text_list)):  
        combined_text += f"</s>{text_list[i]}</s>"  
    return combined_text  


def preprocess_function_multimodel(sample):  
    """  
    Preprocesses a given sample for a multi-model setup by calculating BART scores  
    and formatting the text for BERT input.  
  
    Args:  
        sample (dict): A dictionary containing a key "text", which is a list of lists of strings.  
  
    Returns:  
        dict: A dictionary containing tokenized and preprocessed text data.  
    """  
    num_texts = len(sample["text"][0])  # Number of texts in each sub-sample  
    texts_grouped_by_index = [[] for _ in range(num_texts)]  # Initialize empty lists for grouping texts by index  
  
    # Group texts by their index across sub-samples  
    for sub_sample in sample["text"]:  
        for i in range(num_texts):  
            texts_grouped_by_index[i].append(sub_sample[i])  
  
    # Calculate BART scores for each text pair (text[0] with text[i])  
    bart_scores = [bart_score_in_batch(texts_grouped_by_index[0], texts_grouped_by_index[i]) for i in range(1, num_texts)]  
  
    combined_texts = []  
  
    # Process each sub-sample for BERT input  
    for index, sub_sample in enumerate(sample["text"]):  
        text_array = [sub_sample[0]]  # Start with the input text  
        score_generation_pairs = []  
  
        # Pair scores with their corresponding generations  
        for i in range(1, num_texts):  
            generation_text = sub_sample[i]  
            generation_score = bart_scores[i-1][index]  
            score_generation_pairs.append((generation_score, generation_text))  
  
        # Sort pairs by score in descending order  
        sorted_pairs = sorted(score_generation_pairs, reverse=True)  
  
        # Append sorted texts to text_array  
        for _, sorted_text in sorted_pairs:  
            text_array.append(sorted_text)  
  
        # Combine texts into a single BERT-formatted string  
        combined_text = combine_text_with_BERT_format(text_array)  
        combined_texts.append(combined_text)  
  
    # Tokenize the combined texts for BERT  
    return tokenizer(combined_texts, add_special_tokens=False, truncation=True)  

def preprocess_function_single_from_multimodel(sample):  
    """  
    Extracts the first text from each sub-sample in a multi-model sample and tokenizes it.  
  
    Args:  
        sample (dict): A dictionary containing a key "text", which is a list of lists of strings.  
  
    Returns:  
        dict: A dictionary containing tokenized text data.  
    """  
    combined_texts = []  
  
    # Iterate through each sub-sample  
    for sub_sample in sample["text"]:  
        input_text = sub_sample[0]  # Extract the first text from the sub-sample  
        combined_texts.append(input_text)  # Append it to the list of combined texts  
  
    # Tokenize the combined texts  
    return tokenizer(combined_texts, truncation=True)  
  
  
def check_api_error(data):  
    """  
    Checks if any item in the provided data indicates an API error.  
  
    Args:  
        data (list): A list of items to be checked for API errors.  
  
    Returns:  
        bool: True if an API error or ignore by API error is found, otherwise False.  
    """  
    for item in data:  
        if item == API_ERROR or item == IGNORE_BY_API_ERROR:  # Check for API error indicators  
            return True  # Return True if an error indicator is found  
    return False  # Return False if no error indicators are found  


def train_only_by_transformer_with_test_evaluation_early_stop(train_data, test_data, input_type, num_classes=2):  
    """  
    Trains a transformer model using the provided training and testing datasets with early stopping.  
  
    Args:  
        train_data (Dataset): The training dataset.  
        test_data (Dataset): The testing dataset.  
        input_type (str): The type of input data, either MULTIMODEL or SINGLE_FROM_MULTIMODEL.  
        num_classes (int, optional): The number of classes for classification. Defaults to 2.  
  
    Returns:  
        Trainer: The trained model wrapped in a Trainer object.  
    """  
    # Preprocess datasets based on the input type  
    if input_type == MULTIMODEL:  
        train_data = train_data.map(preprocess_function_multimodel, batched=True)  
        test_data = test_data.map(preprocess_function_multimodel, batched=True)  
    elif input_type == SINGLE_FROM_MULTIMODEL:  
        train_data = train_data.map(preprocess_function_single_from_multimodel, batched=True)  
        test_data = test_data.map(preprocess_function_single_from_multimodel, batched=True)  
      
    # Data collator to pad inputs  
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  
  
    # Load appropriate model based on number of classes  
    if num_classes == 3:  
        model = AutoModelForSequenceClassification.from_pretrained(  
            "pretrained_model/roberta-base_num_labels_3", num_labels=num_classes)  
    else:  
        model = AutoModelForSequenceClassification.from_pretrained(  
            ROBERTA_MODEL_PATHS[MODEL_NAME], num_labels=num_classes)  
      
    learning_rate = LEARNING_RATES[MODEL_NAME]  
    output_folder = "training_with_callbacks"  
  
    # Remove the output folder if it already exists  
    if os.path.exists(output_folder):  
        shutil.rmtree(output_folder)  
      
    # Training arguments  
    training_args = TrainingArguments(  
        output_dir=output_folder,  
        evaluation_strategy="epoch",  
        logging_strategy="epoch",  
        save_strategy="epoch",  
        learning_rate=learning_rate,  
        per_device_train_batch_size=BATCH_SIZE,  
        per_device_eval_batch_size=BATCH_SIZE,  
        num_train_epochs=NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING,  
        weight_decay=0.01,  
        push_to_hub=False,  
        metric_for_best_model=OPTIMIZED_METRIC,  
        load_best_model_at_end=True  
    )  
  
    # Create Trainer object  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_data,  
        eval_dataset=test_data,  
        tokenizer=tokenizer,  
        data_collator=data_collator,  
        compute_metrics=compute_metrics,  
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]  
    )  
  
    # Add custom callback  
    trainer.add_callback(CustomCallback(trainer))  
  
    # Start training  
    trainer.train()  
  
    return trainer  


def calculate_number_of_models(num_columns):  
    """  
    Calculates the number of models required based on the number of columns.  
  
    Args:  
        num_columns (int): The total number of columns.  
  
    Returns:  
        int: The number of models required.  
  
    Raises:  
        Exception: If the number of models cannot be calculated to match the number of columns.  
    """  
    num_models = 0  
    count_human = 1  # Initial count representing human input  
  
    while True:  
        count_single = num_models  # Single model count  
        count_pair = num_models * num_models  # Pair model count  
  
        total_count = count_human + count_single + count_pair  
  
        if total_count == num_columns:  
            return num_models  
        elif total_count > num_columns:  
            raise Exception("Cannot calculate the number of models to match the number of columns")  
  
        num_models += 1  


def read_multimodel_data_from_csv(multimodel_csv_file):  
    """  
    Reads multimodel data from a CSV file and organizes it into a structured format.  
  
    Args:  
        multimodel_csv_file (str): Path to the CSV file containing multimodel data.  
  
    Returns:  
        list: A list of dictionaries, each containing 'human', 'single', and 'pair' data.  
  
    Raises:  
        Exception: If there is an error in reading the CSV file or processing the data.  
    """  
    # Read CSV data into a list of lists  
    input_data = read_csv_data(multimodel_csv_file)  
      
    # Initialize the result list  
    structured_data = []  
      
    # Calculate the number of models based on the number of columns in the first row  
    num_models = calculate_number_of_models(len(input_data[0]))  
      
    # Process each row in the input data  
    for row in input_data:  
        row_data = {}  
        index = 0  
          
        # Extract human data  
        row_data["human"] = row[index]  
        index += 1  
          
        # Extract single model data  
        single_model_data = []  
        for _ in range(num_models):  
            single_model_data.append(row[index])  
            index += 1  
        row_data["single"] = single_model_data  
          
        # Extract pair model data  
        pair_model_data = []  
        for _ in range(num_models):  
            sub_pair_data = []  
            for _ in range(num_models):  
                sub_pair_data.append(row[index])  
                index += 1  
            pair_model_data.append(sub_pair_data)  
        row_data["pair"] = pair_model_data  
          
        # Append the structured row data to the result list  
        structured_data.append(row_data)  
      
    return structured_data  


def check_error(data_item):  
    """  
    Checks for errors in a data item by verifying the 'human', 'single', and 'pair' fields.  
      
    Args:  
        data_item (dict): A dictionary containing 'human', 'single', and 'pair' data.  
          
    Returns:  
        bool: True if any of the fields contain an error, otherwise False.  
    """  
    # Check for API error in the 'human' field  
    if check_api_error(data_item["human"]):  
        return True  
      
    # Check for API error in the 'single' model data  
    for single_text in data_item["single"]:  
        if check_api_error(single_text):  
            return True  
      
    # Get the number of models from the 'single' model data  
    num_models = len(data_item["single"])  
      
    # Check for API error in the 'pair' model data  
    for i in range(num_models):  
        for j in range(num_models):  
            if check_api_error(data_item["pair"][i][j]):  
                return True  
      
    # No errors found  
    return False  



def create_pair_sample(data_item, training_indices):  
    """  
    Creates pair samples for training by comparing human data with machine-generated data.  
      
    Args:  
        data_item (dict): A dictionary containing 'human', 'single', and 'pair' data.  
        training_indices (list): A list of indices used for training.  
          
    Returns:  
        list: A list of dictionaries, each containing a 'text' array and a 'label'.  
    """  
    # Initialize the result list  
    result_samples = []  
      
    # Check if there is any error in the data_item  
    if check_error(data_item):  
        return result_samples  
      
    # Create machine samples  
    for train_idx in training_indices:  
        if data_item["human"] != data_item["single"][train_idx]:  
            text_array = []  
            machine_text = data_item["single"][train_idx]  
            text_array.append(machine_text)  
              
            for sub_idx in training_indices:  
                text_array.append(data_item["pair"][train_idx][sub_idx])  
              
            sample = {  
                "text": text_array,  
                "label": MACHINE_LABEL  
            }  
            result_samples.append(sample)  
      
    # Create human samples  
    text_array = [data_item["human"]]  
      
    for train_idx in training_indices:  
        text_array.append(data_item["single"][train_idx])  
      
    human_sample = {  
        "text": text_array,  
        "label": HUMAN_LABEL  
    }  
      
    # Append human samples for each machine sample  
    num_machine_samples = len(result_samples)  
    for _ in range(num_machine_samples):  
        result_samples.append(human_sample)  
      
    return result_samples  


def create_pair_test_sample(data_item, training_indices, testing_indices):  
    """  
    Creates pair test samples by comparing human data with machine-generated data.  
      
    Args:  
        data_item (dict): A dictionary containing 'human', 'single', and 'pair' data.  
        training_indices (list): A list of indices used for training.  
        testing_indices (list): A list of indices used for testing.  
          
    Returns:  
        list: A list of dictionaries, each containing a 'text' array and a 'label'.  
    """  
    # Initialize the result list  
    result_samples = []  
      
    # Check if there is any error in the data_item  
    if check_error(data_item):  
        return result_samples  
      
    # Create machine samples based on testing indices  
    for test_idx in testing_indices:  
        if data_item["human"] != data_item["single"][test_idx]:  
            text_array = []  
            machine_text = data_item["single"][test_idx]  
            text_array.append(machine_text)  
              
            for train_idx in training_indices:  
                text_array.append(data_item["pair"][test_idx][train_idx])  
              
            sample = {  
                "text": text_array,  
                "label": MACHINE_LABEL  
            }  
            result_samples.append(sample)  
      
    # Create human sample  
    text_array = [data_item["human"]]  
      
    for train_idx in training_indices:  
        text_array.append(data_item["single"][train_idx])  
      
    human_sample = {  
        "text": text_array,  
        "label": HUMAN_LABEL  
    }  
      
    # Append the human sample for each machine sample  
    num_machine_samples = len(result_samples)  
    for _ in range(num_machine_samples):  
        result_samples.append(human_sample)  
      
    return result_samples  



def create_train_val_sample(data, training_indices):  
    """  
    Creates training and validation samples from the provided data.  
      
    Args:  
        data (list): A list of data items, each to be processed.  
        training_indices (list): A list of indices used for training.  
          
    Returns:  
        list: A list of training and validation samples created from the data.  
    """  
    # Initialize the result list  
    result_samples = []  
      
    # Process each item in the data  
    for data_item in data:  
        # Create pair samples for the current item  
        sub_samples = create_pair_sample(data_item, training_indices)  
          
        # Extend the result list with the created sub-samples  
        result_samples.extend(sub_samples)  
      
    return result_samples  
  
  
def create_test_sample(data, training_indices, testing_indices):  
    """  
    Creates test samples from the provided data by comparing human data with machine-generated data.  
      
    Args:  
        data (list): A list of data items, each to be processed.  
        training_indices (list): A list of indices used for training.  
        testing_indices (list): A list of indices used for testing.  
          
    Returns:  
        list: A list of test samples created from the data.  
    """  
    # Initialize the result list  
    result_samples = []  
      
    # Process each item in the data  
    for data_item in data:  
        # Create pair test samples for the current item  
        sub_samples = create_pair_test_sample(data_item, training_indices, testing_indices)  
          
        # Extend the result list with the created sub-samples  
        result_samples.extend(sub_samples)  
      
    return result_samples  


def distribute_data(data, train_indices, test_indices, train_ratio, val_ratio):  
    """  
    Distributes the data into training, validation, and test samples.  
      
    Args:  
        data (list): A list of data items to be split and processed.  
        train_indices (list): A list of indices used for training.  
        test_indices (list): A list of indices used for testing.  
        train_ratio (float): The ratio of data to be used for training.  
        val_ratio (float): The ratio of data to be used for validation.  
          
    Returns:  
        tuple: A tuple containing lists of training, validation, and test samples.  
    """  
    # Split the data into training, validation, and test sets  
    train_data, val_data, test_data = split_train_val_test(data, train_ratio, val_ratio)  
  
    # Create training samples  
    train_samples = create_train_val_sample(train_data, train_indices)  
    write_to_file(OUTPUT_FILE, f"train samples = {len(train_samples)}\n")  
  
    # Create validation samples  
    val_samples = create_train_val_sample(val_data, train_indices)  
    write_to_file(OUTPUT_FILE, f"val samples = {len(val_samples)}\n")  
  
    # Create test samples  
    test_samples = create_test_sample(test_data, train_indices, test_indices)  
    write_to_file(OUTPUT_FILE, f"test samples = {len(test_samples)}\n")  
  
    return train_samples, val_samples, test_samples  
  
  
def convert_to_huggingface_with_multimodel(samples):  
    """  
    Converts a list of samples to the Hugging Face Dataset format.  
      
    Args:  
        samples (list): A list of samples to be converted.  
          
    Returns:  
        Dataset: A Hugging Face Dataset object created from the samples.  
    """  
    return Dataset.from_list(samples)  



def train_by_transformer_with_multimodel_and_early_stop(train_samples, val_samples, input_type):  
    """  
    Trains a transformer model with multimodal data and early stopping.  
      
    Args:  
        train_samples (list): A list of training samples.  
        val_samples (list): A list of validation samples.  
        input_type (str): The type of input data (e.g., multimodal).  
          
    Returns:  
        object: The trained model with early stopping.  
    """  
    # Convert training and validation samples to Hugging Face Dataset format  
    train_data = convert_to_huggingface_with_multimodel(train_samples)  
    val_data = convert_to_huggingface_with_multimodel(val_samples)  
      
    # Train the model with early stopping and return the trained model  
    return train_only_by_transformer_with_test_evaluation_early_stop(train_data, val_data, input_type)  
  
  
def test_by_transformer_with_multimodel(detector, test_samples, input_type):  
    """  
    Tests a trained transformer model with multimodal data.  
      
    Args:  
        detector (object): The trained model to be evaluated.  
        test_samples (list): A list of test samples.  
        input_type (str): The type of input data (e.g., multimodal).  
          
    Returns:  
        None  
    """  
    # Convert test samples to Hugging Face Dataset format  
    test_data = convert_to_huggingface_with_multimodel(test_samples)  
      
    # Apply the appropriate preprocessing function based on the input type  
    if input_type == MULTIMODEL:  
        test_data = test_data.map(preprocess_function_multimodel, batched=True)  
    elif input_type == SINGLE_FROM_MULTIMODEL:  
        test_data = test_data.map(preprocess_function_single_from_multimodel, batched=True)  
      
    # Evaluate the model on the test data  
    result = detector.evaluate(eval_dataset=test_data)  
      
    # Extract and log the ROC AUC score  
    roc_auc = result['eval_roc_auc']  
    write_to_file(OUTPUT_FILE, "roc_auc: %.1f%%" % (roc_auc * 100.0) + "\n")  



def extract_by_feature_kind(samples, feature_type):  
    """  
    Extracts features from the given samples based on the specified feature type.  
  
    Args:  
        samples (list): A list of samples where each sample is a dictionary with 'text' and 'label' keys.  
        feature_type (str): The type of feature to extract.  
  
    Returns:  
        tuple: A tuple containing the extracted features and corresponding labels.  
    """  
    text_1_list = []  
    text_2_list = []  
    labels = []  
      
    for sample in samples:  
        text_1_list.append(sample["text"][0])  
        text_2_list.append(sample["text"][1])  
        labels.append(sample["label"])  
      
    # Extract features in batch based on the feature type  
    features = extract_feature_in_batch(text_1_list, text_2_list, feature_type)  
      
    return features, labels  
  
  
def train_by_feature_kind(train_samples, feature_type):  
    """  
    Trains a model using features extracted from the training samples based on the specified feature type.  
  
    Args:  
        train_samples (list): A list of training samples where each sample is a dictionary with 'text' and 'label' keys.  
        feature_type (str): The type of feature to extract for training.  
  
    Returns:  
        object: The trained model.  
    """  
    # Extract features and labels from the training samples  
    features, labels = extract_by_feature_kind(train_samples, feature_type)  
      
    # Convert features to a numpy array and reshape for training  
    features = np.array(features)  
    features = features.reshape(-1, 1)  
      
    # Train the model using the extracted features and labels  
    model = abstract_train(features, labels)  
      
    return model  


def test_by_feature_kind(detector, samples, feature_type):  
    """  
    Tests a detector using features extracted from the provided samples based on the specified feature type.  
  
    Args:  
        detector (object): The detector model to be evaluated.  
        samples (list): A list of samples where each sample is a dictionary with 'text' and 'label' keys.  
        feature_type (str): The type of feature to extract for testing.  
  
    Returns:  
        None  
    """  
    # Extract features and labels from the samples  
    features, labels = extract_by_feature_kind(samples, feature_type)  
      
    # Convert features to a numpy array and reshape for evaluation  
    features = np.array(features)  
    features = features.reshape(-1, 1)  
      
    # Evaluate the detector model using the extracted features and labels  
    evaluate_model(detector, features, labels)  


def general_process_multimodels_train_val_test(train_samples, val_samples, test_samples):  
    """  
    General process for training, validating, and testing models using multi-model and feature kind approaches.  
  
    Args:  
        train_samples (list): Training samples.  
        val_samples (list): Validation samples.  
        test_samples (list): Test samples.  
  
    Returns:  
        None  
    """  
    # Multi-model approach  
    input_kind = MULTIMODEL  
    write_to_file(OUTPUT_FILE, f"\nInput kind = {input_kind} \n")  
      
    # Train detector using multi-model with early stopping  
    detector = train_by_transformer_with_multimodel_and_early_stop(train_samples, val_samples, input_kind)  
      
    # Evaluate on train set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")  
    test_by_transformer_with_multimodel(detector, train_samples, input_kind)  
      
    # Evaluate on validation set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")  
    test_by_transformer_with_multimodel(detector, val_samples, input_kind)  
      
    # Evaluate on test set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")  
    test_by_transformer_with_multimodel(detector, test_samples, input_kind)  
      
    # Single from multi-model approach  
    input_kind = SINGLE_FROM_MULTIMODEL  
    write_to_file(OUTPUT_FILE, f"\nInput kind = {input_kind} \n")  
      
    # Train detector using single from multi-model with early stopping  
    detector = train_by_transformer_with_multimodel_and_early_stop(train_samples, val_samples, input_kind)  
      
    # Evaluate on train set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")  
    test_by_transformer_with_multimodel(detector, train_samples, input_kind)  
      
    # Evaluate on validation set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")  
    test_by_transformer_with_multimodel(detector, val_samples, input_kind)  
      
    # Evaluate on test set  
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")  
    test_by_transformer_with_multimodel(detector, test_samples, input_kind)  
      
    # Feature kind approach  
    sample_length = len(train_samples[0]["text"])  
    if sample_length == 2:  # Check if the sample length is 2, indicating BART feature kind  
        feature_kind = BART  
        write_to_file(OUTPUT_FILE, f"\nFeature kind = {feature_kind} \n")  
          
        # Train detector using feature kind  
        detector = train_by_feature_kind(train_samples, feature_kind)  
          
        # Evaluate on train set  
        write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")  
        test_by_feature_kind(detector, train_samples, feature_kind)  
          
        # Evaluate on validation set  
        write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")  
        test_by_feature_kind(detector, val_samples, feature_kind)  
          
        # Evaluate on test set  
        write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")  
        test_by_feature_kind(detector, test_samples, feature_kind)  


def process_multi_models_with_validation(multimodel_csv_file, train_indices, test_indices, num_samples):  
    """  
    Processes multi-model data with validation, training, and testing.  
  
    Args:  
        multimodel_csv_file (str): Path to the CSV file containing multi-model data.  
        train_indices (list): Indices for the training data.  
        test_indices (list): Indices for the testing data.  
        num_samples (int): Number of samples to process.  
  
    Returns:  
        None  
    """  
    # Log the details of the process  
    write_to_file(OUTPUT_FILE, f"PROCESSING FILE={multimodel_csv_file} \n")  
    write_to_file(OUTPUT_FILE, f"EXPERIMENT WITH {MODEL_NAME} model \n")  
    write_to_file(OUTPUT_FILE, f"NUMBER OF MAX EPOCHS WITH EARLY STOPPING = {NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING} \n")  
    write_to_file(OUTPUT_FILE, f"PATIENCE = {PATIENCE} \n")  
    write_to_file(OUTPUT_FILE, f"OPTIMIZED METRIC = {OPTIMIZED_METRIC} \n")  
    write_to_file(OUTPUT_FILE, f"BATCH SIZE = {BATCH_SIZE} \n")  
    write_to_file(OUTPUT_FILE, f"Number of samples = {num_samples} \n")  
  
    # Read multi-model data from the CSV file  
    data = read_multimodel_data_from_csv(multimodel_csv_file)  
      
    # Limit data to the specified number of samples  
    data = data[:num_samples]  
  
    # Distribute data into training, validation, and testing sets  
    train_samples, val_samples, test_samples = distribute_data(data, train_indices, test_indices, TRAIN_RATIO, VAL_RATIO)  
  
    # Log the training and testing indices  
    write_to_file(OUTPUT_FILE, f"Multimodel training with train indices {train_indices}, test with test indices {test_indices} \n")  
  
    # Process the multi-models for training, validation, and testing  
    general_process_multimodels_train_val_test(train_samples, val_samples, test_samples)  




def split_train_val_test(data, train_ratio, val_ratio):  
    """  
    Splits the dataset into training, validation, and test sets based on specified ratios.  
  
    Args:  
        data (list): The dataset to be split.  
        train_ratio (float): The ratio of the dataset to be used for training.  
        val_ratio (float): The ratio of the dataset to be used for validation.  
  
    Returns:  
        tuple: A tuple containing three lists - (train_data, val_data, test_data).  
    """  
    # Calculate the number of samples for the training set  
    num_train_samples = int(len(data) * train_ratio)  
      
    # Calculate the number of samples for the validation set  
    num_val_samples = int(len(data) * val_ratio)  
      
    # Split the data into training, validation, and test sets  
    train_data = data[:num_train_samples]  
    val_data = data[num_train_samples:(num_train_samples + num_val_samples)]  
    test_data = data[(num_train_samples + num_val_samples):]  
      
    return train_data, val_data, test_data  
 
  
def main():  
    """  
    Main function to handle argument parsing and execute the sequence of operations  
    including data generation and processing with multiple models.  
    """  
    parser = argparse.ArgumentParser(description='SimLLM.')  
  
    # Argument for specifying the list of large language models  
    parser.add_argument('--LLMs', nargs="+", default=[CHATGPT, "Yi", "OpenChat"],   
                        help='List of large language models')  
  
    # Argument for specifying the list of training indexes  
    parser.add_argument('--train_indexes', type=int, default=[0,1,2], nargs="+",   
                        help='List of training indexes')  
  
    # Argument for specifying the list of testing indexes  
    parser.add_argument('--test_indexes', type=int, default=[0], nargs="+",   
                        help='List of testing indexes')  
  
    # Argument for specifying the number of samples  
    parser.add_argument('--num_samples', type=int, default=5000,   
                        help='Number of samples')  


    # Parse the command-line arguments  
    args = parser.parse_args()  
  
    # Static dataset parameters  
    dataset_name = "xsum"  
    column_name = "document"  
    num_samples = args.num_samples  
    output_file = "data/human.csv"  
  
    # Generate human data with shuffle  
    # generate_human_with_shuffle(dataset_name, column_name, num_samples, output_file)  
  
    # Existing data parameters  
    existing_data_file = output_file  
    existing_kinds = []  
  
    # New kinds of models to generate data with  
    new_kinds = args.LLMs  
  
    # Generate new data with best similarity  
    generate_new_data_with_best_similarity(existing_data_file, existing_kinds, new_kinds)  
  
    # Generate a filename for the multimodel CSV file  
    multimodel_csv_file = generate_file_name(existing_data_file, existing_kinds, new_kinds)  
  
    # Number of samples to process (-1 means process all samples)  
    num_samples_to_process = -1  
  
    # Training and testing indexes from arguments  
    training_indexes = args.train_indexes  
    testing_indexes = args.test_indexes  
  
    # Process multiple models with validation  
    process_multi_models_with_validation(multimodel_csv_file, training_indexes, testing_indexes, num_samples_to_process)  
  
if __name__ == "__main__":  
    main()  
