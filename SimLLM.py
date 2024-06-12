TOGETHER_API_KEY = ""

OPENAI_KEY = ""

GEMINI_KEY = ""

import shutil
import random
import pandas as pd 
import os
import numpy as np
import nltk
import google.generativeai as genai
import csv 
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainerCallback, TrainingArguments, Trainer
from openai import OpenAI
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from os.path import join
from langchain.chat_models import ChatOpenAI
from datasets import load_metric, load_dataset, Dataset
from copy import deepcopy
from bart_score import BARTScorer
import argparse

bartscorer = BARTScorer(device='cuda:0',checkpoint="facebook/bart-large-cnn")

nltk.download('punkt')

RATIO = 0.9

os.environ['OPENAI_API_KEY'] = OPENAI_KEY 

IS_OUTPUT_NORMALIZATION = False

LOG_FILE = "data/99_log.txt"
chat_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125") 


CHATGPT = "ChatGPT"
GEMINI = "Gemini"
LLAMA_2_70_CHAT_TEMP_0 = "LLaMa"

API_ERROR = "API_ERROR"
IGNORE_BY_API_ERROR = "IGNORE_BY_API_ERROR"

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


genai.configure(api_key=GEMINI_KEY,transport='rest')


generation_config = {
  "temperature": 0.0,
}

GEMINI_MODEL = genai.GenerativeModel('gemini-pro', generation_config=generation_config)

LLAMA_2_70_CHAT_PATH = "meta-llama/Llama-2-70b-chat-hf" 
TOGETHER_PATH ='https://api.together.xyz'
api_key=TOGETHER_API_KEY

QWEN_1_5_72B_CHAT_PATH = "Qwen/Qwen1.5-72B-Chat" 
QWEN_1_5_72B_CHAT = "QWEN"


NOUS_HERMES_2_YI_34B_PATH = "NousResearch/Nous-Hermes-2-Yi-34B" 

NOUS_HERMES_2_YI_34B = "Yi"

MIXTRAL_8X7B_INSTRUCT_V0_1_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
MIXTRAL_8X7B_INSTRUCT_V0_1 = "Mixtral"

OLMO_7B_INSTRUCT_PATH = "allenai/OLMo-7B-Instruct" 
OLMO_7B_INSTRUCT = "OLMo"

PHI_2_PATH = "microsoft/phi-2" 
PHI_2 = "Phi"

OPENCHAT_3_5_1210_PATH = "openchat/openchat-3.5-1210" 
OPENCHAT_3_5_1210 = "OpenChat"

WIZARDLM_13B_V1_2_PATH = "WizardLM/WizardLM-13B-V1.2" 
WIZARDLM_13B_V1_2 = "WizardLM"

VICUNA_13B_V1_5_PATH = "lmsys/vicuna-13b-v1.5"  
VICUNA_13B_V1_5 = "Vicuna"

MODEL_DICT = {
    LLAMA_2_70_CHAT_TEMP_0:LLAMA_2_70_CHAT_PATH,
    QWEN_1_5_72B_CHAT:QWEN_1_5_72B_CHAT_PATH ,
    NOUS_HERMES_2_YI_34B:NOUS_HERMES_2_YI_34B_PATH,
    MIXTRAL_8X7B_INSTRUCT_V0_1:MIXTRAL_8X7B_INSTRUCT_V0_1_PATH,
    OLMO_7B_INSTRUCT:OLMO_7B_INSTRUCT_PATH,
    PHI_2:PHI_2_PATH,
    OPENCHAT_3_5_1210:OPENCHAT_3_5_1210_PATH,
    WIZARDLM_13B_V1_2:WIZARDLM_13B_V1_2_PATH,
    VICUNA_13B_V1_5:VICUNA_13B_V1_5_PATH
}

LLAMA_2_70_CHAT = "LLAMA_2_70_CHAT"

OUTPUT_FILE = "data/result.txt"
METRIC_NAME = "roc_auc"
metric = load_metric(METRIC_NAME)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING = 10
PATIENCE = 3
BATCH_SIZE = 64
OPTIMIZED_METRIC = "roc_auc"

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

SEED = 0
BART = "bart"
MULTIMODEL = "multimodel"
SINGLE_FROM_MULTIMODEL = "single_from_multimodel"

HUMAN_LABEL = 0
MACHINE_LABEL = 1

ROBERTA_BASE = "roberta-base"
ROBERTA_BASE_PATH = "roberta-base"
ROBERTA_LARGE_PATH = "roberta-large"
ROBERTA_LARGE = "roberta-large"
MODEL_PATH = {ROBERTA_BASE: ROBERTA_BASE_PATH, ROBERTA_LARGE: ROBERTA_LARGE_PATH}
LEARNING_RATE = {ROBERTA_BASE: 2e-5, ROBERTA_LARGE: 8e-6}

MODEL_NAME = ROBERTA_BASE

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[MODEL_NAME])


def compute_metrics(eval_pred):

   predictions, labels = eval_pred
   predictions = np.argmax(predictions, axis=1)

   return metric.compute(prediction_scores=predictions, references=labels, average="macro")


def abstract_proofread(model_path, temperature, base_url, api_key, prompt):    
    TOGETHER_CLIENT = OpenAI(api_key=api_key,base_url=base_url)
    chat_completion = TOGETHER_CLIENT.chat.completions.create(
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
    return chat_completion.choices[0].message.content

def proofread_by_model_name(model_name, input_text, is_output_normalization):
    base_url = TOGETHER_PATH
    api_key = TOGETHER_API_KEY
    temperature = 0
    if model_name in MODEL_DICT:
        model_path = MODEL_DICT[model_name]
    prompt = f"Proofreading for the text: ```{input_text}```"
    if is_output_normalization:
        prompt = output_normalization(prompt) 

    print(f"prompt = {prompt}")
    return abstract_proofread(model_path,  temperature, base_url, api_key, prompt) 


def gemini_proofread(input_text, is_output_normalization):
    prompt = f"Proofreading for the text: ```{input_text}```"
    if is_output_normalization:
        prompt = output_normalization(prompt)  
    response = GEMINI_MODEL.generate_content(prompt)
    return response.text

def print_and_log(text):
    print(text)
    with open(LOG_FILE, "a+", encoding = 'utf-8') as f:
        f.write(text + "\n")

def write_to_file(filename, text):
    print(text)
    with open(filename, "a+", encoding = 'utf-8') as f:
        f.write(text)


def output_normalization(prompt):
    return prompt + " Please only output the proofread text without any explanation."

def chatGPT_proofread(input_text, is_output_normalization):
    prompt = f"Proofreading for the text: ```{input_text}```"
    if is_output_normalization:
        prompt = output_normalization(prompt) 
    print(f"start call API with promt {prompt}")    
    result = chat_model.predict(prompt)
    print(f"end call API with promt {prompt}")    
    return result 

def normalize_text(input):
    result = input.strip()
    result = result.replace("**", "")
    result = result.replace("\n", " ")
    result = result.replace("  ", " ")
    result = result.replace("  ", " ")
    result = result.replace("  ", " ")
    result = result.replace("  ", " ")     
    return result

def write_to_csv(filename, row):
    with open(filename, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def number_of_csv_lines(filename):
    myFile = pd.read_csv(filename, sep=',').values
    return len(myFile)

def read_csv_data(input_file):
    myFile = pd.read_csv(input_file, dtype='string', keep_default_na=False, sep=',').values
    return myFile

def bart_score(text_1, text_2):
    chatgpt_score = bartscorer.score([text_1], [text_2])
    return chatgpt_score

def check_bart_score(input_text, raw):
    THRESHOLD = -2.459
    temp = normalize_text(raw) 
    score = bart_score(input_text, temp)[0] 
    if score < THRESHOLD:    
        return False
    return True


def get_column(input_file, name):
    df = pd.read_csv(input_file, dtype='string', keep_default_na=False, sep=',')

    saved_column = df[name] 
    return saved_column.values


def generate_column_names(kinds):
    column_names = ['human']
    for name in kinds:
        column_names.append(name)
    for first in kinds:
        for second in kinds:
            column_names.append(f"{first}_{second}")
    return column_names

def write_new_data(output_file, current_generation, column_names):
    data = []
    for column in column_names:
        data.append(current_generation[column])
    write_to_csv(output_file, data)

def refine(input_text, candidate):
    temp = candidate[:]
    temp = temp.strip()
    marks = ["```", "'", '"']
    for mark in marks:
        count_input = input_text.count(mark)
        count_temp = temp.count(mark)
        if count_temp == count_input + 2 and temp.startswith(mark) and temp.endswith(mark):
            temp = temp.strip(mark)
    return temp

def extract_by_best_similarity(input_text, raw):
    raw = refine(input_text,raw) 
    raw_candidates = nltk.sent_tokenize(raw) 
    candidates = []
    for candidate in raw_candidates:
        candidates.extend(candidate.split("\n"))
    best_similarity = -9999
    best_candidate = ""
    for candidate in candidates:
        refined_text = refine(input_text, candidate)
        if check_bart_score(input_text, refined_text): 
            score = bart_score(input_text, refined_text)[0] 
            if score > best_similarity:
                best_similarity = score
                best_candidate = refined_text
    print(f"best_candidate = {best_candidate}")
    if best_candidate == "":
        return input_text
    return best_candidate

def proofread_with_best_similarity(input_text, model_kind):
    input_text = normalize_text(input_text)
    print_and_log(f"INPUT = {input_text}")
    result = ""
    for i in range(1):
        if model_kind == CHATGPT:
            raw = chatGPT_proofread(input_text, is_output_normalization = IS_OUTPUT_NORMALIZATION)  
        elif model_kind == GEMINI:
            raw = gemini_proofread(input_text, is_output_normalization = IS_OUTPUT_NORMALIZATION) 
        else:
            raw = proofread_by_model_name(model_kind, input_text, is_output_normalization = IS_OUTPUT_NORMALIZATION) 
        result = extract_by_best_similarity(input_text, raw) 
        
        print_and_log(f"RAW_{i} = {raw}")
        print_and_log(f"RESULT_{i} = {result}")
        result = normalize_text(result) 
        if result != "":
            return raw, result
    return raw, result

"""Generate file name based on the path of exist_data_file and all kinds in exist_kinds and new_kinds"""
def generate_file_name(exist_data_file, exist_kinds, new_kinds):
    kinds = []
    for kind in exist_kinds:
        kinds.append(kind)
    for kind in new_kinds:
        kinds.append(kind)

    file_path =  os.path.dirname(exist_data_file)
    new_name = "_".join(kinds) + f"_with_best_similarity.csv"
    output_file = os.path.join(file_path, new_name)
    return output_file


def generate_new_data_with_best_similarity(exist_data_file, exist_kinds, new_kinds):
    kinds = []
    for kind in exist_kinds:
        kinds.append(kind)
    for kind in new_kinds:
        kinds.append(kind)

    column_names = generate_column_names(kinds) 
    exist_generation = generate_column_names(exist_kinds) 
    output_file = generate_file_name(exist_data_file, exist_kinds, new_kinds)
    if not os.path.exists(output_file):
        write_to_csv(output_file, column_names) 

    exist_data = dict()
    for kind in exist_generation:
        exist_data[kind] = get_column(exist_data_file, kind) 
    
    input_data = read_csv_data(output_file) 
    start = len(input_data)
    print(f"start = {start}")
    N = len(exist_data["human"])
    GLOBAL_GENERATE_SET = []
    GLOBAL_REUSE = []
    
    for index in range(start, N):
        # generate one
        GENERATE_SET = []
        REUSE = []
        current_generation = dict()
        for kind in exist_generation:
            current_generation[kind] = exist_data[kind][index]
        print(f"current_generation before generation = {current_generation}")
        human_text = current_generation["human"]
        for kind in new_kinds:
            _, generated_text = proofread_with_best_similarity(human_text, kind) 
            current_generation[kind] = generated_text
            GENERATE_SET.append(kind)
        print(f"current_generation after generate one = {current_generation}")
        # generate combination
        for first in kinds:
            for second in kinds:
                combination_name = f"{first}_{second}"
                if not combination_name in current_generation:
                    if first in current_generation and current_generation[first] == human_text:
                        generated_text = current_generation[second]
                        REUSE.append(f"{combination_name} from {second}")
                    else:
                        is_need_generation = True
                        for first_2 in kinds:
                            if first != first_2 and current_generation[first] == current_generation[first_2]:
                                combination_name_2 = f"{first_2}_{second}"
                                if combination_name_2 in current_generation:
                                    generated_text = current_generation[combination_name_2]
                                    REUSE.append(f"{combination_name} from {combination_name_2}")
                                    is_need_generation = False
                                    break
                        if is_need_generation:
                            _, generated_text = proofread_with_best_similarity(current_generation[first], second) 
                            GENERATE_SET.append(f"{first}_{second}")
                    combination_name = f"{first}_{second}"
                    current_generation[combination_name] = generated_text
        write_new_data(output_file, current_generation, column_names) 
        GLOBAL_GENERATE_SET.append(GENERATE_SET)
        GLOBAL_REUSE.append(REUSE)


def shuffle(array, seed):
    for a in array:
        random.Random(seed).shuffle(a)

def generate_human_with_shuffle(dataset_name, colum_name, num_samples, output_file):
    dataset = load_dataset(dataset_name)  
    data  = dataset['train']
    lines = []
    for sample in data:
        nltk_tokens = nltk.sent_tokenize(sample[colum_name])    
        lines.extend(nltk_tokens)
    filter_data = []
    for line in lines:
        if line != "":
            filter_data.append(line)
    lines = filter_data
    shuffle([lines], seed= SEED)
    if not os.path.exists(output_file):
        header = ["human"]
        write_to_csv(output_file, header) 

    number_of_processed_lines = number_of_csv_lines(output_file)  

    print(f"lines before = {lines[:num_samples]}")
    lines = lines[number_of_processed_lines:num_samples]
    print(f"lines after = {lines}")

    for index, human in enumerate(lines):
        h = normalize_text(human) 
        output_data = [h]
        write_to_csv(output_file, output_data)
        print(f"processed {index + 1} / {len(lines)}; {number_of_processed_lines + index + 1} / {num_samples}")



def split(data, ratio):
    train_num = int(len(data) * ratio)
    train_data = data[:train_num]
    test_data = data[train_num:]
    return train_data, test_data

def bart_score_in_batch(text_1, text_2):
    return bartscorer.score(text_1, text_2, batch_size=BATCH_SIZE)

def extract_feature_in_batch(text_1, text_2, feature_kind):
    feature = bart_score_in_batch(text_1, text_2)    
    return feature

def abstract_train(features, labels):
    model = MLPClassifier()
    model.fit(features, labels)
    return model

def evaluate_model(model, features, labels):
    y_pred = model.predict(features)    
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(labels, predictions) 
    write_to_file(OUTPUT_FILE, "Accuracy: %.1f%%" % (accuracy * 100.0) + "\n")
    roc_auc = roc_auc_score(labels, predictions)
    write_to_file(OUTPUT_FILE, "roc_auc: %.1f%%" % (roc_auc * 100.0) + "\n")

def combine_text_with_BERT_format(text_arr):
    result = f"<s>{text_arr[0]}</s>"
    for i in range(1, len(text_arr)):
        result += f"</s>{text_arr[i]}</s>"
    return result
    
def preprocess_function_multimodel(sample):
    n = len(sample["text"][0])
    text = []
    for i in range(n):
        text.append([])
    for sub_sample in sample["text"]:
        for i in range(n):
            text[i].append(sub_sample[i])
    score = []
    for i in range(1, n):
        score.append(bart_score_in_batch(text[0], text[i]))

    combined_text = []
    for index, sub_sample in enumerate(sample["text"]):
        text_arr = []
        input_text = sub_sample[0]
        text_arr.append(input_text)
        n = len(sub_sample)
        arr_tuple = []
        for i in range(1, n):
            generation = sub_sample[i]
            sub_score = score[i-1][index]
            arr_tuple.append((sub_score, generation))
        sort_arr = sorted(arr_tuple,reverse=True)
        for _, text in sort_arr:
            text_arr.append(text)
        combine = combine_text_with_BERT_format(text_arr) 
        combined_text.append(combine)
    return tokenizer(combined_text, add_special_tokens=False, truncation=True)



def preprocess_function_single_from_multimodel(sample):
    combined_text = []
    for sub_sample in sample["text"]:
        input_text = sub_sample[0]
        combined_text.append(input_text)
    return tokenizer(combined_text, truncation=True)

def check_api_error(data):
    for item in data:
        if item == API_ERROR or item == IGNORE_BY_API_ERROR:
            return True
    return False

def train_only_by_transformer_with_test_evaluation_early_stop(train, test, input_kind, number_class = 2):
    if input_kind == MULTIMODEL:
        train = train.map(preprocess_function_multimodel, batched=True) 
        test = test.map(preprocess_function_multimodel, batched=True)
    elif input_kind == SINGLE_FROM_MULTIMODEL:
        train = train.map(preprocess_function_single_from_multimodel, batched=True) 
        test = test.map(preprocess_function_single_from_multimodel, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if number_class == 3:
        model = AutoModelForSequenceClassification.from_pretrained("pretrained_model/roberta-base_num_lables_3", num_labels=number_class)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH[MODEL_NAME], num_labels=number_class)
    learning_rate = LEARNING_RATE[MODEL_NAME]

    folder = "training_with_callbacks"
    if os.path.exists(folder):
        shutil.rmtree(folder)

    args = TrainingArguments(
        f"training_with_callbacks",
        evaluation_strategy = "epoch", 
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING,
        weight_decay=0.01,
        push_to_hub=False,
        metric_for_best_model = OPTIMIZED_METRIC,
        load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    return trainer

def calculate_number_of_model(number_of_column):
    number_of_model = 0
    count_human = 1
    while True:       
        count_single = number_of_model
        count_pair = number_of_model * number_of_model
        if count_human + count_single + count_pair == number_of_column:
            return number_of_model
        elif count_human + count_single + count_pair > number_of_column:
            raise  Exception("Can not calculate number of model")
        number_of_model += 1

def read_multimodel_data_from_csv(multimodel_csv_file):
    input_data = read_csv_data(multimodel_csv_file) 
    result = []
    number_of_model = calculate_number_of_model(len(input_data[0])) 
    for data in input_data:
        item = dict()
        index = 0
        item["human"] = data[index]
        index += 1
        single = []
        for i in range(number_of_model):
            single.append(data[index])
            index += 1
        item["single"] = single
        pair = []
        for i in range(number_of_model):
            sub_pair = []
            for j in range(number_of_model):
                sub_pair.append(data[index])
                index += 1
            pair.append(sub_pair)
        item["pair"] = pair
        result.append(item)
    return result

def check_error(item):
    if check_api_error(item["human"]): 
        return True
    for text in item["single"]:
        if check_api_error(text):
            return True
    number_of_model = len(item["single"])
    for i in range(number_of_model):
        for j in range(number_of_model):
            if check_api_error(item["pair"][i][j]):
                return True
    return False

def create_pair_sample(item, train_index):
    result = []
    if check_error(item): 
        return result
    for index in train_index:
        if item["human"] != item["single"][index]:
            text_arr = []
            machine = item["single"][index]
            text_arr.append(machine)
            for sub_index in train_index:
                text_arr.append(item["pair"][index][sub_index])
            sample = dict()
            sample["text"] = text_arr
            sample["label"] = MACHINE_LABEL
            result.append(sample)
    # create human sample
    text_arr = []
    text_arr.append(item["human"])
    for index in train_index:
        text_arr.append(item["single"][index])
    sample = dict()
    sample["text"] = text_arr
    sample["label"] = HUMAN_LABEL
    number_of_machine = len(result)
    for _ in range(number_of_machine):
        result.append(sample)
    return result

def create_pair_test_sample(item, train_index, test_index):
    result = []
    if check_error(item):
        return result
    for test_idx in test_index:
        if item["human"] != item["single"][test_idx]:
            text_arr = []
            machine = item["single"][test_idx]
            text_arr.append(machine)
            for train_idx in train_index:
                text_arr.append(item["pair"][test_idx][train_idx])
            sample = dict()
            sample["text"] = text_arr
            sample["label"] = MACHINE_LABEL
            result.append(sample)
    # create human sample
    text_arr = []
    text_arr.append(item["human"])
    for index in train_index:
        text_arr.append(item["single"][index])
    sample = dict()
    sample["text"] = text_arr
    sample["label"] = HUMAN_LABEL
    number_of_machine = len(result)
    for _ in range(number_of_machine):
        result.append(sample)
    return result

def create_train_val_sample(data, train_index):
    result = []
    for item in data:
        sub_sample = create_pair_sample(item, train_index) 
        result.extend(sub_sample)
    return result

def create_test_sample(data, train_index, test_index):
    result = []
    for item in data:
        sub_sample = create_pair_test_sample(item, train_index, test_index) 
        result.extend(sub_sample)
    return result


def distribute_data(data, train_index, test_index, train_ratio, val_ratio):
    train_data, val_data, test_data = split_train_val_test(data, train_ratio, val_ratio) 
    train_sample = create_train_val_sample(train_data, train_index) 
    write_to_file(OUTPUT_FILE, f"train samples = {len(train_sample)}\n")
    val_sample = create_train_val_sample(val_data, train_index) 
    write_to_file(OUTPUT_FILE, f"val samples = {len(val_sample)}\n")
    test_sample = create_test_sample(test_data, train_index, test_index) 
    write_to_file(OUTPUT_FILE, f"test samples = {len(test_sample)}\n")
    return train_sample, val_sample, test_sample

def convert_to_huggingface_with_multimodel(sample):
    return Dataset.from_list(sample) 

def train_by_transformer_with_multimodel_and_early_stop(train_sample, val_sample, input_kind):
    train_data = convert_to_huggingface_with_multimodel(train_sample) 
    val_data = convert_to_huggingface_with_multimodel(val_sample)    
    return train_only_by_transformer_with_test_evaluation_early_stop(train_data, val_data, input_kind) 

def test_by_transformer_with_multimodel(detector, test_sample, input_kind):
    test_data = convert_to_huggingface_with_multimodel(test_sample) 
    if input_kind == MULTIMODEL:
        test_data = test_data.map(preprocess_function_multimodel, batched=True) 
    elif input_kind == SINGLE_FROM_MULTIMODEL:
        test_data = test_data.map(preprocess_function_single_from_multimodel, batched=True) 
    result = detector.evaluate(eval_dataset=test_data)
    roc_auc = result['eval_roc_auc']
    write_to_file(OUTPUT_FILE, "roc_auc: %.1f%%" % (roc_auc * 100.0) + "\n")

def extract_by_feature_kind(sample, feature_kind):
    text_1 = []
    text_2 = []
    label = []
    for sub_sample in sample:
        text_1.append(sub_sample["text"][0])
        text_2.append(sub_sample["text"][1])
        label.append(sub_sample["label"])
    feature = extract_feature_in_batch(text_1, text_2, feature_kind) 
    return feature, label

def train_by_feature_kind(train_sample, feature_kind):
    feature, label = extract_by_feature_kind(train_sample, feature_kind) 
    feature = np.array(feature)
    feature = feature.reshape(-1, 1)
    model = abstract_train(feature, label) 
    return model


def test_by_feature_kind(detector, sample, feature_kind):
    feature, label = extract_by_feature_kind(sample, feature_kind) 
    feature = np.array(feature)
    feature = feature.reshape(-1, 1)
    evaluate_model(detector, feature, label)    

def general_process_multimodels_train_val_test(train_sample, val_sample, test_sample):    


    input_kind = MULTIMODEL
    write_to_file(OUTPUT_FILE, f"\ninput_kind = {input_kind} \n")
    detector = train_by_transformer_with_multimodel_and_early_stop(train_sample, val_sample, input_kind) 
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")
    test_by_transformer_with_multimodel(detector, train_sample, input_kind) 

    write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")
    test_by_transformer_with_multimodel(detector, val_sample, input_kind)
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")
    
    test_by_transformer_with_multimodel(detector, test_sample, input_kind)


    input_kind = SINGLE_FROM_MULTIMODEL
    write_to_file(OUTPUT_FILE, f"\ninput_kind = {input_kind} \n")
    detector = train_by_transformer_with_multimodel_and_early_stop(train_sample, val_sample, input_kind)
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")
    test_by_transformer_with_multimodel(detector, train_sample, input_kind)
    write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")
    test_by_transformer_with_multimodel(detector, val_sample, input_kind)
    write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")
    test_by_transformer_with_multimodel(detector, test_sample, input_kind)

    n = len(train_sample[0]["text"])
    if n == 2:
        feature_kind = BART        
        write_to_file(OUTPUT_FILE, f"\nfeature_kind = {feature_kind} \n")
        detector = train_by_feature_kind(train_sample, feature_kind) 
        write_to_file(OUTPUT_FILE, f"EVALUATE ON TRAIN SET \n")
        test_by_feature_kind(detector, train_sample, feature_kind) 
        write_to_file(OUTPUT_FILE, f"EVALUATE ON VALIDATION SET \n")
        test_by_feature_kind(detector, val_sample, feature_kind)
        write_to_file(OUTPUT_FILE, f"EVALUATE ON TEST SET \n")
        test_by_feature_kind(detector, test_sample, feature_kind)

def process_multi_models_with_validation(multimodel_csv_file, train_index, test_index, number_sample):
    write_to_file(OUTPUT_FILE, f"PROCESSING FILE={multimodel_csv_file} \n") 
    write_to_file(OUTPUT_FILE, f"EXPERIMENT WITH {MODEL_NAME} model \n")
    write_to_file(OUTPUT_FILE, f"NUMBER OF NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING = {NUMBER_OF_MAX_EPOCH_WITH_EARLY_STOPPING} \n")
    write_to_file(OUTPUT_FILE, f"PATIENCE = {PATIENCE} \n")
    write_to_file(OUTPUT_FILE, f"OPTIMIZED_METRIC = {OPTIMIZED_METRIC} \n")
    write_to_file(OUTPUT_FILE, f"BATCH_SIZE = {BATCH_SIZE} \n")
    write_to_file(OUTPUT_FILE, f"number_sample = {number_sample} \n")

    data = read_multimodel_data_from_csv(multimodel_csv_file) 
    data = data[:number_sample]

    train_sample, val_sample, test_sample = distribute_data(data, train_index, test_index, TRAIN_RATIO, VAL_RATIO) 
    
    write_to_file(OUTPUT_FILE, f"Multimodel training with {train_index}, test with {test_index} \n")
    general_process_multimodels_train_val_test(train_sample, val_sample, test_sample) 



def split_train_val_test(data, train_ratio, val_ratio):
    train_num = int(len(data) * train_ratio)
    val_num = int(len(data) * val_ratio)
    train_data = data[:train_num]
    val_data = data[train_num:(train_num + val_num)]
    test_data = data[(train_num + val_num):]
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='SimLLM.')
    parser.add_argument('--LLMs', nargs="+", default=[CHATGPT, NOUS_HERMES_2_YI_34B, OPENCHAT_3_5_1210], help='List of large language models')
    parser.add_argument('--train_indexes', type = int, default=[0,1,2], nargs="+", help='List of training indexes')
    parser.add_argument('--test_indexes', type = int, default=[0], nargs="+", help='List of testing indexes')

    parser.add_argument('--num_samples', type = int, default=5000, help='Number of samples')

    args = parser.parse_args()

    dataset_name = "xsum"
    colum_name = "document"
    num_samples = args.num_samples
    output_file = "data/human.csv"
    generate_human_with_shuffle(dataset_name, colum_name, num_samples, output_file) 


    exist_data_file = output_file
    exist_kinds = []
    new_kinds = args.LLMs
    generate_new_data_with_best_similarity(exist_data_file, exist_kinds, new_kinds) 

    multimodel_csv_file = generate_file_name(exist_data_file, exist_kinds, new_kinds)
    number_sample = -1 # run with all samples
    train_index = args.train_indexes
    test_index = args.test_indexes
    process_multi_models_with_validation(multimodel_csv_file, train_index, test_index, number_sample) 

main() 