# tune.decoder.py

```python

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from unsloth.chat_templates import train_on_responses_only

import torch
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
import os,random,sys,json,re

random.seed()
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

seed = get_truly_random_seed_through_os()
set_seed(seed)

json_file = sys.argv[1]
with open(json_file,"r") as jf:
    config = json.load(jf)

MODEL = config["MODEL"]
TRAIN_FILE = config["TRAIN_FILE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
OVERWRITE = bool(config["OVERWRITE"])
BATCH_SIZE = int(config['BATCH_SIZE'])
EPOCHS = int(config["EPOCHS"])
LRATE = float(config["LRATE"])
STEPS = int(config["STEPS"])
LOAD_4BIT = config["LOAD_4BIT"].lower() == "true"
LOAD_8BIT = config["LOAD_8BIT"].lower() == "true"
FULLTUNE = config["FULLTUNE"].lower() == "true"
OPTIMIZER = config["OPTIM"]
MAXSEQ= int(config["MAXSEQ"])
if("PERCENT" in config):
    PERCENT = int(config["PERCENT"])
else:
    PERCENT = 100
if("NUM_SAMPLES" in config):
    NUM_SAMPLES = int(config["NUM_SAMPLES"])
else:
    NUM_SAMPLES=0
if("SELECT_OUTPUT" in config):
    SELECT_OUTPUT = config["SELECT_OUTPUT"]
else:
    SELECT_OUTPUT = "output"
if("SHUFFLE" in config):
    os.system("python " + config["SHUFFLE"])

#config["EPOCHS"]= str(EPOCHS)
#with open(json_file,"w") as jf:
#    json.dump(config,jf,indent=4)

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
if(not is_bfloat16_supported()):
    dtype = torch.float16
    fp16 = True
else:
    dtype = torch.bfloat16
    bf16 = True

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    dtype=dtype,
    max_seq_length = MAXSEQ,
    load_in_4bit=LOAD_4BIT,
    load_in_8bit=LOAD_8BIT,
    full_finetuning=FULLTUNE,
    device_map = "auto"
)


print("-----------------------------------------------------")
print("Configuration")
print("-----------------------------------------------------")
print("MODEL",MODEL)
print("TRAIN_FILE",TRAIN_FILE)
print("OUTPUT_DIR",OUTPUT_DIR)
print("BATCH_SIZE","AUTO")
print("EPOCHS",EPOCHS)
print("LRATE",LRATE)
print("STEPS",STEPS)
print("LOAD_4BIT",LOAD_4BIT)
print("LOAD_8BIT",LOAD_8BIT)
print("FULLTUNE",FULLTUNE)
print("MAXSEQ",MAXSEQ)
print("-----------------------------------------------------")


dataset = load_dataset("json", data_files=TRAIN_FILE, split="train").shuffle(seed)

if(NUM_SAMPLES==0):
    NUM_SAMPLES = int(len(dataset) * (PERCENT / 100))

# Select only the first X% of the dataset
dataset = dataset.select(range(NUM_SAMPLES))

output_tags = [
    "output",
    "response",
    "answer",
    "assistant",       
]

ignore_tags = [
    "style_guide",
    "ignore",
    "system",
    "select_output",
    "opcode",    
]
input_tags = [
    "instruction",
    "question",
    "input",    
    "context",
    "user",
    "keywords",
    "tweets",
    "hashtags",
    "visualtags",
    "entitytags",
    "attributes",
    "characteristics",
    "features",
    "properties",
    "key_points",
    "key_attributes",
    "key_characteristics",
    "key_features",
    "key_properties",        
    "term",
    "entity",
    "title",
    "definition",
    "description",
    "summary",
    "positive_prompt",
    "negative_prompt",
    "ai_persona_prompt",
    "persona_prompt",
    "detailed_ai_persona_prompt",
    "entity_persona",
]

def clean(example):
    x = {}
    for k in list(example.keys()):
        key = k.lower().replace(" ","_")
        if(type(example[k]) is list):
            x[key] = []
            for j in example[k]:
                if(j is None): continue
                if(j == ''): continue
                if(len(j) == 0): continue                 
                x[key].append(j)
        else:
            j = example[k]
            if(j is None): continue
            if(j == ''): continue
            if(len(j) == 0): continue 
            x[key] = j 
    return x
 
def merge(example):
    x = {}
    for k in list(example.keys()):
        key = k.lower().replace(" ","_")
        if(k in output_tags):
            if(type(example[k]) is list):
                x["output"] = '\n\n'.join(example[k])
            else:
                x["output"] = example[k]        
        else:
            if(type(example[k]) is list):
                x[key] = '\n\n'.join(example[k])
            else:
                x[key] = example[k]        

    return x

def select_input(ex):
    for k in input_tags:        
        if(k in ex): return ex[k]
    return None 

def select_output(example):
    return example[SELECT_OUTPUT]
    

def to_text(example):
    global SELECT_OUTPUT
    print(SELECT_OUTPUT)
    ex = dict(example)    
        
    input = {}
    if(SELECT_OUTPUT == "Random"):
        x = random.choice(example.keys())
        while x in input_tags:
            x = random.choice(example.keys())
        SELECT_OUTPUT = x
    
    for k in list(ex.keys()):
        if(ex[k] is None): continue
        if(k != SELECT_OUTPUT):
            input[k] = ex[k]
       
    temp = ''
    for k in list(input.keys()):
        if(type(input[k]) is list):
            temp = temp + k + ": " + ",".join(input[k]) + "\n"
        else:
            temp = temp + k + ": " + input[k] + "\n"
    if(SELECT_OUTPUT != 'output'):
        temp = SELECT_OUTPUT + ": " + temp.strip()
    else:
        temp = temp.strip()
    input = f"""```{temp}```"""
    
    temp = select_output(ex)
    if(type(temp) is list):
        temp = '\n\n'.join(temp)
    temp = temp.strip()
    output = f"""```{temp}```<|endoftext|><|end_of_text|>"""

    print("### Prompt:")
    print(input)
    print("### Response:")
    print(output)
    return {
        "text": f"### Prompt:\n\n{input}\n\n### Response:\n\n{output}<|endoftext|><|end_of_text|>"
    }

train_dataset = dataset.map(to_text, remove_columns=dataset.column_names)
last_checkpoint = None
last_checkpoint_step = 0




print("-------------------------------------------------------------")

if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    # Extract the step count from checkpoint path (e.g., "checkpoint-500")
    last_checkpoint_step = int(last_checkpoint.split("-")[-1])
else:
    print("No previous checkpoint found. Training from scratch.")

total_samples = len(train_dataset)
print("Total Samples:",total_samples)
num_gpus = max(1, torch.cuda.device_count())  # Ensure at least 1 (for CPU training)
print("Num GPU:",num_gpus)
print("Batch Size/Device:",BATCH_SIZE)
print("Gradient Steps:", STEPS)
# Compute steps for one epoch based on current dataset size
num_update_steps_per_epoch = total_samples // (
    num_gpus * BATCH_SIZE * STEPS
)

print("Steps: ",num_update_steps_per_epoch)
# Adjust max_steps based on last checkpoint
max_steps = last_checkpoint_step + num_update_steps_per_epoch
print(f"Updated max_steps: {max_steps}")

print("-------------------------------------------------------------")

resume = last_checkpoint is not None

if(FULLTUNE == False):
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = seed,
        max_seq_length = MAXSEQ,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

args = SFTConfig(
    max_seq_length = MAXSEQ,
    #per_device_train_batch_size = BATCH_SIZE,
    auto_find_batch_size=True,           # <--- This enables automatic batch sizing
    gradient_accumulation_steps = STEPS,
    learning_rate = LRATE,
    warmup_steps = 10,
    logging_steps = 1,
    output_dir = OUTPUT_DIR,
    optim = OPTIMIZER,
    num_train_epochs = EPOCHS,
    seed = seed,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    resume_from_checkpoint = resume,
    lr_scheduler_type = "linear",
    neftune_noise_alpha=5
)

trainer = SFTTrainer(
    model = model,
    dataset_text_field = "text",
    packing = False,
    train_dataset = train_dataset,
    #eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    args = args,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Prompt:\n\n",
    response_part    = "### Response:\n\n"
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

checkpoint = None
if resume == True:
    checkpoint = last_checkpoint

trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Saving Model....")
#trainer.save(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```
