[TOC]

### Introduction: Fine-Tuning LLaMA 3.1 with Tree of Thoughts Reasoning

This project focuses on training and fine-tuning a Large Language Model (LLM) to enhance its reasoning capabilities through a structured problem-solving framework. Using the LLaMA 3.1 8B model as the base, the training process employs the high-quality Tree of Thoughts BASE 24k dataset from Hugging Face. The dataset consists of 24,000 question-answer pairs designed to promote structured thinking and iterative reasoning.

Dataset Overview: Tree of Thoughts BASE 24k
The Tree of Thoughts dataset serves as a foundation for fine-tuning LLMs by emphasizing structured reasoning and problem-solving strategies.

### Key Features:

- 24,000 Q&A Pairs: A rich collection of step-by-step solutions to diverse problems.
- Tree of Thoughts Approach: Answers are organized to reflect clear, logical thought processes.
- High-Quality Data: Created using Grok and LLaMA 3.1 70B, ensuring premium content.
- Diverse Topics: Covers a wide range of disciplines to foster versatile reasoning skills.

### Applications:

The dataset is designed to enhance the model’s ability to:

- Break down complex problems into manageable sub-questions.
- Explore multiple factors and perspectives systematically.
- Address inconsistencies and refine reasoning iteratively.
- Develop coherent, actionable strategies.
- Training Approaches

### Three advanced fine-tuning methods are applied to the dataset to optimize reasoning capabilities:

- LoRA (Low-Rank Adaptation): Reduces memory and computational costs by updating only low-rank matrices, enabling efficient adaptation with minimal parameter changes. https://arxiv.org/abs/2106.09685

- DoRA (Weight-Decomposed Low-Rank Adaptation): Builds on LoRA by decomposing pre-trained weights into magnitude and direction components, enhancing fine-tuning precision and efficiency. https://arxiv.org/abs/2402.09353

- NEFTune (Noisy Embeddings for Fine-Tuning): Adds noise to embedding layers during fine-tuning to improve generalization and ensure robust performance across varied instruction-based tasks. https://arxiv.org/abs/2310.05914

This project aims to push the boundaries of AI reasoning by combining cutting-edge datasets and innovative fine-tuning techniques, paving the way for more intelligent, versatile, and structured problem-solving capabilities in LLMs.

### Preparation

#### Config files

##### qlora:

- model repo: https://huggingface.co/OnlyThings/L3.1-test-qlora-model

```
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false
chat_template: llama3

datasets:
  - path: data/Tree_Of_Thoughts_BASE_24k_rename
    type: alpaca_chat.load_qa
    conversation: llama3
dataset_prepared_path: last_run_prepared
val_set_size: 0.0
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

lora_r: 128
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
# use_dora: true
# neftune_noise_alpha: 5

wandb_project: L3.1-test-qlora
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 1
micro_batch_size: 4
num_epochs: 3
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.00013

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: unsloth
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|end_of_text|>"
save_safetensors: true
```

##### dora

- model repo: https://huggingface.co/OnlyThings/L3.1-test-dora-model

```
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false
chat_template: llama3

datasets:
  - path: data/Tree_Of_Thoughts_BASE_24k_rename
    type: alpaca_chat.load_qa
    conversation: llama3
dataset_prepared_path: last_run_prepared
val_set_size: 0.0
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

lora_r: 128
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
use_dora: true
# neftune_noise_alpha: 5

wandb_project: L3.1-test-dora
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 1
micro_batch_size: 4
num_epochs: 3
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.00013

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: unsloth
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|end_of_text|>"
save_safetensors: true
```

##### neftune

- model repo: https://huggingface.co/OnlyThings/L3.1-test-neftune-model

```
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false
chat_template: llama3

datasets:
  - path: data/Tree_Of_Thoughts_BASE_24k_rename
    type: alpaca_chat.load_qa
    conversation: llama3
dataset_prepared_path: last_run_prepared
val_set_size: 0.0
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

lora_r: 128
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
# use_dora: true
neftune_noise_alpha: 5

wandb_project: L3.1-test-neftune
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 1
micro_batch_size: 4
num_epochs: 3
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.00013

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: unsloth
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|end_of_text|>"
save_safetensors: true
```

#### Goals

- Finetune Llama 3.1 8B
- Make a train qlora
- Make a train qlora(dora)
- Make a train qlora(neftune 5)
- Check what is the best between simple qlora train, dora train or neftune train

I will use the following dataset: https://huggingface.co/datasets/terrycraddock/Tree_Of_Thoughts_BASE_24k
This dataset was designed with Llama 3.1 70B and GROK, two LLM models much larger and more powerful than Llama 3.1 8B.

It will allow the model to learn reflexion, and to integrate a ToT (tree of taught) into it.
As the dataset is based on a larger version of Llama 3.1, the result should improve the basic output of the 8B model.
Please note, however: The aim here is not to improve the conversation, nor to make the bot more human, we're trying to make it more coherent and avoid wrong answers due to poor development of logical thinking as LLMs just predict the next most probable token. We do this by introducing thinking and reasoning before the actual output of the LLMs final answer.

#### Configuration and graph

I've used our previous configuration to make this train, slightly modified to smooth the curves of the wandb graphic.

##### qlora graph

![graph qlora](https://i.ibb.co/YX5bdZ0/image.png)

##### dora graph

![graph dora](https://i.ibb.co/r2ccDCS/image.png)

##### neftune graph

![graph neft](https://i.ibb.co/D12J5kK/image.png)

Apart from the fact that I'm using “DoRA” and “Neftune” for trains 2 and 3 respectively, the general configuration doesn't change.
We can therefore judge the models in the same way, with the same prompts.
Neftune “5” is chosen as the official paper, this being the value that gives the best results.

The models will be trained on Runpod using the “Axolotl” tool, with the following settings:

“adapter: qlora” (qlora only) : Train 1
“use_dora: true” : Train 2
“neftune_noise_alpha: 5” : Train 3

#### Expected results

This dataset teaches the model to think logically, resulting in more consistent answers.
The dataset page promises a 15% to 20% increase in the rate of correct answers compared with the basic model.
The resulting model COULD be able to answer with better reply than the base model, even without using the <thinking> and <output> tags, although this is not an expected result we search.
The resulting model will be able to do much better than the base model by using the <thinking> and <output> tags provided for this purpose during training.

Most of today's successful LLM models (Claude, ChatGPT, Mistral LeChat...) use a “ToT” in the background that is not shown to the user in the final prompt, which allows them to obtain much more precise answers than their competitors. This experience is an open door to assert that it theoretically works on any model, even if we're going to concentrate here on Llama 3.1 8B.

We expect one of the train (qlora, dora or neftune) to be better, more usable than the other. We want to see what stand out.

### The test

- Find some questions that Llama 3.1 8B has difficulty answering
- Ask the 3 models the same questions
- Obtain answers with and without ToT tags
- Compare results

IMPORTANT: Deterministic value and fixed seed (1337) was used to not be biased by randomness.
If a system prompt is used, it is the following: Use <thinking> and <output>

![Configuration of KoboldAI](https://gcdnb.pbrd.co/images/mx1UNeAJuNyB.png?o=1)

#### Question 1 - The strawberry case

How many letters R in the word strawberry

[One of the source of the exact prompt that make LLM goes crazy](https://www.linkedin.com/pulse/how-many-r-letters-word-strawberry-why-do-we-need-check-yesha-sivan-odxgf#:~:text=There%20are%20actually%20three%20%22r,in%20the%20word%20%22strawberry.%22)

Answer: _3_

Most model have an issue with this question so, that's the first one that come to my mind when I started this, the idea of the issue coming from the tokens is debatable as I succeded to get a correct answer out of the finetuned model.

qlora and neftune train did good with the thinking process, however, dora failed on all case.
Note that every FINETUNED model, that is NOT using the thinking bracket on this specific prompt just fail and repeat what the base model say.

##### Answers

**BASE - ❌ **FAIL\*\*

![base1](https://i.ibb.co/GnDhJ0C/image.png)

**QLORA (no system prompt) - ❌ **FAIL\*\*

![qlora clean1](https://i.ibb.co/Jzsrqmg/image.png)

**QLORA (with system prompt) - ✅ **PASS\*\*

![qlora 1](https://i.ibb.co/Vm3TfJq/image.png)

**DORA (no system prompt) - ❌ **FAIL\*\*

![dora clean1](https://i.ibb.co/SPBkcv2/image.png)

**DORA (with system prompt) - ❌ **FAIL\*\*

![dora 1](https://i.ibb.co/QMsjMS7/image.png)

**NEFTUNE (no system prompt) - ❌ **FAIL\*\*

![neftune clean1](https://i.ibb.co/H7gM82J/image.png)

**NEFTUNE (with system prompt) - ✅ **PASS\*\*

![neftune 1](https://i.ibb.co/Pj0v5HH/image.png)

#### Question 2 - The Sally riddle

Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?

Answer: _1_

The riddle most of the LLM have trouble to resolve. No surprise here, even Llama 3.1 70B fails most of the time.
The models did not get the right answer.

##### Answers

**BASE - ❌ **FAIL\*\*

![base2](https://i.ibb.co/zrfXp0z/image.png)

**QLORA (no system prompt) - ❌ **FAIL\*\*

![qlora clean2](https://i.ibb.co/sqtTW07/image.png)

**QLORA (with system prompt) - ❌ **FAIL\*\*

![qlora 2](https://i.ibb.co/3Y1ScT5/image.png)

**DORA (no system prompt) - ❌ **FAIL\*\*

![dora clean2](https://i.ibb.co/3z6XG5h/image.png)

**DORA (with system prompt) - ❌ **FAIL\*\*

![dora 2](https://i.ibb.co/fQQ6ZrD/image.png)

**NEFTUNE (no system prompt) - ❌ **FAIL\*\*

![neftune clean2](https://i.ibb.co/R6GCKj0/image.png)

**NEFTUNE (with system prompt) - ❌ **FAIL\*\*

![neftune 2](https://i.ibb.co/3Ywf6nJ/image.png)

#### Question 3 - The Trolley Problem

A train arrives at high speed. On its way, five people are tied to the rails and are about to be run over. You can divert the train's trajectory, but if you do, it will run over a person attached to the rails on the other trajectory. So what do you do?

Answer: _Divert it/Don't give an answer_

The base model give a direct answer, and the worst out of them : Killing 5 people AND divert the train (illogical)
I considered to choose "Divert it" as the definitive answer as it implies the less amount of casualty, and the fact of not giving a clear answer okay, but not the best.

All of the trained models gave a good answer, the thinking bracket worked like a charm on this situation, giving the best answer.

The neftune models showed their weakness here with the format, where it included the thinking bracket from the training when we didn't ask, and didn't closed them when we asked for them. Most likely this is due to the 'noise" added, so Neftune is not the best use case for accurate output and critical thinking, Neftuen seems better in unpredictable interesting output and fun casual conversations, but not when reliability is needed.
They also didn't gave definitive answer.

##### Answers

**BASE - ❌ **FAIL\*\*
![base2](https://i.ibb.co/wRztbWC/image.png)

**QLORA (no system prompt) (Mixed):**
![qlora clean2](https://i.ibb.co/rcJ44yt/image.png)

**QLORA (with system prompt) - ✅ **PASS\*\*
![qlora 2](https://i.ibb.co/4Sc4MGV/image.png)

**DORA (no system prompt) (Mixed):**
![dora clean2](https://i.ibb.co/Kz58cBQ/image.png)

**DORA (with system prompt) - ✅ **PASS\*\*
![dora 2](https://i.ibb.co/3YSd1kX/image.png)

**NEFTUNE (no system prompt) (Mixed):**
![neftune clean2](https://i.ibb.co/xqzBxsR/image.png)
_Note: wanted to finish with <output> without system prompt_

**NEFTUNE (with system prompt) (Mixed):**
![neftune 2](https://i.ibb.co/xGQfNPm/image.png)
_Note: Didn't end </output>_

#### Question 4 - Common sense/logic

I'm in my bedroom, there is a desk. On my desk, there is a book. I get up to grab the apple on my desk, walk to the bathroom and come back to sit in my chair. Where is the apple ?

Answer: _In your hand_

The goal here is to give the model a scenario that is less "logical" than normal. Here, it's very unlikely that an human would do that for any reason, and the weird situation is made to confuse the model. Adding detail that doesn't matter confuse it even more.
The base model gives a bad answer.
The finetuned models did great overall, but the usage of the thinking bracket confused the model even more in some situation: qlora gives a wrong answer but explain why, dora was wrong and neftune was right, despite having the same very long thinking development, he got the right answer in his guess.

##### Answers

**BASE - ❌ **FAIL\*\*
![base2](https://i.ibb.co/1bxHg9z/image.png)

**QLORA (no system prompt) - ✅ **PASS\*\*
![qlora clean2](https://i.ibb.co/mhLK1c0/image.png)

**QLORA (with system prompt) (Mixed):**
![qlora 2](https://i.ibb.co/ZYBc72c/image.png)
_Note: answer is technically not wrong because the thinking process isn't, but it's not what I wanted_

**DORA (no system prompt) - ✅ **PASS\*\*
![dora clean2](https://i.ibb.co/T05qMPF/image.png)

**DORA (with system prompt) - ❌ **FAIL\*\*
![dora 2](https://i.ibb.co/Z8pWZbg/image.png)
_Note: too long reply / bad answer_

**NEFTUNE (no system prompt) - ✅ **PASS\*\*
![neftune clean2](https://i.ibb.co/09vKmGv/image.png)

**NEFTUNE (with system prompt) - ✅ **PASS\*\*
![neftune 2](https://i.ibb.co/SvQ82Rq/image.pnge.png)
_Note: too long like dora, thinking dev like dora, but end up with the right answer_

### Going further

Now that we tried the most "known" LLM prompt causing issues and some conversation, let's go further and try our model for what is was specified to do : think, use the bracket for thinking, reasoning and answering.
Let's dive into prompt that will test the limit of the base model, and show how our model handles problematic prompts.
Those prompt were specifically crafted to fail in our setup on the base model 100% of the time with the deterministic configuration shown at the top of this documentation.

I've seen two instance where the base model produced garbage while testing:

- The model enters in a loop, and it's impossible to stop it
- The model output his reply in a weird format like "$\boxed{answer}$", probably because it is too confused

![llama error](https://i.ibb.co/BtjgbYt/image.png)

The model used will be the one simply trained (qlora) with the system prompt instructing to use the bracket.

#### Question 1

"In a town, 50% of people always tell the truth, and 50% always lie. You meet someone from the town who says, 'I am a liar.' What does this mean?"

Success: **Recognizing that this is a paradox—a liar cannot truthfully claim to be a liar. A rational model would state that this person's statement cannot hold logically.**
Failure: **Claiming the statement is true or false without mentioning the contradiction.**

##### Answers

![Q1 answer](https://i.ibb.co/d6Kdm3F/image.png)

#### Question 2

"Two trains are on the same track, starting 100 km apart. Train A travels at 60 km/h, Train B travels at 40 km/h, and a fly starts on Train A and flies to Train B at 80 km/h. It continues flying back and forth between the trains until they collide. How far has the fly flown when they collide?"

Success: **Recognizing that the time to collision is 1 hour (100 km / combined 100 km/h), and thus the fly travels 80 km in that time.**
Failure: **Trying to sum the fly's individual back-and-forth distances instead of solving for time.**

##### Answers

![Q2 answer](https://i.ibb.co/mtX5LVt/image.png)

#### Question 3

"If you had a bag with 3 red balls, 3 green balls, and 3 blue balls, how many draws would you need to guarantee having 2 balls of each color?"

Success: **The correct answer is 6 draws. No matter how you draw, after 6 draws, you will have at least 2 balls of each color due to the worst-case scenario.**
Failure: **Any answer that doesn't properly account for the worst-case scenario or give the wrong number of draws.**

##### Answers

![Q3 answer](https://i.ibb.co/1Q4qY3P/image.png)

#### Question 4

"You are given two envelopes, each containing an unknown amount of money. One envelope contains twice the amount of the other. You may choose either envelope, but once you choose, you are allowed to switch. What should you do? Should you switch, stay with your choice, or is there no difference?"

Success: **Acknowledging the paradox of the situation, realizing that switching appears rational (since you "might" win more), but there's no definitive answer due to the structure of the problem. The issue lies in the illogical assumption that you can always benefit from switching.**
Failure: **Offering a simple choice or trying to calculate the best option without considering the inherent logical flaw in the setup.**

##### Answers

![Q4 answer](https://i.ibb.co/r2XVGCG/image.png)

#### Question 5

"A judge tells a condemned prisoner that they will be hanged at noon on one weekday in the following week, but the execution will be a surprise (i.e., the prisoner won't know the day until the executioner comes). The prisoner logically deduces that he cannot be hanged on Friday, then eliminates Thursday, Wednesday, and so on. Eventually, he concludes that he cannot be hanged at all. However, the execution occurs on Wednesday, surprising the prisoner. Explain this paradox."

Success: **Explaining the contradiction that arises from the assumption that the prisoner can "predict" the hanging. The paradox relies on the idea that the surprise element creates an expectation loop that ultimately causes the prisoner to miscalculate.**
Failure: **Offering a simple explanation without addressing the logical flaw in the prisoner's reasoning.**

##### Answers

![Q5 answer](https://i.ibb.co/hHMxQMm/image.png)

### Everyday usage

I used the three (well, two, neftune were really not stable enough) in multiple occasion, and here is my observation :

- The model did gain a boost in performance over the base Llama 3.1 8B for the majority of subjects I tried
- But the question "known" to be an issue with a lot of other models will probably be an issue here toon because the model is small
- Neftune is NOT stable enough for this sort of thing (brackets in the dataset) and tend to break/make mistakes a lot
- The usage of the thinking process have made the model smarter

For accurate specific question answers, _qlora_ seems to be best which was the purpose of this task.

Next up I will compare the 8b _qlora_ with the 70b _qlora_ model to see if introducing more parameters does answer all the questions correctly.

### Comparing 8B output to 70B output

The 70B model: https://huggingface.co/OnlyThings/L3.1-70B-test-qlora-model

#### Configuration

![graph 70b](https://i.ibb.co/qNdD3dT/image.png)

The base model was Llama-3.1-70B-Instruct, trained with the same configuration than our 8B, using the exact same dataset, asking the exact same questions. One shoot. Let's compare!

```
base_model: meta-llama/Llama-3.1-70B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false
chat_template: llama3

datasets:
  - path: data/Tree_Of_Thoughts_BASE_24k_rename
    type: alpaca_chat.load_qa
    conversation: llama3
dataset_prepared_path: last_run_prepared
val_set_size: 0.0
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

lora_r: 128
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:
# use_dora: true
# neftune_noise_alpha: 5

wandb_project: L3.1-70B-test-qlora
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 1
micro_batch_size: 1
num_epochs: 3
optimizer: paged_adamw_8bit
lr_scheduler: cosine
learning_rate: 0.00013

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: unsloth
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 2
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  pad_token: "<|end_of_text|>"
save_safetensors: true
```

#### The ReTest

**Question 1** - The strawberry case ✅ **PASS**

![Q1 70B](https://i.ibb.co/52Y7Z2h/image.png)

**Note of comparison**: !~ Get straight to the point, but give more detail than 8B, not a lot to say here. ~!

**Question 2** - The Sally riddle ✅ **PASS**

![Q2 70B](https://i.ibb.co/mDKPCv4/image.png)

**Note of comparison**: !~ HE FINALLY DID IT. That's interesting. We finally got the right answer out. ~!

**Question 3** - The Trolley Problem ✅ **PASS**

![Q3 70B](https://i.ibb.co/xXZm3T0/image.png)

**Note of comparison**: !~ Right answer again, more detailed than the smaller model, more step in the process. ~!

**Question 4** - Common sense/logic - ❌ **FAIL**

![Q4 70B](https://i.ibb.co/NrMJRk5/image.png)

**Note of comparison**: !~ Spoiler, that's the only question the model got wrong, like with the 8B qlora, he gave a thinking process that is not wrong in itself poiting toward an acceptable
answer, but it still get tricked into nonsense, and for a 70B, I give it a ❌ **FAIL**. ~!

#### Going Even Further

**Question 1** - ✅ **PASS**

![Q1 70B-2](https://i.ibb.co/vZYjw36/image.png)

**Note of comparison**: !~ Better thinking process in my opinion. Also feel more logical ~!

**Question 2** - ✅ **PASS**

![Q2 70B-2](https://i.ibb.co/LND1TK4/image.png)

**Note of comparison**: !~ Doing more process in less step than 8B, more logical, answer is right, thinking process too ~!

**Question 3** - ✅ **PASS**

![Q3 70B-2](https://i.ibb.co/R0hrnNR/image.png)

**Note of comparison**: !~ Like the question just before, we can see the thinking process is more evolved, it get the logical thinking good with more logic, less step, right answer ~!

**Question 4** - ✅ **PASS**

![Q4 70B-2](https://i.ibb.co/850krvt/image.png)

**Note of comparison**: !~ This time is actually reply to the question with a good answer, not a vague one. Explain with getting straight to the point the 50/50 chance (it don't matter) ~!

**Question 5** - ✅ **PASS**

![Q5 70B-2](https://i.ibb.co/xM8vnMj/image.png)

**Note of comparison**: !~ The most easy to see improvement in my opinion, the explaination is more well written than the 8B, it's easier to read, less longer and a correct answer ~!

#### Conclusion (8B/70B)

Without surprise, the 70B is more powerful than the 8B trained on the same source, even with the problematics prompt, however there is still a margin of error as the only fail the model got during
our test represent.

Outside of this failure, all the reply was better and more well written. The reply that got smaller was straight to the point, and the reply that got longer was just better logic.

Present issues or in the future could be fixed by:

- Using a bigger model (than the 70B) to make a new dataset using the thinking process
- Doing a full finetune of the model
- Using the perfect sample for what we need (temp, top_k...)
- Using special token for the thinking and output process (?)

Any of this can't be part of our experiment as it could biase the end result.

### NOTE

- We didn't add any new tokens, to avoid distorting the experience (as did the author)
- The Instruct version of the model has been chosen and not the Base, because this is not an FFT (full finetune) and therefore, the model is already ready for a conversation/chat, and has already
  been pre-trained on different domains.
