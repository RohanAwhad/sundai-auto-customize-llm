


Model:
  - Qwen/Qwen3-4B-Instruct-2507
  - 

Which Tool:  
Which knowledge (set of docs):  

Evaluate 2 things:

1. Knowledge about new tool/docs:
  - How to evaluate new knowledge customized model?
    - Internal QA
  - How to evaluate new tool customized model?
    - Hermes like tool calling evaluation?

2. General Knowledge that model possessed before the customization:  
  - How do we evaluate previous skills/knowledge recall
    - IF-Eval


# Synthetic Data Generation:

## Knowledge:

> [Refer SDG HUB Knowledge Generation example](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/tree/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning)

- For now we used: GLM-4.7-fp8

## Tools

Input Tool Definition  
  name:  openclaw agents  
  description: Use this for creating/ configuring agents  
  arguments:...  


- LLM-generated Scenarios based on definition:
  - create an data analysis agent
  - create an code gent

- Based on the scenarios generate user messages 

- Based on the user messages generate conversation traces

- Clean conversation traces:
  - Remove skill prompt from system prompt
  - Remove tool exploration from the conversation history

- Expected Output Trace [single turn]:

  - System prompt [basic/general]
  - Human Message
  - Successful attempt of assistant on calling those tools
  - Tool Response
  - Assistant response to the human


> Data Mixing [30% original dataset + 70% new dataset]

# Training

## [Mid] SFT - Training for customized knowledge about tool / doc

- https://github.com/RohanAwhad/oc-user-sim/blob/main/train_lora.py

## On-policy Distillation


# Evaluation:



| Model | Internal QA Eval (Knowledge) | IF-eval (Chat) |
|---|---|---|
| Qwen3-4B-Instruct | 3% | 82.25% |
| + original-dataset [30%] + midtrain (70%) |  | 58% |
| + original-dataset [30%] + midtrain (70%) + distill |  |  |



> * For QA Eval we use llm-as judge for binary scoring 
> * we are reporting prompt-level strict accuracy for IF-Eval 


Example from on-policy blog. This is how we will show our results!

| Model | Internal QA Eval (Knowledge) | IF-eval (Chat) |
|---|---|---|
| Qwen3-8B | 18% | 85% |
| + midtrain (100%) | 43% | 45% |
| + original-dataset [30%] + midtrain (70%) | 36% | 79% |
| + original-dataset [30%] + midtrain (70%) + distill | 41% | 83% |





---

# References:

- [Document Knowledge Ingestion](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/tree/main/examples/knowledge_tuning/enhanced_summary_knowledge_tuning)
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Hermes Tool calling dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [Toucan paper](https://arxiv.org/abs/2510.01179): "Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs."

