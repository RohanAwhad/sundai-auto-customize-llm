"""
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run teacher_server.py
"""
# ===
# Model
# ===
import os
import torch
import torch.nn.functional as F

from peft import PeftConfig
from transformers import AutoModelForCausalLM

STUDENT_MODEL_NAME = "./knowledge-ingestion-test/model/checkpoint-178"
assert os.path.exists(STUDENT_MODEL_NAME), "Student model not found"
config = PeftConfig.from_pretrained(STUDENT_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# generate logprobs
@torch.no_grad()
def generate_logprobs(input_ids, attention_mask):
    input_ids = torch.tensor(input_ids); attention_mask = torch.tensor(attention_mask)
    input_ids = input_ids.to(model.device); attention_mask = attention_mask.to(model.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    logprobs = logits.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logprobs

# ===
# API
# ===
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class LogprobsRequest(BaseModel):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]

class LogprobsResponse(BaseModel):
    logprobs: List[List[float]]

@app.post("/logprobs")
def get_logprobs(request: LogprobsRequest) -> LogprobsResponse:
    logprobs = generate_logprobs(request.input_ids, request.attention_mask)
    return LogprobsResponse(logprobs=logprobs.tolist())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)