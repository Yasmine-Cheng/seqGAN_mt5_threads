# -*- coding: utf-8 -*-
"""
T5模型核心實現 - 支持討論串處理
"""
from datetime import datetime
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from config import (DEVICE, GEN_NUM_EPOCH_PRETRAIN, openLog, T5_MODEL_NAME, 
                   T5_LEARNING_RATE, T5_TASK_PREFIX, DISCUSSION_TASK_PREFIX)
from data_processing import get_tokenizer, post_process_chinese_text

class T5Model(nn.Module):
    def __init__(self, model_name=T5_MODEL_NAME, is_discussion_model=False):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.is_discussion_model = is_discussion_model
        self.task_prefix = DISCUSSION_TASK_PREFIX if is_discussion_model else T5_TASK_PREFIX
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 確保labels是Long類型
        if labels is not None:
            labels = labels.long()
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate_from_discussion(self, post, tokenizer=None, max_length=100):
        """從討論串貼文生成回應"""
        if tokenizer is None:
            tokenizer = get_tokenizer()
            
        # 準備輸入
        input_text = f"{self.task_prefix}POST: {post}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(DEVICE)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=1.0,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
            
        # 解碼生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 後處理
        processed_text = post_process_chinese_text(generated_text)
        
        return processed_text

def pretrain_T5Model(train_data=None, tokenizer=None, batch_size=4, model_name=T5_MODEL_NAME, is_discussion_model=False):
    """預訓練T5模型"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # 初始化模型
    model = T5Model(model_name=model_name, is_discussion_model=is_discussion_model)
    model = nn.DataParallel(model)
    model.to(DEVICE)
    
    # 如果沒有提供訓練數據，僅返回模型
    if train_data is None:
        return model
        
    # 提取輸入數據和標籤
    input_ids = train_data["input_ids"]
    attention_mask = train_data["attention_mask"]
    labels = train_data["labels"] if "labels" in train_data else input_ids.clone()
    
    # 確保labels是Long類型
    labels = labels.long()
    
    # 準備訓練
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=T5_LEARNING_RATE)
    
    log = openLog()
    model_type = "討論串" if is_discussion_model else "標準"
    log.write(f'    預訓練{model_type}T5模型: {datetime.now()}\n')
    
    # 訓練循環
    for epoch in range(GEN_NUM_EPOCH_PRETRAIN):
        pointer = 0
        epoch_loss = []
        
        while pointer + batch_size <= len(input_ids):
            batch_input_ids = input_ids[pointer:pointer+batch_size]
            batch_attention_mask = attention_mask[pointer:pointer+batch_size]
            batch_labels = labels[pointer:pointer+batch_size]
            
            # 確保批次標籤也是Long類型
            batch_labels = batch_labels.long()
            
            # 訓練步驟
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
            pointer = pointer + batch_size
            
        log.write('      epoch: '+str(epoch)+' loss: '+str(sum(epoch_loss)/len(epoch_loss))+'\n')
    
    log.close()
    return model

# 新增: 專門用於預訓練討論串模型的函數
def pretrain_discussion_T5Model(discussion_threads, tokenizer=None, batch_size=1, model_name=T5_MODEL_NAME):
    """預訓練專門用於討論串的T5模型"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # 準備討論串數據
    train_data = {
        "input_ids": [],
        "attention_mask": [],
    }
    
    for thread in discussion_threads:
        train_data["input_ids"].append(thread["input_ids"])
        train_data["attention_mask"].append(thread["attention_mask"])
    
    # 合併為批次
    train_data["input_ids"] = torch.cat(train_data["input_ids"], dim=0)
    train_data["attention_mask"] = torch.cat(train_data["attention_mask"], dim=0)
    
    # 使用一般預訓練函數，但設置為討論串模型
    return pretrain_T5Model(
        train_data=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        model_name=model_name,
        is_discussion_model=True
    )

def test_genText(model, tokenizer=None, start_text="START", batch_size=1, max_length=100):
    """測試T5模型文本生成"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    log = openLog('test.txt')
    log.write('\n\nTest t5_model.test_genText: {}\n'.format(datetime.now()))
    
    # 準備輸入
    task_prefix = model.module.task_prefix if hasattr(model, 'module') else model.task_prefix
    input_texts = [f"{task_prefix} {start_text}"] * batch_size
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(DEVICE)
    
    # 生成文本
    with torch.no_grad():
        if hasattr(model, 'module'):
            outputs = model.module.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=1.1,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                num_return_sequences=batch_size
            )
        else:
            outputs = model.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=1.1,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                num_return_sequences=batch_size
            )
            
    # 解碼生成的文本
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 後處理文本
    processed_texts = [post_process_chinese_text(text) for text in generated_texts]
    
    # 輸出結果
    for i, text in enumerate(generated_texts):
        log.write(f"\nGenerated text {i+1}: {text}\n")
        
    log.write("\nProcessed texts:\n")
    for i, text in enumerate(processed_texts):
        log.write(f"{i+1}. {text}\n")
    
    log.write('\n  t5_model.test_genText SUCCESSFUL: {}\n'.format(datetime.now()))
    log.close()
    
    return processed_texts

# 新增: 測試討論串生成
def test_discussion_generation(model, tokenizer=None, post="討論主題", batch_size=1):
    """測試討論串回應生成"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    log = openLog('test.txt')
    log.write('\n\nTest t5_model.test_discussion_generation: {}\n'.format(datetime.now()))
    
    # 準備不同的討論主題
    posts = [post] if batch_size == 1 else [f"{post} {i+1}" for i in range(batch_size)]
    
    generated_responses = []
    for p in posts:
        if hasattr(model, 'module'):
            response = model.module.generate_from_discussion(p, tokenizer)
        else:
            response = model.generate_from_discussion(p, tokenizer)
        generated_responses.append(response)
        
        # 記錄到日誌
        log.write(f"\n討論主題: {p}\n")
        log.write(f"生成回應: {response}\n")
        log.write("-" * 50 + "\n")
    
    log.write('\n  t5_model.test_discussion_generation SUCCESSFUL: {}\n'.format(datetime.now()))
    log.close()
    
    return generated_responses

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    model = T5Model()
    model.to(DEVICE)
    generated_texts = test_genText(model, tokenizer, batch_size=2)
    print(generated_texts)