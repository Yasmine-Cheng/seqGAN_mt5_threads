# -*- coding: utf-8 -*-
"""
純T5生成器實現，不涉及任何格式轉換 - 修改版：實現滑動窗口式預訓練
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, BertTokenizer
from config import SEQ_LENGTH, DEVICE, GEN_NUM_EPOCH, openLog, T5_TASK_PREFIX, T5_MODEL_NAME, T5_LEARNING_RATE, MAX_WINDOW_SIZE
from data_processing import get_tokenizer, prepare_chinese_dialogue_pairs, post_process_chinese_text, prepare_sliding_window_data

class T5Generator(nn.Module):
    def __init__(self, model_name=T5_MODEL_NAME, task_prefix=T5_TASK_PREFIX):
        super().__init__()
        # 初始化T5模型和tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = get_tokenizer()
        self.task_prefix = task_prefix
        self.current_epoch = 0
        self.total_epochs = 10
        
    def forward(self, input_ids, attention_mask=None, labels=None, rewards=None):
        """前向傳播"""
        # 確保labels是long類型
        if labels is not None:
            labels = labels.long()
            
        # 標準T5前向傳播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 如果沒有提供獎勵，返回標準輸出
        if rewards is None:
            return outputs
            
        # 如果提供了獎勵，計算策略梯度損失
        loss = outputs.loss
        logits = outputs.logits
        
        # 計算策略梯度損失 (RL部分)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 獲取所選token的log概率
        selected_log_probs = torch.gather(
            log_probs.view(-1, log_probs.size(-1)),
            1,
            labels.view(-1, 1)
        ).view(labels.size())
        
        # 應用獎勵
        rl_loss = -(selected_log_probs * rewards).mean()
        
        # 動態調整權重
        alpha = max(0.2, 1 - (self.current_epoch / self.total_epochs))  # 從1逐漸減至0.2
        
        # 結合標準損失與RL損失
        combined_loss = alpha * loss + (1-alpha) * rl_loss
        
        # 創建新的輸出對象，包含組合損失
        class CombinedOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
                
        return CombinedOutput(combined_loss, logits)
    
    def generate(self, input_ids=None, attention_mask=None, input_text=None,
                batch_size=1, max_length=SEQ_LENGTH, temperature=1.1, top_p=0.9):
        """生成文本"""
        # 如果提供了文本而不是token IDs，先進行編碼
        if input_text is not None:
            if isinstance(input_text, str):
                input_text = [input_text]
                
            # 添加任務前綴
            prefixed_texts = [self.task_prefix + text for text in input_text]
            encodings = self.tokenizer(
                prefixed_texts,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
        # 如果未提供輸入，生成START token序列
        if input_ids is None:
            start_text = ["START"] * batch_size
            prefixed_texts = [self.task_prefix + text for text in start_text]
            encodings = self.tokenizer(
                prefixed_texts,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=0.9,  # 降低溫度，獲得更連貫的文本
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.5,  # 增加重複懲罰
                no_repeat_ngram_size=3,  # 防止3-gram重複
                num_return_sequences=batch_size,
                length_penalty=1.5  # 鼓勵生成更長的序列
            )
            
        return outputs

def process_generated_text(texts, tokenizer):
    """處理生成的文本，清理extra標記"""
    if isinstance(texts, torch.Tensor):
        # 如果是tensor，先用tokenizer解碼
        texts = tokenizer.batch_decode(texts, skip_special_tokens=True)
    
    processed_texts = []
    for text in texts:
        # 移除extra標記
        for i in range(100):
            text = text.replace(f"extra{i}", "")
        
        # 移除特殊標記
        text = text.replace("[PAD]", "")
        text = text.replace("[CLS]", "")
        text = text.replace("[SEP]", "")
        text = text.replace("[UNK]", "")
        text = text.replace("[MASK]", "")
        
        # 移除任務前綴和START標記
        text = text.replace(T5_TASK_PREFIX, "")
        text = text.replace("START", "")
        
        # 移除討論標記
        text = text.replace("[文章摘要]", "")
        for i in range(10):  # 假設最多10樓
            text = text.replace(f"[留言_{i}樓]", "")
        
        # 整理空格
        text = ' '.join(text.split())
        
        processed_texts.append(text.strip())
    
    return processed_texts

def set_epoch(model, epoch, total):
    """設置當前epoch用於損失權重調整"""
    if hasattr(model, 'module'):
        model.module.current_epoch = epoch
        model.module.total_epochs = total
    else:
        model.current_epoch = epoch
        model.total_epochs = total

def pretrain_generator_with_sliding_windows(file_path, batch_size=8, epochs=GEN_NUM_EPOCH):
    """使用滑動窗口策略預訓練T5生成器"""
    # 創建生成器和tokenizer
    generator = T5Generator()
    generator = nn.DataParallel(generator)
    generator.to(DEVICE)
    tokenizer = get_tokenizer()
    
    # 獲取滑動窗口訓練數據
    train_data = prepare_sliding_window_data(file_path, tokenizer)
    
    # 創建優化器
    optimizer = torch.optim.AdamW(generator.parameters(), lr=T5_LEARNING_RATE)
    
    # 開始訓練
    log = openLog()
    log.write('    pretraining T5 generator with sliding windows: {}\n'.format(datetime.now()))
    
    input_ids = train_data['input_ids']
    attention_mask = train_data['attention_mask']
    labels = train_data['labels']
    
    # 訓練循環
    for epoch in range(epochs):
        # 設置當前epoch
        set_epoch(generator, epoch, epochs)
        
        # 打亂資料順序
        indices = torch.randperm(len(input_ids))
        shuffled_input_ids = input_ids[indices]
        shuffled_attention_mask = attention_mask[indices]
        shuffled_labels = labels[indices]
        
        pointer = 0
        total_loss = 0
        batch_count = 0
        
        while pointer + batch_size <= len(shuffled_input_ids):
            # 獲取批次
            batch_input_ids = shuffled_input_ids[pointer:pointer+batch_size]
            batch_attention_mask = shuffled_attention_mask[pointer:pointer+batch_size]
            batch_labels = shuffled_labels[pointer:pointer+batch_size]
            
            # 模型訓練
            outputs = generator(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )
            
            loss = outputs.loss
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            
            # 更新統計
            total_loss += loss.item()
            batch_count += 1
            pointer += batch_size
            
        # 紀錄進度
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log.write('      epoch: {} loss: {:.4f}\n'.format(epoch+1, avg_loss))
    
    log.close()
    return generator

# 原始預訓練函數-保留以保持兼容性
def pretrain_generator(train_data=None, attention_mask=None, batch_size=8, epochs=GEN_NUM_EPOCH, file_path=None):
    """預訓練T5生成器 - 如果提供file_path則使用滑動窗口策略，否則使用原有策略"""
    if file_path is not None:
        return pretrain_generator_with_sliding_windows(file_path, batch_size, epochs)
    
    # 創建生成器和tokenizer
    generator = T5Generator()
    generator = nn.DataParallel(generator)
    generator.to(DEVICE)
    tokenizer = get_tokenizer()
    
    # 創建優化器
    optimizer = torch.optim.AdamW(generator.parameters(), lr=T5_LEARNING_RATE)
    
    # 開始訓練
    log = openLog()
    log.write('    pretraining T5 generator with dialogue style: {}\n'.format(datetime.now()))
    
    # 將訓練數據轉換為對話格式 - 使用中文特定的對話對構造方法
    dialogue_pairs = prepare_chinese_dialogue_pairs(train_data, tokenizer)
    
    for epoch in range(epochs):
        # 設置當前epoch
        set_epoch(generator, epoch, epochs)
        
        # 打亂對話對順序
        indices = torch.randperm(len(dialogue_pairs))
        shuffled_pairs = [dialogue_pairs[i] for i in indices]
        
        pointer = 0
        total_loss = 0
        batch_count = 0
        
        while pointer + batch_size <= len(shuffled_pairs):
            # 獲取批次
            batch_pairs = shuffled_pairs[pointer:pointer+batch_size]
            
            # 分離上下文和回應
            contexts, responses = zip(*batch_pairs)
            
            # 添加任務前綴並編碼
            prefixed_contexts = [f"{T5_TASK_PREFIX}{ctx}" for ctx in contexts]
            context_encodings = tokenizer(
                prefixed_contexts,
                padding="max_length",
                truncation=True,
                max_length=SEQ_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)
            
            # 編碼回應作為標籤
            response_encodings = tokenizer(
                responses,
                padding="max_length",
                truncation=True,
                max_length=SEQ_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)
            
            # 模型訓練 - 輸入上下文，預測回應
            outputs = generator(
                input_ids=context_encodings.input_ids,
                attention_mask=context_encodings.attention_mask,
                labels=response_encodings.input_ids
            )
            
            loss = outputs.loss
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            
            # 更新統計
            total_loss += loss.item()
            batch_count += 1
            pointer += batch_size
            
        # 紀錄進度
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log.write('      epoch: {} loss: {:.4f}\n'.format(epoch+1, avg_loss))
    
    log.close()
    return generator

def train_generator_with_rewards(generator, input_data, rewards, attention_mask=None, batch_size=8, epochs=1):
    """使用獎勵訓練生成器"""
    # 確保輸入是批次
    if len(input_data.shape) == 1:
        input_data = input_data.unsqueeze(0)
        rewards = rewards.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
            
    # 優化器
    optimizer = torch.optim.AdamW(generator.parameters(), lr=T5_LEARNING_RATE)
    
    # 訓練循環
    log = openLog()
    log.write('    training generator with rewards: {}\n'.format(datetime.now()))
    
    for epoch in range(epochs):
        # 更新epoch計數 - 用於損失權重調整
        if hasattr(generator, 'module'):
            current_epoch = generator.module.current_epoch
            total_epochs = generator.module.total_epochs
            generator.module.current_epoch = min(current_epoch + 1, total_epochs)
        else:
            current_epoch = generator.current_epoch
            total_epochs = generator.total_epochs
            generator.current_epoch = min(current_epoch + 1, total_epochs)
            
        pointer = 0
        total_loss = 0
        batch_count = 0
        
        while pointer + batch_size <= len(input_data):
            # 獲取批次
            batch_input = input_data[pointer:pointer+batch_size]
            batch_rewards = rewards[pointer:pointer+batch_size]
            batch_mask = None
            if attention_mask is not None:
                batch_mask = attention_mask[pointer:pointer+batch_size]
                
            # 前向傳播，包含獎勵
            outputs = generator(
                input_ids=batch_input,
                attention_mask=batch_mask,
                labels=batch_input.clone(),
                rewards=batch_rewards
            )
            
            loss = outputs.loss
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            
            # 更新統計
            total_loss += loss.item()
            batch_count += 1
            pointer += batch_size
            
        # 紀錄進度
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log.write('      epoch: {} loss: {:.4f}\n'.format(epoch+1, avg_loss))
    
    log.close()
    return generator

def sanityCheck_T5Generator(batch_size=1, sample_size=5):
    """測試T5生成器的功能"""
    log = openLog('test.txt')
    log.write('\n\nTest generator_t5.sanityCheck_T5Generator: {}\n'.format(datetime.now()))
    
    # 創建和測試生成器
    generator = T5Generator()
    generator.to(DEVICE)
    
    # 生成測試文本
    test_texts = generator.generate(
        input_text=["START"] * batch_size, 
        batch_size=batch_size
    )
    
    # 解碼生成的文本
    tokenizer = get_tokenizer()
    generated_texts = tokenizer.batch_decode(test_texts, skip_special_tokens=True)
    
    # 後處理文本，處理extra標記
    processed_texts = process_generated_text(generated_texts, tokenizer)
    
    log.write("Generated texts:\n")
    for i, text in enumerate(generated_texts):
        log.write(f"{i+1}. {text}\n")
        
    log.write("\nProcessed texts:\n")
    for i, text in enumerate(processed_texts):
        log.write(f"{i+1}. {text}\n")
    
    log.close()
    
    return generator, test_texts, processed_texts

if __name__ == '__main__':
    model, tokens, texts = sanityCheck_T5Generator(batch_size=2, sample_size=6)
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")