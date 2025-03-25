# -*- coding: utf-8 -*-
"""
T5版本的Rollout機制 - 修改為支持討論串格式
"""
import torch
import torch.nn as nn
from datetime import datetime
from transformers import T5ForConditionalGeneration, BertTokenizer

from config import DEVICE, ROLLOUT_ITER, SEQ_LENGTH, openLog, T5_TASK_PREFIX
from data_processing import get_tokenizer, post_process_chinese_text

class T5Rollout(nn.Module):
    def __init__(self, generator=None, r_update_rate=0.8):
        super().__init__()
        
        if generator is not None:
            # 從生成器獲取必要的組件
            if hasattr(generator, 'module'):
                base_model = generator.module.model
                self.tokenizer = generator.module.tokenizer
                self.task_prefix = generator.module.task_prefix
            else:
                base_model = generator.model
                self.tokenizer = generator.tokenizer
                self.task_prefix = generator.task_prefix
                
            # 創建相同架構的模型
            self.model = T5ForConditionalGeneration.from_pretrained(base_model.config._name_or_path)
            # 複製權重
            self.model.load_state_dict(base_model.state_dict())
            self.model.to(DEVICE)
        else:
            # 默認初始化
            self.tokenizer = get_tokenizer()
            self.task_prefix = T5_TASK_PREFIX
            self.model = None
            
        self.r_update_rate = r_update_rate
    
    def rollout_generation(self, prefix_ids, given_num, max_length=SEQ_LENGTH):
        """從前given_num個token開始，生成後續文本"""
        # 解碼前給定的token
        batch_size = prefix_ids.size(0)
        prefix_text = self.tokenizer.batch_decode(prefix_ids[:, :given_num], skip_special_tokens=True)
        
        # 添加任務前綴
        prefixed_texts = [self.task_prefix + text for text in prefix_text]
        
        # 編碼
        encodings = self.tokenizer(
            prefixed_texts,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_length=max_length,  # 使用指定的max_length
                do_sample=True,
                temperature=1.1,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                num_return_sequences=1
            )
            
        # 確保輸出長度正確
        pad_token_id = self.tokenizer.pad_token_id
        if outputs.size(1) != max_length:
            if outputs.size(1) < max_length:
                # 填充
                padding = torch.ones(outputs.size(0), max_length - outputs.size(1), 
                                   dtype=outputs.dtype, device=DEVICE) * pad_token_id
                outputs = torch.cat([outputs, padding], dim=1)
            else:
                # 截斷
                outputs = outputs[:, :max_length]
        
        return outputs
    
    def update_params(self, generator):
        """更新模型參數"""
        if hasattr(generator, 'module'):
            self.model.load_state_dict(generator.module.model.state_dict())
        else:
            self.model.load_state_dict(generator.model.state_dict())

def get_rewards(samples, rollout, discriminator):
    """計算獎勵"""
    batch_size = samples.size(0)
    seq_length = samples.size(1)
    rewards = torch.zeros(batch_size, seq_length, device=DEVICE)
    
    # 對每個位置進行rollout
    for i in range(ROLLOUT_ITER):
        for given_num in range(1, seq_length):
            # 從給定的token生成完整序列
            rollout_samples = rollout.rollout_generation(samples, given_num)
            
            # 使用判別器評估
            discrimination = discriminator(rollout_samples)
            
            # 獲取正類別的概率作為獎勵
            real_prob = discrimination[:, 1]
            rewards[:, given_num-1] += real_prob
            
        # 對完整序列的評估
        discrimination = discriminator(samples)
        real_prob = discrimination[:, 1]
        rewards[:, seq_length-1] += real_prob
        
    # 計算平均獎勵
    rewards = rewards / ROLLOUT_ITER
    
    return rewards

def sanityCheck_T5Rollout(batch_size=2):
    """測試T5 Rollout功能"""
    log = openLog('test.txt')
    log.write('\n\nTest rollout.sanityCheck_T5Rollout: {}\n'.format(datetime.now()))
    
    try:
        # 導入T5Generator
        from generator_t5 import T5Generator
        
        # 創建生成器和rollout
        generator = T5Generator()
        generator.to(DEVICE)
        
        rollout = T5Rollout(generator)
        rollout.to(DEVICE)
        
        # 生成一些樣本
        samples = generator.generate(input_text=["START"] * batch_size)
        
        # 測試rollout生成
        given_num = 3  # 使用前3個token
        rollout_samples = rollout.rollout_generation(samples, given_num)
        
        # 解碼並記錄結果
        tokenizer = get_tokenizer()
        original_texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
        rollout_texts = tokenizer.batch_decode(rollout_samples, skip_special_tokens=True)
        
        # 後處理文本
        processed_original = [post_process_chinese_text(text) for text in original_texts]
        processed_rollout = [post_process_chinese_text(text) for text in rollout_texts]
        
        log.write("Original texts:\n")
        for i, text in enumerate(original_texts):
            log.write(f"{i+1}. {text}\n")
            
        log.write("\nProcessed original texts:\n")
        for i, text in enumerate(processed_original):
            log.write(f"{i+1}. {text}\n")
            
        log.write("\nRollout texts (given_num={}):\n".format(given_num))
        for i, text in enumerate(rollout_texts):
            log.write(f"{i+1}. {text}\n")
            
        log.write("\nProcessed rollout texts:\n")
        for i, text in enumerate(processed_rollout):
            log.write(f"{i+1}. {text}\n")
        
        log.write('\n  rollout.sanityCheck_T5Rollout SUCCESSFUL!\n')
        log.close()
        
        return rollout
        
    except Exception as e:
        log.write('\n  rollout.sanityCheck_T5Rollout FAILED: {}\n'.format(str(e)))
        log.close()
        return None

if __name__ == '__main__':
    rollout = sanityCheck_T5Rollout(batch_size=2)