# -*- coding: utf-8 -*-
"""
完全統一使用T5的SeqGAN主程序 - 修改版，支持滑動窗口
"""
import sys
from datetime import datetime
import torch
import pandas as pd
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, PATH, openLog, T5_TASK_PREFIX, SEQ_LENGTH
from data_processing import process_text_file, decode_texts, get_tokenizer, post_process_chinese_text, prepare_sliding_window_data
from discriminator import train_discriminator
from generator_t5 import T5Generator, pretrain_generator, train_generator_with_rewards, set_epoch
from rollout import T5Rollout, get_rewards

def ensure_length(tensor, target_length, pad_token_id):
    """確保張量有正確的序列長度，通過填充或截斷"""
    if tensor.size(1) == target_length:
        return tensor, None  # 不需要修改
        
    if tensor.size(1) < target_length:
        # 創建填充張量
        padding = torch.ones(tensor.size(0), target_length - tensor.size(1), 
                           dtype=tensor.dtype, device=DEVICE) * pad_token_id
        padded_tensor = torch.cat([tensor, padding], dim=1)
        
        # 創建新的注意力掩碼
        attention_mask = torch.ones_like(padded_tensor)
        attention_mask[:, tensor.size(1):] = 0  # 填充部分為0
        
        return padded_tensor, attention_mask
    else:
        # 截斷到目標長度
        return tensor[:, :target_length], None

def prepare_generation_input(text, task_prefix=T5_TASK_PREFIX):
    """為T5模型準備生成輸入"""
    # 確保使用正確的前綴
    if not text.startswith(task_prefix):
        text = task_prefix + text
    return text

def main(batch_size, input_file=None, num=None):
    """主函數"""
    if batch_size is None:
        batch_size = 1
    
    # 初始化tokenizer
    tokenizer = get_tokenizer()
    pad_token_id = tokenizer.pad_token_id
    
    # 確認輸入檔案
    if input_file is None:
        input_file = "ptt_real.csv"  # 預設的討論串資料檔
    
    # 讀取數據
    log = openLog()
    log.write("###### loading and processing data: {}\n".format(datetime.now()))
    
    # 判斷是否使用滑動窗口預處理
    if input_file.endswith('.csv'):  # 假設CSV檔案是討論串格式
        # 使用滑動窗口預處理
        log.write("###### using sliding window preprocessing for discussion data\n")
        train_data = prepare_sliding_window_data(input_file, tokenizer, max_tokens=SEQ_LENGTH)
        token_ids = train_data['input_ids']
        attention_mask = train_data['attention_mask']
    else:
        # 使用傳統預處理
        log.write("###### using traditional preprocessing\n")
        token_ids, attention_mask, texts, _ = process_text_file(
            input_file,
            num=num,
            max_length=SEQ_LENGTH,
            add_task_prefix=True
        )
    
    if batch_size > len(token_ids):
        batch_size = len(token_ids)
    
    log.write("###### start to pretrain T5 generator: {}\n".format(datetime.now()))
    log.close()
    
    # 預訓練生成器 - 使用滑動窗口或對話風格
    if input_file.endswith('.csv'):
        # 使用滑動窗口預訓練
        generator = pretrain_generator(
            file_path=input_file,
            batch_size=batch_size
        )
    else:
        # 使用傳統對話風格預訓練
        generator = pretrain_generator(
            train_data=token_ids,
            attention_mask=attention_mask,
            batch_size=batch_size
        )
    
    # 生成假樣本
    log = openLog()
    log.write("###### generating fake samples: {}\n".format(datetime.now()))
    log.close()
    
    with torch.no_grad():
        # 準備生成輸入
        if input_file.endswith('.csv'):
            # 從訓練樣本中提取摘要文本用於生成
            sample_texts = []
            for sample in train_data['samples'][:min(len(train_data['samples']), len(token_ids))]:
                sample_texts.append(f"[文章摘要] {sample['post'][:100]}")  # 使用文章摘要作為輸入
        else:
            sample_texts = ["START"] * len(token_ids)
            
        # 使用統一的輸入準備函數
        input_texts = [prepare_generation_input(text) for text in sample_texts]
        fake_samples = generator.module.generate(
            input_text=input_texts,
            max_length=SEQ_LENGTH
        )
        
        # 確保維度正確
        fake_samples, fake_mask = ensure_length(fake_samples, SEQ_LENGTH, pad_token_id)
        if fake_mask is None:
            fake_mask = torch.ones_like(fake_samples)
    
    # 預訓練判別器
    log = openLog()
    log.write("###### start to pretrain discriminator: {}\n".format(datetime.now()))
    log.close()
    
    discriminator = train_discriminator(
        real_data=token_ids,
        fake_data=fake_samples,
        real_mask=attention_mask,
        fake_mask=fake_mask,
        batch_size=batch_size
    )
    
    # 初始化rollout
    rollout = T5Rollout(generator)
    rollout = torch.nn.DataParallel(rollout)
    rollout.to(DEVICE)
    
    # 對抗訓練
    log = openLog()
    log.write("###### start adversarial training: {}\n".format(datetime.now()))
    log.close()
    
    for total_batch in range(TOTAL_BATCH):
        log = openLog()
        log.write('batch: {} : {}\n'.format(total_batch, datetime.now()))
        print('batch: {} : {}\n'.format(total_batch, datetime.now()))
        log.close()
        
        # 更新epoch計數 - 用於損失權重調整
        set_epoch(generator, total_batch, TOTAL_BATCH)
        
        # 生成樣本
        with torch.no_grad():
            # 準備生成輸入
            if input_file.endswith('.csv'):
                # 隨機選擇訓練樣本來生成新樣本
                indices = torch.randperm(len(train_data['samples']))[:batch_size]
                gen_texts = []
                for i in indices:
                    sample = train_data['samples'][i]
                    gen_texts.append(f"[文章摘要] {sample['post'][:100]}")
            else:
                gen_texts = ["START"] * batch_size
                
            # 使用統一的輸入準備函數
            input_texts = [prepare_generation_input(text) for text in gen_texts]
            samples = generator.module.generate(
                input_text=input_texts,
                max_length=SEQ_LENGTH
            )
            
            # 確保維度正確
            samples, samples_mask = ensure_length(samples, SEQ_LENGTH, pad_token_id)
        
        # 計算獎勵
        rewards = get_rewards(samples, rollout.module, discriminator)
        
        # 訓練生成器
        generator = train_generator_with_rewards(
            generator=generator,
            input_data=samples,
            rewards=rewards,
            attention_mask=samples_mask,
            batch_size=batch_size
        )
        
        # 更新rollout
        rollout.module.update_params(generator)
        
        # 訓練判別器
        for iter_n_dis in range(DIS_NUM_EPOCH):
            log = openLog()
            log.write('  iter_n_dis: {} : {}\n'.format(iter_n_dis, datetime.now()))
            log.close()
            
            # 生成新樣本
            with torch.no_grad():
                # 準備生成輸入
                if input_file.endswith('.csv'):
                    # 隨機選擇訓練樣本來生成新樣本
                    indices = torch.randperm(len(train_data['samples']))[:len(token_ids)]
                    gen_texts = []
                    for i in indices:
                        sample = train_data['samples'][i]
                        gen_texts.append(f"[文章摘要] {sample['post'][:100]}")
                else:
                    gen_texts = ["START"] * len(token_ids)
                    
                # 使用統一的輸入準備函數
                input_texts = [prepare_generation_input(text) for text in gen_texts]
                new_samples = generator.module.generate(
                    input_text=input_texts,
                    max_length=SEQ_LENGTH
                )
                
                # 確保維度正確
                new_samples, new_mask = ensure_length(new_samples, SEQ_LENGTH, pad_token_id)
            
            # 訓練判別器
            discriminator = train_discriminator(
                real_data=token_ids,
                fake_data=new_samples,
                real_mask=attention_mask,
                fake_mask=new_mask,
                batch_size=batch_size
            )
    
    # 訓練完成
    log = openLog()
    log.write('###### training done: {}\n'.format(datetime.now()))
    log.close()
    
    # 保存模型
    try:
        torch.save(generator, PATH + 't5_generator.pkl')
        torch.save(tokenizer, PATH + 't5_tokenizer.pkl')
        print('successfully saved T5 generator model.')
    except Exception as e:
        print('error: model saving failed!!!!!!', str(e))
    
    # 生成最終樣本
    log = openLog('genTxt_t5.txt')
    with torch.no_grad():
        # 準備生成輸入
        if input_file.endswith('.csv'):
            # 使用訓練樣本來生成最終樣本
            final_texts = []
            for i in range(batch_size):
                idx = i % len(train_data['samples'])
                sample = train_data['samples'][idx]
                final_texts.append(f"[文章摘要] {sample['post'][:100]}")
        else:
            final_texts = ["START"] * batch_size
            
        # 使用統一的輸入準備函數
        input_texts = [prepare_generation_input(text) for text in final_texts]
        final_samples = generator.module.generate(
            input_text=input_texts,
            max_length=SEQ_LENGTH
        )
    
    # 解碼和記錄生成的文本
    generated_texts = decode_texts(final_samples, tokenizer, log)
    
    # 後處理文本並記錄
    processed_texts = [post_process_chinese_text(text) for text in generated_texts]
    log.write("\n後處理的生成文本:\n")
    for text in processed_texts:
        log.write(text + '\n')
        
    log.close()
    print(processed_texts)

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    try:
        input_file = sys.argv[2]
    except IndexError:
        input_file = "ptt_real.csv"
    try:
        num = int(sys.argv[3])
    except IndexError:
        num = 10
        
    main(batch_size, input_file, num)