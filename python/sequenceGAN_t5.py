# -*- coding: utf-8 -*-
"""
完全統一使用T5的SeqGAN主程序 - 修改版，支持滑動窗口，添加內存優化
"""
import sys
import os
from datetime import datetime
import torch
import pandas as pd
from config import TOTAL_BATCH, DIS_NUM_EPOCH, DEVICE, PATH, openLog, T5_TASK_PREFIX, SEQ_LENGTH, openLog
from data_processing import process_text_file, decode_texts, get_tokenizer, post_process_chinese_text, prepare_sliding_window_data
from discriminator import train_discriminator
from generator_t5 import T5Generator, pretrain_generator, train_generator_with_rewards, set_epoch
from rollout import T5Rollout, get_rewards

# 啟用內存優化設置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 實現自動內存管理
def check_gpu_memory():
    """檢查當前GPU內存使用情況並清理"""
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_cached() / 1024**3
    
    log = openLog()
    log.write(f"GPU內存: 已分配 {allocated:.2f} GB, 緩存 {cached:.2f} GB\n")
    log.close()
    
    return allocated, cached

# 對主要功能函數添加自動內存管理
def memory_managed_function(func):
    """裝飾器: 在函數執行前後檢查並清理內存"""
    def wrapper(*args, **kwargs):
        # 執行前清理
        check_gpu_memory()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 執行後清理
            torch.cuda.empty_cache()
            check_gpu_memory()
    
    return wrapper

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

@memory_managed_function
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
    
    # 動態調整批次大小
    if batch_size > len(token_ids):
        batch_size = len(token_ids)
    if batch_size > 8:  # 安全限制批次大小
        batch_size = 8
        log.write("批次大小已限制為8，以避免內存問題\n")
    
    log.write("###### start to pretrain T5 generator: {}\n".format(datetime.now()))
    log.close()

    # 清理內存
    torch.cuda.empty_cache()
    
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

    try:
        # 清理GPU記憶體
        torch.cuda.empty_cache()
        
        # 從討論串留言數量統計中找出最適合的樣本
        discussion_counts = train_data.get('discussion_counts', {})
        samples_by_count = {}
        
        # 將樣本按留言數量分組
        for i, sample in enumerate(train_data['samples']):
            message_count = sample.get('message_count', 0)
            # 將留言數量分組 (0-5, 6-10, 11-15, 16+)
            group_key = min(message_count // 5, 3)
            if group_key not in samples_by_count:
                samples_by_count[group_key] = []
            samples_by_count[group_key].append(i)
        
        # 優先選擇留言數量少的樣本，以減少內存消耗
        selected_samples = []
        target_count = min(10, len(token_ids))  # 限制總樣本數
        
        # 從留言數量最少的組開始選擇
        for group_key in sorted(samples_by_count.keys()):
            indices = samples_by_count[group_key]
            # 隨機選擇這個組中的樣本
            selected = torch.randperm(len(indices))[:min(len(indices), target_count - len(selected_samples))]
            selected_samples.extend([indices[i.item()] for i in selected])
            if len(selected_samples) >= target_count:
                break
        
        fake_samples_list = []
        fake_mask_list = []
        
        with torch.no_grad():
            for i, sample_idx in enumerate(selected_samples):
                sample = train_data['samples'][sample_idx]
                sample_text = f"[文章摘要] {sample['post'][:50]}"
                
                log = openLog()
                log.write(f"  生成樣本 {i+1}/{len(selected_samples)}, 留言數: {sample.get('message_count', 0)}\n")
                log.close()
                
                # 每次只處理一個樣本
                input_text = prepare_generation_input(sample_text)
                
                # 生成文本
                sample_output = generator.module.generate(
                    input_text=[input_text],
                    max_length=SEQ_LENGTH // 2  # 縮短生成長度
                )
                
                # 確保維度正確
                sample_output, sample_mask = ensure_length(sample_output, SEQ_LENGTH, pad_token_id)
                if sample_mask is None:
                    sample_mask = torch.ones_like(sample_output)
                
                fake_samples_list.append(sample_output)
                fake_mask_list.append(sample_mask)
                
                # 立即清理內存
                torch.cuda.empty_cache()
        
        # 合併所有生成的樣本
        fake_samples = torch.cat(fake_samples_list, dim=0)
        fake_mask = torch.cat(fake_mask_list, dim=0)
        
        # 如果需要更多樣本，複製現有樣本
        if fake_samples.size(0) < len(token_ids):
            repeat_times = (len(token_ids) + fake_samples.size(0) - 1) // fake_samples.size(0)
            fake_samples = fake_samples.repeat(repeat_times, 1)[:len(token_ids)]
            fake_mask = fake_mask.repeat(repeat_times, 1)[:len(token_ids)]
            
    except Exception as e:
        log = openLog()
        log.write(f"Error generating fake samples: {str(e)}\n")
        log.close()
        raise e
    
    # 預訓練判別器
    log = openLog()
    log.write("###### start to pretrain discriminator: {}\n".format(datetime.now()))
    log.close()
    
    # 清理內存
    torch.cuda.empty_cache()
    
    # 減少判別器訓練的批次大小
    dis_batch_size = max(1, batch_size // 2)
    
    discriminator = train_discriminator(
        real_data=token_ids,
        fake_data=fake_samples,
        real_mask=attention_mask,
        fake_mask=fake_mask,
        batch_size=dis_batch_size
    )
    
    # 初始化rollout
    log = openLog()
    log.write("###### initializing rollout: {}\n".format(datetime.now()))
    log.close()
    
    torch.cuda.empty_cache()
    
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
        
        # 清理內存
        torch.cuda.empty_cache()
        
        # 更新epoch計數 - 用於損失權重調整
        set_epoch(generator, total_batch, TOTAL_BATCH)
        
        # 生成樣本
        with torch.no_grad():
            # 針對每個批次單獨生成和訓練
            micro_batch_size = 1  # 極小批次大小
            adv_samples_list = []
            adv_masks_list = []
            rewards_list = []
            
            # 優先選擇留言數量少的樣本進行生成
            selected_indices = []
            for group_key in sorted(samples_by_count.keys()):
                indices = samples_by_count[group_key]
                # 隨機選擇這個組中的樣本
                select_count = min(len(indices), batch_size - len(selected_indices))
                if select_count > 0:
                    idx_selected = torch.randperm(len(indices))[:select_count]
                    selected_indices.extend([indices[i.item()] for i in idx_selected])
                if len(selected_indices) >= batch_size:
                    break
            
            # 對每個選定的樣本進行操作
            for i, sample_idx in enumerate(selected_indices):
                try:
                    sample = train_data['samples'][sample_idx]
                    sample_text = f"[文章摘要] {sample['post'][:50]}"
                    
                    # 記錄生成信息
                    log = openLog()
                    log.write(f"  對抗訓練樣本 {i+1}/{len(selected_indices)}\n")
                    log.close()
                    
                    # 為當前樣本準備輸入
                    input_text = prepare_generation_input(sample_text)
                    
                    # 為當前樣本生成文本
                    sample_output = generator.module.generate(
                        input_text=[input_text],
                        max_length=SEQ_LENGTH // 2
                    )
                    
                    # 確保維度正確
                    sample_output, sample_mask = ensure_length(sample_output, SEQ_LENGTH, pad_token_id)
                    if sample_mask is None:
                        sample_mask = torch.ones_like(sample_output)
                    
                    # 為當前樣本計算獎勵
                    sample_reward = get_rewards(sample_output, rollout.module, discriminator)
                    
                    # 保存結果
                    adv_samples_list.append(sample_output)
                    adv_masks_list.append(sample_mask)
                    rewards_list.append(sample_reward)
                    
                    # 立即清理內存
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    log = openLog()
                    log.write(f"  樣本生成錯誤: {str(e)}\n")
                    log.close()
                    continue
            
            # 合併所有樣本
            if adv_samples_list:
                samples = torch.cat(adv_samples_list, dim=0)
                samples_mask = torch.cat(adv_masks_list, dim=0)
                rewards = torch.cat(rewards_list, dim=0)
            else:
                log = openLog()
                log.write("  無法生成任何樣本，跳過當前批次\n")
                log.close()
                continue
        
        # 訓練生成器
        log = openLog()
        log.write("  訓練生成器\n")
        log.close()
        
        torch.cuda.empty_cache()
        
        generator = train_generator_with_rewards(
            generator=generator,
            input_data=samples,
            rewards=rewards,
            attention_mask=samples_mask,
            batch_size=micro_batch_size  # 使用極小批次
        )
        
        # 更新rollout
        log = openLog()
        log.write("  更新rollout\n")
        log.close()
        
        torch.cuda.empty_cache()
        
        rollout.module.update_params(generator)
        
        # 訓練判別器
        log = openLog()
        log.write("  訓練判別器\n")
        log.close()
        
        torch.cuda.empty_cache()
        
        for iter_n_dis in range(DIS_NUM_EPOCH):
            log = openLog()
            log.write('  iter_n_dis: {} : {}\n'.format(iter_n_dis, datetime.now()))
            log.close()
            
            # 生成新樣本 - 使用相同的動態生成方式
            dis_samples_list = []
            dis_masks_list = []
            
            with torch.no_grad():
                # 對每個選定的樣本進行操作
                for i, sample_idx in enumerate(selected_indices):
                    try:
                        sample = train_data['samples'][sample_idx]
                        sample_text = f"[文章摘要] {sample['post'][:50]}"
                        
                        # 為當前樣本準備輸入
                        input_text = prepare_generation_input(sample_text)
                        
                        # 為當前樣本生成文本
                        sample_output = generator.module.generate(
                            input_text=[input_text],
                            max_length=SEQ_LENGTH // 2
                        )
                        
                        # 確保維度正確
                        sample_output, sample_mask = ensure_length(sample_output, SEQ_LENGTH, pad_token_id)
                        if sample_mask is None:
                            sample_mask = torch.ones_like(sample_output)
                        
                        # 保存結果
                        dis_samples_list.append(sample_output)
                        dis_masks_list.append(sample_mask)
                        
                        # 立即清理內存
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        log = openLog()
                        log.write(f"  判別器樣本生成錯誤: {str(e)}\n")
                        log.close()
                        continue
                
                # 合併所有樣本
                if dis_samples_list:
                    new_samples = torch.cat(dis_samples_list, dim=0)
                    new_mask = torch.cat(dis_masks_list, dim=0)
                else:
                    log = openLog()
                    log.write("  無法為判別器生成樣本，跳過當前迭代\n")
                    log.close()
                    continue
            
            # 訓練判別器
            discriminator = train_discriminator(
                real_data=token_ids[:len(new_samples)],  # 確保真實樣本和假樣本數量一致
                fake_data=new_samples,
                real_mask=attention_mask[:len(new_samples)],
                fake_mask=new_mask,
                batch_size=micro_batch_size
            )
    
    # 訓練完成
    log = openLog()
    log.write('###### training done: {}\n'.format(datetime.now()))
    log.close()
    
    # 清理內存
    torch.cuda.empty_cache()
    
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
        # 準備生成輸入 - 選擇留言數量最少的樣本
        min_count_group = min(samples_by_count.keys())
        final_indices = samples_by_count[min_count_group][:batch_size]
        final_texts = []
        
        for idx in final_indices:
            sample = train_data['samples'][idx]
            final_texts.append(f"[文章摘要] {sample['post'][:50]}")
            
        # 使用統一的輸入準備函數
        input_texts = [prepare_generation_input(text) for text in final_texts]
        
        # 逐一生成避免內存問題
        final_samples_list = []
        for i, text in enumerate(input_texts):
            log.write(f"生成最終樣本 {i+1}/{len(input_texts)}\n")
            sample = generator.module.generate(
                input_text=[text],
                max_length=SEQ_LENGTH
            )
            final_samples_list.append(sample)
            torch.cuda.empty_cache()
        
        # 合併生成的樣本
        if final_samples_list:
            final_samples = torch.cat(final_samples_list, dim=0)
        else:
            log.write("無法生成任何最終樣本\n")
            log.close()
            return
    
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