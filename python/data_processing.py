# -*- coding: utf-8 -*-
"""
完全基於T5的數據處理模塊 - 修改版
"""
import os
import shutil
import torch
from transformers import T5ForConditionalGeneration, BertTokenizer  # 修改這裡
import pandas as pd
from config import SEQ_LENGTH, GENERATE_NUM, DEVICE, PATH, T5_MODEL_NAME, T5_TASK_PREFIX, T5_SUMMARY_LENGTH, MAX_WINDOW_SIZE, openLog

# 全局tokenizer
global_tokenizer = None

# 清除之前的tokenizer緩存
def clear_tokenizer_cache():
    global global_tokenizer
    global_tokenizer = None
    # 可選：清除transformers緩存
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir):
        try:
            # 可選的激進清除 - 謹慎使用
            # shutil.rmtree(cache_dir)
            print("建議手動刪除緩存：" + cache_dir)
        except:
            pass
            
def get_tokenizer():
    """獲取或初始化全局tokenizer"""
    global global_tokenizer
    if global_tokenizer is None:
        global_tokenizer = BertTokenizer.from_pretrained(T5_MODEL_NAME)  # 修改這裡
    return global_tokenizer

def process_text_file(file_path, max_length=SEQ_LENGTH, num=None, add_task_prefix=True):
    """使用T5 tokenizer處理文本文件，可選是否添加任務前綴"""
    tokenizer = get_tokenizer()
    raw_texts = []
    
    # 讀取文本文件
    if file_path.endswith(('.pkl', '.csv')):
        if file_path.endswith('.pkl'):
            data = pd.read_pickle(PATH + file_path)
        else:
            data = pd.read_csv(PATH + file_path)
        
        if num is not None:
            data = data[:num]
            
        # 提取並處理文本
        for row in data.values:
            text = ' '.join([str(item) for item in row if pd.notna(item)])
            raw_texts.append(text)
    else:
        # 處理純文本文件
        count = 0
        with open(PATH + file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                raw_texts.append(line.strip())
                count += 1
                if num is not None and count >= num:
                    break
    
    # 統一添加START標記
    processed_texts = ["START " + text for text in raw_texts]
    
    # 可選添加任務前綴
    if add_task_prefix:
        processed_texts = [T5_TASK_PREFIX + text for text in processed_texts]
    
    # 使用T5 tokenizer處理
    encodings = tokenizer(
        processed_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 轉移到設備上
    token_ids = encodings.input_ids.to(DEVICE)
    attention_mask = encodings.attention_mask.to(DEVICE)
    
    return token_ids, attention_mask, processed_texts, tokenizer

def transform_dataset(dataframe):
    """將表格數據轉換為字典格式的討論串"""
    discussions = {}
    current_post = None
    current_responses = []
    
    for idx, row in dataframe.iterrows():
        post = row['post']
        response = row['response']
        
        # 如果是新主題
        if current_post is None or post != current_post:
            # 保存前一個討論（如果存在）
            if current_post is not None:
                discussions[current_post] = current_responses
            
            # 開始新討論
            current_post = post
            current_responses = [response]
        else:
            # 繼續添加到當前討論
            current_responses.append(response)
    
    # 保存最後一個討論
    if current_post is not None:
        discussions[current_post] = current_responses
    
    return discussions

def create_sliding_windows(post, responses, tokenizer=None, max_tokens=SEQ_LENGTH, max_window_size=MAX_WINDOW_SIZE):
    """為討論串創建滑動窗口訓練樣本"""
    training_samples = []
    
    # 創建文章摘要（取前T5_SUMMARY_LENGTH個字符）
    post_summary = post[:T5_SUMMARY_LENGTH] if len(post) > T5_SUMMARY_LENGTH else post
    
    # 為每個回應創建滑動窗口
    for target_idx in range(1, len(responses)):
        # 確定窗口大小
        window_size = min(target_idx, max_window_size)
        
        # 檢查token數量限制
        if tokenizer:
            window_size = adjust_window_size_by_tokens(
                post_summary, responses, target_idx, window_size, 
                tokenizer, max_tokens
            )
        
        # 獲取窗口內的回應
        window_start = target_idx - window_size
        window_responses = responses[window_start:target_idx]
        
        # 創建訓練樣本
        input_text = format_window_input(post_summary, window_responses)
        target_text = responses[target_idx]
        
        training_samples.append({
            'input': input_text,
            'target': target_text,
            'post': post,
            'window_indices': list(range(window_start, target_idx)),
            'target_index': target_idx
        })
    
    return training_samples

def adjust_window_size_by_tokens(post_summary, responses, target_idx, initial_window_size, 
                               tokenizer, max_tokens):
    """根據token數量動態調整窗口大小"""
    summary_tokens = len(tokenizer.encode(post_summary))
    reserved_tokens = 100  # 預留給特殊標記和其他元素
    
    available_tokens = max_tokens - summary_tokens - reserved_tokens
    window_size = 0
    
    # 從最近的回應開始計算
    for i in range(target_idx - 1, max(0, target_idx - initial_window_size) - 1, -1):
        response_tokens = len(tokenizer.encode(responses[i]))
        if available_tokens >= response_tokens:
            available_tokens -= response_tokens
            window_size += 1
        else:
            break
    
    return window_size

def format_window_input(post_summary, window_responses, task_prefix=T5_TASK_PREFIX):
    """格式化窗口輸入為T5可處理的格式"""
    input_text = f"{task_prefix}[文章摘要] {post_summary}\n"
    
    for i, response in enumerate(window_responses):
        position = len(window_responses) - i  # 1表示最近的留言
        input_text += f"[留言_{position}樓] {response}\n"
    
    return input_text

def process_discussion_data(dataframe, tokenizer, max_tokens=SEQ_LENGTH, max_window_size=MAX_WINDOW_SIZE):
    """處理完整討論資料集"""
    # 轉換資料結構
    discussions = transform_dataset(dataframe)
    
    all_training_samples = []
    
    # 為每個討論創建訓練樣本
    for post, responses in discussions.items():
        if len(responses) >= 2:  # 至少需要2個回應才能形成訓練樣本
            samples = create_sliding_windows(
                post, responses, tokenizer, max_tokens, max_window_size
            )
            all_training_samples.extend(samples)
    
    return all_training_samples

def prepare_sliding_window_data(file_path, tokenizer=None, max_tokens=SEQ_LENGTH, max_window_size=MAX_WINDOW_SIZE):
    """準備滑動窗口訓練資料 - 添加留言數量檢查"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # 讀取資料
    if file_path.endswith('.pkl'):
        df = pd.read_pickle(PATH + file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(PATH + file_path)
    else:
        raise ValueError("不支持的文件格式")
    
    # 統計留言數量
    discussion_counts = {}
    current_post = None
    current_count = 0
    
    for idx, row in df.iterrows():
        post = row['post']
        if current_post is None or post != current_post:
            if current_post is not None:
                discussion_counts[current_post] = current_count
            current_post = post
            current_count = 1
        else:
            current_count += 1
    
    # 保存最後一個討論串的留言數量
    if current_post is not None:
        discussion_counts[current_post] = current_count
    
    # 打印討論串留言數量統計
    log = openLog('discussion_stats.txt')
    log.write("討論串留言數量統計:\n")
    for post, count in discussion_counts.items():
        log.write(f"討論串: {post[:50]}... - 留言數量: {count}\n")
    log.close()
    
    # 處理討論資料
    training_samples = process_discussion_data(df, tokenizer, max_tokens, max_window_size)
    
    # 為每個訓練樣本添加留言數量信息
    for sample in training_samples:
        post = sample['post']
        sample['message_count'] = discussion_counts.get(post, 0)
    
    # 將訓練樣本轉換為模型輸入格式
    inputs = tokenizer([sample['input'] for sample in training_samples], 
                      padding="max_length", 
                      truncation=True,
                      max_length=max_tokens,
                      return_tensors="pt")
    
    targets = tokenizer([sample['target'] for sample in training_samples],
                       padding="max_length",
                       truncation=True, 
                       max_length=max_tokens,
                       return_tensors="pt")
    
    return {
        'input_ids': inputs.input_ids.to(DEVICE),
        'attention_mask': inputs.attention_mask.to(DEVICE),
        'labels': targets.input_ids.to(DEVICE),
        'samples': training_samples,  # 保留原始樣本信息
        'discussion_counts': discussion_counts  # 添加討論串留言數量信息
    }

def prepare_dialogue_pairs(token_ids, tokenizer):
    """將訓練數據轉換為對話對 (上下文, 回應)"""
    dialogue_pairs = []
    
    # 解碼token_ids為文本
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    # 從文本中創建對話對
    # 方法1: 相鄰文本作為對話對
    for i in range(0, len(texts)-1, 2):
        if i+1 < len(texts):
            context = texts[i].strip()
            response = texts[i+1].strip()
            
            # 確保上下文和回應都不為空
            if context and response:
                dialogue_pairs.append((context, response))
    
    # 如果對話對太少，可以使用滑動窗口創建更多對話對
    if len(dialogue_pairs) < len(texts) // 4:
        for i in range(len(texts)-1):
            context = texts[i].strip()
            response = texts[i+1].strip()
            
            if context and response:
                dialogue_pairs.append((context, response))
    
    # 確保有足夠的對話對進行訓練
    if not dialogue_pairs:
        # 如果無法構建真實對話對，則創建虛構對話
        for text in texts:
            if len(text.split()) > 6:  # 確保文本足夠長
                # 將文本分為前半部分和後半部分
                words = text.split()
                mid = len(words) // 2
                context = " ".join(words[:mid])
                response = " ".join(words[mid:])
                
                if context and response:
                    dialogue_pairs.append((context, response))
    
    return dialogue_pairs

def prepare_chinese_dialogue_pairs(token_ids, tokenizer):
    """針對中文文本構建對話對 - 現在使用滑動窗口策略"""
    dialogue_pairs = []
    
    # 解碼token_ids為文本
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    # 從文本中創建類似滑動窗口的對話對
    for i in range(2, len(texts)):
        # 使用前兩個文本作為上下文，第三個作為回應
        if i >= 2:
            # 模擬文章摘要 + 留言的格式
            context = f"[文章摘要] {texts[i-2][:T5_SUMMARY_LENGTH]}\n[留言_1樓] {texts[i-1]}"
            response = texts[i]
            
            # 移除任務前綴
            if T5_TASK_PREFIX in context:
                context = context.replace(T5_TASK_PREFIX, "")
            if T5_TASK_PREFIX in response:
                response = response.replace(T5_TASK_PREFIX, "")
                
            # 移除"START"標記
            context = context.replace("START", "").strip()
            response = response.replace("START", "").strip()
            
            # 確保上下文和回應都不為空
            if context and response:
                dialogue_pairs.append((context, response))
    
    # 如果對話對太少，可以使用較小的窗口
    if len(dialogue_pairs) < len(texts) // 4:
        for i in range(1, len(texts)):
            context = f"[文章摘要] {texts[i-1][:T5_SUMMARY_LENGTH]}"
            response = texts[i]
            
            # 清理文本
            if T5_TASK_PREFIX in context:
                context = context.replace(T5_TASK_PREFIX, "")
            if T5_TASK_PREFIX in response:
                response = response.replace(T5_TASK_PREFIX, "")
            context = context.replace("START", "").strip()
            response = response.replace("START", "").strip()
            
            if context and response:
                dialogue_pairs.append((context, response))
    
    return dialogue_pairs

def gen_synthetic_data(num=GENERATE_NUM, max_length=SEQ_LENGTH, add_task_prefix=True):
    """生成合成數據用於訓練"""
    tokenizer = get_tokenizer()
    
    # 生成隨機文本 (這裡使用簡單範例，實際應用可能需要更複雜的生成方法)
    random_texts = ["合成文本 " + str(i) for i in range(num)]
    
    # 添加任務前綴
    if add_task_prefix:
        random_texts = [T5_TASK_PREFIX + text for text in random_texts]
    
    # 編碼文本
    encodings = tokenizer(
        random_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return encodings.input_ids.to(DEVICE), encodings.attention_mask.to(DEVICE)

def gen_label(num=GENERATE_NUM, target_space=2, fixed_value=None):
    """生成標籤"""
    if fixed_value is None:
        return torch.randint(low=0, high=target_space, size=(num,), device=DEVICE).long()
    else:
        assert fixed_value < target_space
        return torch.randint(low=fixed_value, high=fixed_value+1, size=(num,), device=DEVICE).long()

def decode_texts(token_ids, tokenizer=None, log=None):
    """解碼token IDs為文本"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    if log is not None:
        for text in texts:
            log.write(text + '\n')
            
    return texts

def post_process_chinese_text(text):
    """後處理中文生成文本"""
    # 移除所有特殊標記 - 對應BertTokenizer的標記形式
    text = text.replace("[PAD]", "")
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.replace("[UNK]", "")
    text = text.replace("[MASK]", "")
    
    # 處理extra tokens (UER T5模型特有)
    for i in range(100):
        text = text.replace(f"extra{i}", "")
    
    # 移除任務前綴
    text = text.replace(T5_TASK_PREFIX, "")
    
    # 移除START標記
    text = text.replace("START", "")
    
    # 移除討論標記
    text = text.replace("[文章摘要]", "")
    for i in range(10):  # 假設最多10樓
        text = text.replace(f"[留言_{i}樓]", "")
    
    # 移除英文指令殘餘
    text = text.replace("generate continuation:", "")
    text = text.replace("response:", "")
    
    # 整理空格
    text = ' '.join(text.split())
    
    return text.strip()

def prepare_chinese_generation_input(text):
    """為中文T5模型準備生成輸入"""
    # 確保使用中文前綴
    if not text.startswith(T5_TASK_PREFIX):
        text = T5_TASK_PREFIX + text
    return text

# 兼容原始代碼的函數 - 為了向後兼容，但內部完全使用T5處理
def read_sampleFile(file='ptt.pkl', pad_token='PAD', num=None):
    token_ids, attention_mask, _, tokenizer = process_text_file(file, num=num)
    
    # 創建簡化的詞彙表 (僅包含特殊token) - 實際使用tokenizer
    vocabulary = {
        "PAD": tokenizer.pad_token_id, 
        "START": tokenizer.convert_tokens_to_ids("▁START"),
        "END": tokenizer.eos_token_id
    }
    reverse_vocab = {v: k for k, v in vocabulary.items()}
    
    # 計算序列長度
    x_lengths = attention_mask.sum(dim=1).tolist()
    
    return token_ids, vocabulary, reverse_vocab, x_lengths

def decode(token_tbl, reverse_vocab, log=None):
    return decode_texts(token_tbl, log=log)