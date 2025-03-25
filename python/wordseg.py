# -*- coding: utf-8 -*-
"""
簡化的wordseg.py，專門處理CSV格式的討論串資料
"""
import pandas as pd
from transformers import AutoTokenizer
from config import PATH, SEQ_LENGTH, T5_MODEL_NAME, DEVICE

def readDiscussionCSV(inputFilename='ptt_real.csv', outputFilename='ptt_discussions.pkl', num=None):
    """直接讀取CSV格式的討論串資料，只使用post和response欄位"""
    try:
        # 讀取CSV檔案
        data = pd.read_csv(PATH + inputFilename)
        
        # 如果指定了數量，只取前num筆資料
        if num is not None:
            data = data.head(num)
    except Exception as e:
        print(f"讀取資料失敗: {str(e)}")
        return None
    
    # 檢查必要的欄位是否存在
    required_columns = ['post', 'response']
    if not all(col in data.columns for col in required_columns):
        print(f"資料缺少必要的欄位，需要: {required_columns}")
        return None
    
    # 按照相同的post分組為討論串
    discussion_threads = []
    current_post = None
    current_messages = []
    thread_index = 0
    
    for idx, row in data.iterrows():
        post = row['post']
        response = row['response']
        
        # 如果是新的討論主題，開始新的討論串
        if post != current_post:
            if current_messages:
                discussion_threads.append({
                    'thread_id': thread_index,
                    'thread_post': current_post,
                    'messages': current_messages
                })
                thread_index += 1
            current_messages = []
            current_post = post
        
        # 直接添加response留言到討論串
        current_messages.append(response)
    
    # 添加最後一個討論串
    if current_messages:
        discussion_threads.append({
            'thread_id': thread_index,
            'thread_post': current_post,
            'messages': current_messages
        })
    
    # 將討論串資料轉換為DataFrame並保存
    discussions_df = pd.DataFrame({
        'thread_id': [t['thread_id'] for t in discussion_threads],
        'thread_post': [t['thread_post'] for t in discussion_threads],
        'messages_count': [len(t['messages']) for t in discussion_threads],
        'messages': [t['messages'] for t in discussion_threads]
    })
    
    # 保存處理後的資料
    discussions_df.to_pickle(PATH + outputFilename)
    print(f"處理完成: 找到 {len(discussion_threads)} 個討論串")
    
    return discussions_df

def prepare_t5_discussion_input(discussion_thread, tokenizer, max_length=SEQ_LENGTH):
    """為T5模型準備單個討論串的輸入 - 適用於留言列表"""
    # 構建討論串文本
    thread_texts = []
    
    # 添加主貼文
    thread_texts.append(f"POST: {discussion_thread['thread_post']}")
    
    # 添加所有留言 - 直接從留言列表中添加
    for message in discussion_thread['messages']:
        thread_texts.append(f"RESP: {message}")
    
    # 合併為單一文本
    full_text = " ".join(thread_texts)
    
    # 使用T5 tokenizer編碼
    encoding = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 確保編碼在正確的設備上
    encoding_dict = {
        'input_ids': encoding.input_ids.to(DEVICE),
        'attention_mask': encoding.attention_mask.to(DEVICE),
        'thread_text': full_text
    }
    
    return encoding_dict

def prepare_all_discussions_for_t5(discussions_df, tokenizer, max_length=SEQ_LENGTH):
    """為T5模型準備所有討論串"""
    encoded_threads = []
    
    for _, row in discussions_df.iterrows():
        encoded = prepare_t5_discussion_input(row, tokenizer, max_length)
        encoded_threads.append(encoded)
    
    return encoded_threads

def load_t5_tokenizer(model_name=T5_MODEL_NAME):
    """載入T5 tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"載入tokenizer失敗: {str(e)}")
        return None

#%%
if __name__ == '__main__':
    # 處理CSV格式的討論串資料
    discussions = readDiscussionCSV(inputFilename='ptt_real.csv', outputFilename='ptt_discussions.pkl')
    
    if discussions is not None:
        # 載入T5 tokenizer
        tokenizer = load_t5_tokenizer()
        
        if tokenizer is not None:
            # 為T5模型準備討論串資料
            encoded_threads = prepare_all_discussions_for_t5(discussions, tokenizer)
            print(f"成功準備了 {len(encoded_threads)} 個討論串的T5編碼")