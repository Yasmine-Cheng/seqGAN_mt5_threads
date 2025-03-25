# -*- coding: utf-8 -*-
"""
用於生成文本的腳本 - 使用T5模型，支持討論串格式輸入
"""
import sys
import torch
import pandas as pd
from config import PATH, openLog, T5_TASK_PREFIX, T5_SUMMARY_LENGTH
from data_processing import decode_texts, get_tokenizer, post_process_chinese_text

def prepare_generation_input(text, task_prefix=T5_TASK_PREFIX):
    """為T5模型準備生成輸入"""
    # 確保使用正確的前綴
    if not text.startswith(task_prefix):
        text = task_prefix + text
    return text

def prepare_discussion_input(post, responses=None, task_prefix=T5_TASK_PREFIX):
    """準備討論串生成的輸入格式"""
    # 獲取文章摘要
    post_summary = post[:T5_SUMMARY_LENGTH] if len(post) > T5_SUMMARY_LENGTH else post
    
    # 構建輸入文本
    input_text = f"[文章摘要] {post_summary}"
    
    # 添加回應（如果有）
    if responses and len(responses) > 0:
        for i, response in enumerate(responses):
            input_text += f"\n[留言_{i+1}樓] {response}"
    
    # 添加任務前綴
    return prepare_generation_input(input_text, task_prefix)

def main(batch_size=1, input_file=None):
    # 嘗試加載已儲存的模型
    try:
        generator = torch.load(PATH+'t5_generator.pkl')
        tokenizer = get_tokenizer()
    except FileNotFoundError:
        print("錯誤: 找不到模型或tokenizer文件。請先訓練模型。")
        return None
    
    # 判斷是否提供了討論串資料
    if input_file and input_file.endswith('.csv'):
        # 讀取討論串資料
        try:
            df = pd.read_csv(PATH + input_file)
            if 'post' not in df.columns or 'response' not in df.columns:
                print("錯誤: CSV文件格式不符合要求，需要包含'post'和'response'列")
                return None
        except Exception as e:
            print(f"讀取討論串資料出錯: {str(e)}")
            return None
            
        # 將DataFrame轉換為討論串字典
        discussions = {}
        current_post = None
        current_responses = []
        
        for idx, row in df.iterrows():
            post = row['post']
            response = row['response']
            
            if current_post is None or post != current_post:
                if current_post is not None:
                    discussions[current_post] = current_responses
                
                current_post = post
                current_responses = [response]
            else:
                current_responses.append(response)
        
        # 保存最後一個討論
        if current_post is not None:
            discussions[current_post] = current_responses
        
        # 生成每個討論串的回應
        all_generated_texts = []
        
        for post, responses in discussions.items():
            # 每個討論串生成多個樣本
            samples_per_discussion = max(1, batch_size // len(discussions))
            
            # 準備輸入文本
            input_texts = []
            for _ in range(samples_per_discussion):
                input_text = prepare_discussion_input(post, responses)
                input_texts.append(input_text)
            
            # 生成文本
            with torch.no_grad():
                if hasattr(generator, 'module'):
                    samples = generator.module.generate(
                        input_text=input_texts,
                        batch_size=len(input_texts),
                        temperature=1.1
                    )
                else:
                    samples = generator.generate(
                        input_text=input_texts,
                        batch_size=len(input_texts),
                        temperature=1.1
                    )
            
            # 解碼生成的文本
            generated_texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
            processed_texts = [post_process_chinese_text(text) for text in generated_texts]
            
            # 保存結果
            for text in processed_texts:
                all_generated_texts.append({
                    'post': post,
                    'previous_responses': responses,
                    'generated_response': text
                })
        
        # 輸出到文件
        log = openLog('genTxt_discussions.txt')
        for item in all_generated_texts:
            log.write("----- 原文 -----\n")
            log.write(item['post'][:200] + "...\n\n")
            log.write("----- 前幾則留言 -----\n")
            for i, resp in enumerate(item['previous_responses']):
                log.write(f"[{i+1}樓] {resp}\n")
            log.write("\n----- 生成的回應 -----\n")
            log.write(item['generated_response'] + "\n\n")
            log.write("="*50 + "\n\n")
        log.close()
        
        return all_generated_texts
    
    else:
        # 傳統生成方式（不使用討論串）
        log = openLog('genTxt_predict_t5.txt')
        log.write("生成文本 (批次大小={}):\n".format(batch_size))
        
        with torch.no_grad():
            if hasattr(generator, 'module'):
                # 使用統一的輸入準備函數
                input_texts = [prepare_generation_input("START") for _ in range(batch_size)]
                samples = generator.module.generate(
                    input_text=input_texts, 
                    batch_size=batch_size,
                    temperature=1.1
                )
            else:
                # 使用統一的輸入準備函數
                input_texts = [prepare_generation_input("START") for _ in range(batch_size)]
                samples = generator.generate(
                    input_text=input_texts, 
                    batch_size=batch_size,
                    temperature=1.1
                )
        
        # 解碼並記錄生成的文本
        generated_texts = decode_texts(samples, tokenizer, log)
        
        # 後處理文本並記錄
        processed_texts = [post_process_chinese_text(text) for text in generated_texts]
        log.write("\n後處理的生成文本:\n")
        for text in processed_texts:
            log.write(text + '\n')
            
        log.close()
        
        return processed_texts

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
    try:
        input_file = sys.argv[2]
    except IndexError:
        input_file = None
    
    generated_texts = main(batch_size, input_file)
    if generated_texts:
        print("\n生成的文本:")
        if isinstance(generated_texts, list) and isinstance(generated_texts[0], dict):
            # 如果是討論串格式的結果
            for i, item in enumerate(generated_texts[:5]):  # 只顯示前5個結果
                print(f"\n=== 示例 {i+1} ===")
                print(f"文章: {item['post'][:50]}...")
                print(f"生成回應: {item['generated_response']}")
        else:
            # 如果是普通文本列表
            for i, text in enumerate(generated_texts[:5]):  # 只顯示前5個結果
                print(f"{i+1}. {text}")
    else:
        print("無法生成文本，請先訓練模型。")