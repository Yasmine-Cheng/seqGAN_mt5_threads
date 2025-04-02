# validate_t5_generation.py
# -*- coding: utf-8 -*-
"""
驗證預先生成並保存的T5 SeqGAN模型文本質量
"""
import sys
import os
import pandas as pd
import random
import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

from config import PATH, openLog

class ChineseTokenizer:
    def tokenize(self, text):
        return list(jieba.cut(text))

def prepare_test_data(csv_file, max_samples=30):
    """從CSV準備測試數據集"""
    print(f"讀取測試數據: {csv_file}")
    df = pd.read_csv(PATH + csv_file)
    
    # 確保包含 post 和 response 列
    if not all(col in df.columns for col in ['post', 'response']):
        print("錯誤: CSV缺少必要的post和response欄位")
        return None, None
    
    # 收集貼文和回應
    posts = []
    responses = []
    
    # 按照相同的post分組
    grouped = df.groupby('post')
    
    # 限制樣本數量
    selected_posts = list(grouped.groups.keys())
    if max_samples < len(selected_posts):
        selected_posts = random.sample(selected_posts, max_samples)
    
    for post in selected_posts:
        group = grouped.get_group(post)
        posts.append(post)
        responses.append(group['response'].tolist())
    
    print(f"準備好 {len(posts)} 個測試樣本")
    return posts, responses

def load_generated_text(gen_file):
    """加載生成的文本文件"""
    print(f"讀取生成文本: {gen_file}")
    try:
        with open(PATH + gen_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 清洗文本行（去除空行和特殊字符）
        generated_texts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('生成最終'):
                # 排除開頭的日誌信息和空行
                generated_texts.append(line)
        
        print(f"成功讀取 {len(generated_texts)} 條生成文本")
        return generated_texts
    except Exception as e:
        print(f"讀取生成文本失敗: {str(e)}")
        return []

def calculate_bleu(references, candidates):
    """計算BLEU分數"""
    # 分詞
    tokenized_refs = [[list(jieba.cut(ref)) for ref in refs] for refs in references]
    tokenized_cands = [list(jieba.cut(cand)) for cand in candidates]
    
    # 計算BLEU-1, BLEU-2, BLEU-3, BLEU-4分數
    smoothie = SmoothingFunction().method1
    bleu_scores = {}
    for i in range(1, 5):
        weights = tuple([1.0/i] * i + [0.0] * (4-i))
        bleu_scores[f'bleu-{i}'] = corpus_bleu(
            tokenized_refs, 
            tokenized_cands, 
            weights=weights,
            smoothing_function=smoothie
        )
    
    return bleu_scores

def calculate_rouge(references, candidates):
    """計算ROUGE分數"""
    # 創建評分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=ChineseTokenizer())
    
    all_scores = []
    for i, (cand, refs) in enumerate(zip(candidates, references)):
        # 對每個候選文本，計算與所有參考文本的ROUGE分數，取最高
        best_score = None
        
        for ref in refs:
            # 計算ROUGE分數
            score = scorer.score(ref, cand)
            
            if best_score is None:
                best_score = score
            else:
                # 更新最高分
                for metric in best_score:
                    if score[metric].fmeasure > best_score[metric].fmeasure:
                        best_score[metric] = score[metric]
        
        all_scores.append(best_score)
    
    # 計算平均分數
    avg_scores = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        avg_scores[metric] = {
            'precision': sum(s[metric].precision for s in all_scores) / len(all_scores),
            'recall': sum(s[metric].recall for s in all_scores) / len(all_scores),
            'fmeasure': sum(s[metric].fmeasure for s in all_scores) / len(all_scores)
        }
    
    return avg_scores

def calculate_bertscore(references, candidates, language="zh"):
    """計算BERTScore"""
    print("計算BERTScore...")
    # 展平參考文本列表，取每組參考文本的第一個
    flat_refs = [refs[0] for refs in references]
    
    try:
        # 計算BERTScore
        P, R, F1 = score(candidates, flat_refs, lang=language, verbose=True)
        
        # 轉換為浮點數列表
        results = {
            'precision': [p.item() for p in P],
            'recall': [r.item() for r in R],
            'f1': [f.item() for f in F1]
        }
        
        # 計算平均值
        avg_results = {
            'precision': sum(results['precision']) / len(results['precision']),
            'recall': sum(results['recall']) / len(results['recall']),
            'f1': sum(results['f1']) / len(results['f1'])
        }
        
        return avg_results
    except Exception as e:
        print(f"計算BERTScore時發生錯誤: {str(e)}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

def visualize_results(bleu_scores, rouge_scores, bertscore, output_file):
    """視覺化評估結果"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # BLEU 得分
    bleu_values = list(bleu_scores.values())
    bleu_labels = list(bleu_scores.keys())
    axs[0, 0].bar(bleu_labels, bleu_values)
    axs[0, 0].set_title('BLEU Scores')
    axs[0, 0].set_ylim(0, 1)
    
    # ROUGE F1 得分
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    rouge_values = [rouge_scores[m]['fmeasure'] for m in rouge_metrics]
    axs[0, 1].bar(rouge_metrics, rouge_values)
    axs[0, 1].set_title('ROUGE F1 Scores')
    axs[0, 1].set_ylim(0, 1)
    
    # BERTScore
    bertscore_metrics = ['precision', 'recall', 'f1']
    bertscore_values = [bertscore[m] for m in bertscore_metrics]
    axs[1, 0].bar(bertscore_metrics, bertscore_values)
    axs[1, 0].set_title('BERTScore')
    axs[1, 0].set_ylim(0, 1)
    
    # 保留一個子圖為空或用於其他用途
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"視覺化結果已保存至: {output_file}")
    
    return fig

def evaluate_diversity(texts):
    """評估文本多樣性"""
    # 計算平均長度
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    
    # 分詞
    tokenized_texts = [list(jieba.cut(text)) for text in texts]
    tokens_per_text = [len(tokens) for tokens in tokenized_texts]
    avg_tokens = sum(tokens_per_text) / len(tokens_per_text)
    
    # 詞彙多樣性 - Type-Token Ratio
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    unique_tokens = set(all_tokens)
    type_token_ratio = len(unique_tokens) / len(all_tokens) if all_tokens else 0
    
    # 詞頻統計
    token_counter = {}
    for tokens in tokenized_texts:
        for token in tokens:
            if token in token_counter:
                token_counter[token] += 1
            else:
                token_counter[token] = 1
    
    # 找出最常用的前10個詞
    most_common = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    
    diversity_metrics = {
        "avg_length": avg_length,
        "avg_tokens": avg_tokens,
        "vocabulary_size": len(unique_tokens),
        "type_token_ratio": type_token_ratio,
        "most_common_words": {word: count for word, count in most_common}
    }
    
    return diversity_metrics

def main():
    # 檢查參數
    if len(sys.argv) < 3:
        print("用法: python validate_t5_generation.py [生成文本文件名] [參考數據CSV] [最大樣本數]")
        return
    
    gen_file = sys.argv[1]  # 生成文本文件
    csv_file = sys.argv[2]  # 參考數據CSV
    
    max_samples = 30  # 預設評估30個樣本
    if len(sys.argv) > 3:
        try:
            max_samples = int(sys.argv[3])
        except ValueError:
            print(f"警告: 無效的樣本數，使用預設值 {max_samples}")
    
    # 檢查文件存在
    if not os.path.exists(PATH + gen_file):
        print(f"錯誤: 找不到生成文本文件: {PATH + gen_file}")
        return
    
    if not os.path.exists(PATH + csv_file):
        print(f"錯誤: 找不到CSV文件: {PATH + csv_file}")
        return
    
    # 讀取生成的文本
    generated_texts = load_generated_text(gen_file)
    if not generated_texts:
        return
    
    # 準備測試數據
    posts, responses = prepare_test_data(csv_file, max_samples)
    if not posts:
        return
    
    # 確保參考和生成文本數量匹配
    min_len = min(len(generated_texts), len(posts))
    generated_texts = generated_texts[:min_len]
    references = responses[:min_len]
    
    print(f"將對比 {min_len} 對生成文本和參考文本")
    
    # 評估結果
    print("評估生成結果...")
    
    # 計算BLEU分數
    print("計算BLEU分數...")
    bleu_scores = calculate_bleu(references, generated_texts)
    
    # 計算ROUGE分數
    print("計算ROUGE分數...")
    rouge_scores = calculate_rouge(references, generated_texts)
    
    # 計算BERTScore
    print("計算BERTScore分數...")
    bertscore = calculate_bertscore(references, generated_texts)
    
    # 評估文本多樣性
    print("評估文本多樣性...")
    diversity_metrics = evaluate_diversity(generated_texts)
    
    # 輸出評估報告
    report_file = f"{PATH}validation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("===== T5 SeqGAN 生成文本驗證報告 =====\n\n")
        
        # BLEU分數
        f.write("BLEU分數:\n")
        for k, v in bleu_scores.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        # ROUGE分數
        f.write("\nROUGE分數:\n")
        for metric, scores in rouge_scores.items():
            f.write(f"  {metric}:\n")
            for k, v in scores.items():
                f.write(f"    {k}: {v:.4f}\n")
        
        # BERTScore
        f.write("\nBERTScore:\n")
        for k, v in bertscore.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        # 多樣性指標
        f.write("\n文本多樣性指標:\n")
        f.write(f"  平均文本長度: {diversity_metrics['avg_length']:.2f} 字符\n")
        f.write(f"  平均分詞數: {diversity_metrics['avg_tokens']:.2f} 詞\n")
        f.write(f"  詞彙量: {diversity_metrics['vocabulary_size']} 個不同詞\n")
        f.write(f"  詞彙多樣性 (TTR): {diversity_metrics['type_token_ratio']:.4f}\n")
        
        f.write("\n最常見的詞彙:\n")
        for word, count in diversity_metrics['most_common_words'].items():
            f.write(f"  {word}: {count} 次\n")
    
    print(f"評估報告已保存至: {report_file}")
    
    # 視覺化結果
    viz_file = f"{PATH}validation_visual.png"
    visualize_results(bleu_scores, rouge_scores, bertscore, viz_file)
    
    # 保存生成的文本與參考文本對比
    output_file = f"{PATH}generated_vs_reference.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("生成的文本與參考文本比較:\n\n")
        for i, (post, gen_text, refs) in enumerate(zip(posts[:min_len], generated_texts, references)):
            f.write(f"樣本 {i+1}:\n")
            f.write(f"原文: {post[:100]}...\n")
            f.write(f"生成的回應: {gen_text}\n")
            f.write("參考回應:\n")
            for j, ref in enumerate(refs):
                f.write(f"  {j+1}. {ref}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"生成文本與參考文本比較已保存至: {output_file}")
    print("驗證完成!")

if __name__ == "__main__":
    main()