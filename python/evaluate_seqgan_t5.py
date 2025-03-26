# evaluate_seqgan_t5.py
# -*- coding: utf-8 -*-
"""
使用TextEvaluator評估T5 SeqGAN模型在討論串生成任務上的表現
"""
import sys
import os
import torch
import pandas as pd
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 無需GUI的後端

from config import PATH, openLog
from data_processing import get_tokenizer, post_process_chinese_text
from sequenceGAN_generate_t5 import prepare_discussion_input, load_model_safely, generate_safely

# 將TextEvaluator類保存為單獨的文件
def save_text_evaluator():
    """保存TextEvaluator到文件"""
    with open('text_evaluator.py', 'w', encoding='utf-8') as f:
        f.write('''# -*- coding: utf-8 -*-
"""
使用多種指標評估生成文本質量的工具
"""
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
import nltk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import jieba  # 導入中文分詞工具

# 下載必要的NLTK資源
nltk.download('punkt', quiet=True)

# 創建中文分詞器類
class ChineseTokenizer:
    def tokenize(self, text):
        return list(jieba.cut(text))

class TextEvaluator:
    def __init__(self, language="zh"):
        self.language = language
        self.smoothie = SmoothingFunction().method1
        
        # 針對中文文本修改 RougeScorer 初始化
        if language == "zh":
            # 使用自定義的中文分詞器類
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=ChineseTokenizer())
        else:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        # 不在初始化時加載模型
        self.tokenizer = None
        self.lm_model = None
        
        if language == "zh":
            self.lm_model_name = "ckiplab/gpt2-base-chinese"
        else:
            self.lm_model_name = "gpt2"
    
    def tokenize_text(self, texts):
        """將文本分詞"""
        if self.language == "zh":
            return [list(jieba.cut(text)) for text in texts]
        else:
            return [text.split() for text in texts]
    
    def calculate_bleu(self, references, candidates):
        """計算BLEU分數"""
        # 分詞
        if self.language == "zh":
            tokenized_refs = [[list(jieba.cut(ref)) for ref in refs] for refs in references]
            tokenized_cands = [list(jieba.cut(cand)) for cand in candidates]
        else:
            tokenized_refs = [[ref.split() for ref in refs] for refs in references]
            tokenized_cands = [cand.split() for cand in candidates]
        
        # 計算BLEU-1, BLEU-2, BLEU-3, BLEU-4分數
        bleu_scores = {}
        for i in range(1, 5):
            weights = tuple([1.0/i] * i + [0.0] * (4-i))
            bleu_scores[f'bleu-{i}'] = corpus_bleu(
                tokenized_refs, 
                tokenized_cands, 
                weights=weights,
                smoothing_function=self.smoothie
            )
        
        return bleu_scores
    
    def calculate_rouge(self, references, candidates):
        """計算ROUGE分數 - 針對中文文本優化"""
        all_scores = []
        
        for i, (cand, refs) in enumerate(zip(candidates, references)):
            # 對每個候選文本，計算與所有參考文本的ROUGE分數，取最高
            best_score = None
            
            for ref in refs:
                # 計算ROUGE分數
                score = self.rouge_scorer.score(ref, cand)
                
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
    
    def calculate_bertscore(self, references, candidates):
        """計算BERTScore"""
        # 展平參考文本列表，取每組參考文本的第一個
        flat_refs = [refs[0] for refs in references]
        
        try:
            # 計算BERTScore
            P, R, F1 = score(candidates, flat_refs, lang=self.language)
            
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
            print(f"計算BERTScore時發生錯誤: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def discriminator_score(self, discriminator, generated_texts, real_texts=None, batch_size=16):
        """使用判別器評分"""
        if discriminator is None:
            return {"error": "判別器未提供"}
            
        try:
            discriminator.eval()
            
            # 計算生成文本的分數
            gen_scores = []
            for i in range(0, len(generated_texts), batch_size):
                batch = generated_texts[i:i+batch_size]
                tokenized = self.tokenize_text(batch)
                
                with torch.no_grad():
                    scores = discriminator(tokenized)
                    gen_scores.extend(scores.tolist())
            
            # 計算真實文本的分數
            real_scores = []
            if real_texts:
                for i in range(0, len(real_texts), batch_size):
                    batch = real_texts[i:i+batch_size]
                    tokenized = self.tokenize_text(batch)
                    
                    with torch.no_grad():
                        scores = discriminator(tokenized)
                        real_scores.extend(scores.tolist())
            
            return {
                'generated_scores': gen_scores,
                'generated_avg': sum(gen_scores) / len(gen_scores) if gen_scores else 0,
                'real_scores': real_scores,
                'real_avg': sum(real_scores) / len(real_scores) if real_scores else 0,
            }
        except Exception as e:
            print(f"計算判別器分數時發生錯誤: {e}")
            return {"error": f"計算判別器分數時發生錯誤: {e}"}
    
    def evaluate_all(self, references, candidates, discriminator=None, real_texts=None):
        """評估所有指標"""
        results = {}
        
        print("計算BLEU分數...")
        results['bleu'] = self.calculate_bleu(references, candidates)
        
        print("計算ROUGE分數...")
        results['rouge'] = self.calculate_rouge(references, candidates)
        
        print("計算BERTScore...")
        results['bertscore'] = self.calculate_bertscore(references, candidates)

        # 如果提供了判別器和真實文本，計算判別器分數
        if discriminator is not None:
            print("計算判別器分數...")
            results['discriminator'] = self.discriminator_score(discriminator, candidates, real_texts)
        else:
            print("未提供判別器，跳過判別器分數計算")
        
        return results
    
    def generate_report(self, results, output_file=None):
        """生成評估報告"""
        report = "===== SeqGAN 生成文本評估報告 =====\\n\\n"
        
        # BLEU
        report += "BLEU Scores:\\n"
        for k, v in results['bleu'].items():
            report += f"  {k}: {v:.4f}\\n"
        
        # ROUGE
        report += "\\nROUGE Scores:\\n"
        for metric, scores in results['rouge'].items():
            report += f"  {metric}:\\n"
            for k, v in scores.items():
                report += f"    {k}: {v:.4f}\\n"
        
        # BERTScore
        report += "\\nBERTScore:\\n"
        for k, v in results['bertscore'].items():
            report += f"  {k}: {v:.4f}\\n"
        
        # 判別器分數
        if 'discriminator' in results:
            if 'error' in results['discriminator']:
                report += f"\\n判別器分數: {results['discriminator']['error']}\\n"
            else:
                report += f"\\n判別器分數:\\n"
                report += f"  生成文本平均分數: {results['discriminator']['generated_avg']:.4f}\\n"
                if results['discriminator']['real_scores']:
                    report += f"  真實文本平均分數: {results['discriminator']['real_avg']:.4f}\\n"
        
        # 輸出到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"報告已保存至 {output_file}")
        
        return report
    
    def visualize_results(self, results, output_file=None):
        """視覺化評估結果"""
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # BLEU 得分
            bleu_values = list(results['bleu'].values())
            bleu_labels = list(results['bleu'].keys())
            axs[0, 0].bar(bleu_labels, bleu_values)
            axs[0, 0].set_title('BLEU Scores')
            axs[0, 0].set_ylim(0, 1)
            
            # ROUGE F1 得分
            rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
            rouge_values = [results['rouge'][m]['fmeasure'] for m in rouge_metrics]
            axs[0, 1].bar(rouge_metrics, rouge_values)
            axs[0, 1].set_title('ROUGE F1 Scores')
            axs[0, 1].set_ylim(0, 1)
            
            # BERTScore
            bertscore_metrics = ['precision', 'recall', 'f1']
            bertscore_values = [results['bertscore'][m] for m in bertscore_metrics]
            axs[1, 0].bar(bertscore_metrics, bertscore_values)
            axs[1, 0].set_title('BERTScore')
            axs[1, 0].set_ylim(0, 1)
 
            # 添加判別器分數(如果有)
            if 'discriminator' in results and 'error' not in results['discriminator']:
                disc_df = pd.DataFrame({
                    'Type': ['Generated', 'Real'] if results['discriminator']['real_scores'] else ['Generated'],
                    'Score': [results['discriminator']['generated_avg']] + 
                            ([results['discriminator']['real_avg']] if results['discriminator']['real_scores'] else [])
                })
                
                # 創建第三行的圖表
                fig.set_size_inches(15, 15)
                ax_disc = fig.add_subplot(3, 2, 5)
                disc_df.plot.bar(x='Type', y='Score', ax=ax_disc)
                ax_disc.set_title('Discriminator Scores')
                ax_disc.set_ylim(0, 1)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file)
                print(f"視覺化結果已保存至 {output_file}")
                
            return fig
        except Exception as e:
            print(f"視覺化結果時發生錯誤: {e}")
            return None
''')
    print("TextEvaluator已保存至text_evaluator.py")

def prepare_test_discussions(csv_file, max_samples=100):
    """從CSV準備測試數據集"""
    print(f"讀取測試數據: {csv_file}")
    df = pd.read_csv(PATH + csv_file)
    
    # 確定回應列名稱
    if 'response' in df.columns:
        response_col = 'response'
    elif 'next_resp' in df.columns:
        response_col = 'next_resp'
    else:
        print("錯誤: CSV缺少回應欄位")
        return None
    
    # 將DataFrame轉換為討論串字典
    discussions = {}
    current_post = None
    current_responses = []
    
    for idx, row in df.iterrows():
        post = row['post']
        response = row[response_col]
        
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
    
    # 篩選掉回應數量小於2的討論串
    discussions = {post: resps for post, resps in discussions.items() if len(resps) >= 2}
    
    # 限制樣本數量
    discussion_posts = list(discussions.keys())
    if max_samples < len(discussion_posts):
        discussion_posts = random.sample(discussion_posts, max_samples)
    
    test_discussions = {post: discussions[post] for post in discussion_posts}
    print(f"準備好 {len(test_discussions)} 個測試討論串")
    return test_discussions

def generate_responses(model, test_discussions, tokenizer, max_samples=50):
    """使用模型生成回應"""
    print(f"開始生成回應...")
    
    references = []
    candidates = []
    posts = []
    prev_responses = []
    
    # 限制處理的討論串數量
    test_posts = list(test_discussions.keys())[:max_samples]
    progress_bar = tqdm(test_posts, desc="生成回應中")
    
    for post in progress_bar:
        responses = test_discussions[post]
        
        # 檢查是否有足夠的回應作為參考
        if len(responses) < 2:
            continue
        
        # 使用除最後一條外的所有留言作為輸入
        input_responses = responses[:-1]
        
        # 最後一條留言作為參考
        reference = [responses[-1]]
        
        # 準備輸入
        input_text = prepare_discussion_input(post, input_responses)
        
        # 生成回應
        try:
            generated_texts = generate_safely(model, [input_text], tokenizer, 1)
            if not generated_texts or generated_texts[0] == "[生成失敗]":
                continue
                
            candidate = generated_texts[0]
            
            # 保存結果
            references.append(reference)
            candidates.append(candidate)
            posts.append(post)
            prev_responses.append(input_responses)
        except Exception as e:
            print(f"生成時出錯: {str(e)}")
            continue
    
    return references, candidates, posts, prev_responses

def main():
    # 首先保存TextEvaluator
    save_text_evaluator()
    
    try:
        # 導入TextEvaluator
        from text_evaluator import TextEvaluator
    except ImportError:
        print("無法導入TextEvaluator，請確保text_evaluator.py已正確保存")
        return
    
    if len(sys.argv) < 2:
        print("用法: python evaluate_seqgan_t5.py [csv檔案] [最大樣本數]")
        return
    
    csv_file = sys.argv[1]
    max_samples = 30  # 預設評估30個樣本
    if len(sys.argv) > 2:
        try:
            max_samples = int(sys.argv[2])
        except ValueError:
            print(f"警告: 無效的樣本數，使用預設值 {max_samples}")
    
    # 檢查CSV檔案
    if not os.path.exists(PATH + csv_file):
        print(f"錯誤: 找不到CSV檔案: {PATH + csv_file}")
        return
    
    # 檢查模型檔案
    model_path = PATH + 't5_generator.pkl'
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型檔案: {model_path}")
        return
    
    # 準備測試數據
    test_discussions = prepare_test_discussions(csv_file, max_samples)
    if not test_discussions:
        return
    
    # 載入模型
    model = load_model_safely(model_path)
    if model is None:
        print("模型載入失敗")
        return
    
    tokenizer = get_tokenizer()
    print("模型和tokenizer載入成功")
    
    # 生成回應
    references, candidates, posts, prev_responses = generate_responses(
        model, test_discussions, tokenizer, max_samples)
    
    if not candidates:
        print("沒有成功生成任何回應")
        return
    
    print(f"成功生成 {len(candidates)} 個回應")
    
    # 初始化評估器
    print("初始化評估器...")
    evaluator = TextEvaluator(language="zh")
    
    # 評估生成結果
    print("開始評估...")
    results = evaluator.evaluate_all(references, candidates)
    
    # 生成報告
    report = evaluator.generate_report(results, output_file=f"{PATH}evaluation_report.txt")
    print("評估報告:")
    print(report)
    
    # 嘗試視覺化結果
    try:
        fig = evaluator.visualize_results(results, output_file=f"{PATH}evaluation_results.png")
        print(f"視覺化結果已保存至: {PATH}evaluation_results.png")
    except Exception as e:
        print(f"視覺化結果時出錯: {str(e)}")
    
    # 保存生成的文本和參考文本
    output_file = f"{PATH}generated_vs_reference.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("生成的文本與參考文本比較:\n\n")
        for i, (post, prev_resp, cand, refs) in enumerate(zip(posts, prev_responses, candidates, references)):
            f.write(f"樣本 {i+1}:\n")
            f.write(f"原文: {post[:100]}...\n")
            f.write(f"前述留言: {prev_resp[-1] if prev_resp else '無'}\n")
            f.write(f"生成的回應: {cand}\n")
            f.write(f"參考回應: {refs[0]}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"生成文本與參考文本比較已保存至: {output_file}")
    print("評估完成!")

if __name__ == "__main__":
    main()