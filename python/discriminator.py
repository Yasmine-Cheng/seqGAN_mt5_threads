# -*- coding: utf-8 -*-
"""
專為T5 token IDs設計的判別器 - 支持討論串輸入
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SEQ_LENGTH, EMB_SIZE, FILTER_SIZE, NUM_FILTER, DIS_NUM_EPOCH_PRETRAIN, DEVICE, openLog
from data_processing import gen_synthetic_data, gen_label, get_tokenizer

class Discriminator(nn.Module):
    def __init__(self, filter_sizes=None, num_filters=None, dropout_rate=0.2):
        super().__init__()
        
        # 設置參數
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]  # 常見的CNN過濾器大小
        if num_filters is None:
            num_filters = [100, 100, 100]  # 每種大小的過濾器數量
            
        self.tokenizer = get_tokenizer()
        vocab_size = self.tokenizer.vocab_size
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE)
        
        # 卷積層
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, EMB_SIZE)) 
            for f, n in zip(filter_sizes, num_filters)
        ])
        
        # 輸出層
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(sum(num_filters), 2)  # 二分類：真/假
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, attention_mask=None):
        # x的形狀: [batch_size, seq_len]
        # 嵌入層
        x = self.embedding(x)  # [batch_size, seq_len, emb_size]
        
        # 應用注意力掩碼（如果提供）
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            
        # 添加通道維度
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb_size]
        
        # 應用卷積和池化
        pooled_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))  # [batch_size, num_filters, seq_len-filter_size+1, 1]
            h = F.max_pool2d(h, (h.size(2), 1))  # [batch_size, num_filters, 1, 1]
            h = h.squeeze(-1).squeeze(-1)  # [batch_size, num_filters]
            pooled_outputs.append(h)
            
        # 組合所有池化特徵
        h_pool = torch.cat(pooled_outputs, dim=1)  # [batch_size, sum(num_filters)]
        
        # 丟棄和全連接層
        h_drop = self.dropout(h_pool)
        logits = self.fc(h_drop)
        probs = self.softmax(logits)
        
        return probs

# 新增: 專門用於討論串評估的判別器
class DiscussionDiscriminator(nn.Module):
    def __init__(self, filter_sizes=None, num_filters=None, dropout_rate=0.2):
        super().__init__()
        
        # 基本設置與標準判別器相同
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        if num_filters is None:
            num_filters = [100, 100, 100]
            
        self.tokenizer = get_tokenizer()
        vocab_size = self.tokenizer.vocab_size
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, EMB_SIZE)
        
        # 卷積層
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, EMB_SIZE)) 
            for f, n in zip(filter_sizes, num_filters)
        ])
        
        # 額外的LSTM層用於捕捉討論串序列特徵
        self.lstm = nn.LSTM(
            input_size=sum(num_filters),
            hidden_size=sum(num_filters) // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # 輸出層
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(sum(num_filters), 2)  # 二分類
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, attention_mask=None):
        # x的形狀: [batch_size, seq_len]
        batch_size = x.size(0)
        
        # 嵌入層
        x = self.embedding(x)  # [batch_size, seq_len, emb_size]
        
        # 應用注意力掩碼
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            
        # 添加通道維度
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb_size]
        
        # 應用卷積和池化
        pooled_outputs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool2d(h, (h.size(2), 1))
            h = h.squeeze(-1).squeeze(-1)
            pooled_outputs.append(h)
            
        # 組合所有池化特徵
        h_pool = torch.cat(pooled_outputs, dim=1)  # [batch_size, sum(num_filters)]
        
        # 使用LSTM進一步處理序列特徵
        h_pool = h_pool.unsqueeze(1)  # [batch_size, 1, sum(num_filters)]
        lstm_out, _ = self.lstm(h_pool)  # [batch_size, 1, sum(num_filters)]
        lstm_out = lstm_out.squeeze(1)  # [batch_size, sum(num_filters)]
        
        # 丟棄和全連接層
        h_drop = self.dropout(lstm_out)
        logits = self.fc(h_drop)
        probs = self.softmax(logits)
        
        return probs

def train_discriminator(real_data, fake_data, real_mask=None, fake_mask=None, batch_size=32, epochs=DIS_NUM_EPOCH_PRETRAIN, use_discussion_discriminator=False):
    """訓練判別器"""
    # 準備數據
    if real_mask is None:
        real_mask = torch.ones_like(real_data)
    if fake_mask is None:
        fake_mask = torch.ones_like(fake_data)
        
    # 準備標籤：1表示真實，0表示虛假
    real_labels = torch.ones(len(real_data), device=DEVICE).long()
    fake_labels = torch.zeros(len(fake_data), device=DEVICE).long()
    
    # 組合數據
    combined_data = torch.cat([real_data, fake_data])
    combined_masks = torch.cat([real_mask, fake_mask])
    combined_labels = torch.cat([real_labels, fake_labels])
    
    # 創建和訓練判別器
    if use_discussion_discriminator:
        discriminator = DiscussionDiscriminator(filter_sizes=FILTER_SIZE[:3], num_filters=NUM_FILTER[:3])
    else:
        discriminator = Discriminator(filter_sizes=FILTER_SIZE[:3], num_filters=NUM_FILTER[:3])
        
    discriminator = nn.DataParallel(discriminator)
    discriminator.to(DEVICE)
    
    # 優化器和損失函數
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 訓練循環
    log = openLog()
    log.write('    training discriminator: {}\n'.format(datetime.now()))
    
    for epoch in range(epochs):
        # 洗牌數據
        indices = torch.randperm(len(combined_data))
        shuffled_data = combined_data[indices]
        shuffled_masks = combined_masks[indices]
        shuffled_labels = combined_labels[indices]
        
        # 批次訓練
        pointer = 0
        total_loss = 0
        batch_count = 0
        
        while pointer + batch_size <= len(shuffled_data):
            # 獲取批次
            batch_data = shuffled_data[pointer:pointer+batch_size]
            batch_masks = shuffled_masks[pointer:pointer+batch_size]
            batch_labels = shuffled_labels[pointer:pointer+batch_size]
            
            # 前向傳播
            predictions = discriminator(batch_data, batch_masks)
            loss = criterion(predictions, batch_labels)
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新統計
            total_loss += loss.item()
            batch_count += 1
            pointer += batch_size
            
        # 紀錄進度
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log.write('      epoch: {} loss: {:.4f}\n'.format(epoch+1, avg_loss))
    
    log.close()
    return discriminator

# 新增: 專門用於訓練討論串判別器的函數
def train_discussion_discriminator(discussion_threads, fake_threads, batch_size=1, epochs=DIS_NUM_EPOCH_PRETRAIN):
    """專門訓練討論串判別器"""
    # 提取輸入ID和掩碼
    real_data = []
    real_masks = []
    for thread in discussion_threads:
        real_data.append(thread['input_ids'])
        real_masks.append(thread['attention_mask'])
    
    fake_data = []
    fake_masks = []
    for thread in fake_threads:
        fake_data.append(thread['input_ids'])
        fake_masks.append(thread['attention_mask'])
    
    # 合併為批次
    real_data = torch.cat(real_data, dim=0)
    real_masks = torch.cat(real_masks, dim=0)
    fake_data = torch.cat(fake_data, dim=0)
    fake_masks = torch.cat(fake_masks, dim=0)
    
    # 使用討論串專用判別器
    return train_discriminator(
        real_data=real_data,
        fake_data=fake_data,
        real_mask=real_masks,
        fake_mask=fake_masks,
        batch_size=batch_size,
        epochs=epochs,
        use_discussion_discriminator=True
    )

def sanityCheck_discriminator(batch_size=1):
    """測試判別器功能"""
    log = openLog('test.txt')
    log.write('\n\nTest discriminator.sanityCheck_discriminator: {}\n'.format(datetime.now()))
    
    # 生成測試數據
    token_ids, attention_mask = gen_synthetic_data(num=batch_size*2)
    real_data = token_ids[:batch_size]
    fake_data = token_ids[batch_size:]
    
    # 訓練判別器
    discriminator = train_discriminator(
        real_data=real_data, 
        fake_data=fake_data, 
        batch_size=batch_size
    )
    
    # 測試判別器
    with torch.no_grad():
        predictions = discriminator(fake_data)
    
    log.write('  y_pred shape: '+str(predictions.shape)+'\n')
    log.close()
    
    return discriminator, predictions

if __name__ == '__main__':
    model, y_pred = sanityCheck_discriminator(batch_size=4)