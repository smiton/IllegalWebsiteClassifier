# coding: UTF-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                               # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                   # 验证集
        self.test_path = dataset + '/data/test.txt'                                 # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                               # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'       # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 3000                                             # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                                     # 类别数
        self.num_epochs = 30                                                         # epoch数
        self.batch_size = 48                                                        # mini-batch大小
        self.pad_size = 256                                                         # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                                   # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


def rand_init_hidden(batch_size):
    """
    random initialize hidden variable
    """
    return Variable(torch.randn(2 * 2, batch_size, 768, device='cuda')),\
           Variable(torch.randn(2 * 2, batch_size, 768, device='cuda'))


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.out_size = 3
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.hidden_dim = config.hidden_size
        self.lstm = nn.LSTM(config.hidden_size*self.out_size, config.hidden_size, num_layers=2, bidirectional=True, dropout=0.2,
                            batch_first=True)
        self.fc1 = nn.Linear(config.hidden_size*2*config.pad_size,  config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        self.hidden = rand_init_hidden(config.batch_size)

    def forward(self, sentence):
        context = sentence[0]  # 输入的句子
        mask = sentence[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True)
        encoder_out_last = encoder_out[-1*self.out_size:]
        embeds = torch.cat([conv for conv in encoder_out_last], 2)

        batch_size = context.size(0)
        hidden = rand_init_hidden(batch_size)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        out = self.fc1(lstm_out)
        out = self.fc2(out)
        return out
