import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--dataset', default="wikipedia", type=str)
parser.add_argument('--maxlen', default=50, type=int, help='输入模型的最大序列长度')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--hidden_units', default=50, type=int, help='一个embedding向量的长度')

parser.add_argument('--syn_seq_length', default=5, type=int, help='length of synth sequence')
parser.add_argument('--lr_syn_seq', default=0.3, type=float, help='learning rate for synthetic data')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for model')
parser.add_argument('--num_epochs', default=51, type=int)
parser.add_argument('--num_eval', default=10, type=int, help='验证时用多少个随机模型在已经训练好的数据上验证')
parser.add_argument('--epoch_num_for_eval', default=5, type=int, help='训练多少个epoch后开始验证')
parser.add_argument('--epoch_eval_train', default=1, type=int, help='验证时每一个模型训练轮数')
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--outer_loops', default=50, type=int)
parser.add_argument('--inner_loops', default=1, type=int)
parser.add_argument('--dis_metric', default='ours', type=str, help='度量梯度间差异的方法')
parser.add_argument('--model', default='SASRec', type=str)
parser.add_argument('--tau', default=5.0, type=float, help='temperature for gumbel softmax')
parser.add_argument('--co_cluster_num', default=5, type=int, help='ml-1m:avaiable: , '
    'wikipedia:avaiable:5, 10, 15, 50, 100, Beauty:avaiable: 5, 10, Steam:avaiable: 5, 10, 15, 50, Games:avaiable: 5, 10, '
    'Video:avaiable: 5, 10')
parser.add_argument('--num_per_cluster', default=160, type=int, help='syn data num of each cluster')
parser.add_argument('--cluster_prob_thre', default=0.1, type=float)

args = parser.parse_args()