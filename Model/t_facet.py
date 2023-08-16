import numpy as np

def create_matrix(n, k):
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(k):
            if i-j >= 0:
                mat[i][i-j] = 1
    return mat


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_heads, num_nodes, dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()

        print('BatchMultiHeadGraphAttention', n_heads, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        #self.w = nn.Parameter(torch.Tensor(self.n_head, num_nodes, num_nodes))
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        #nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):

        bs, ch, n, dim = h.size()
        #h_prime = torch.matmul(h, self.w)
        h_prime = h
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)

        return output, attn
        
# MutiChannel_GAT(kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout)
class MutiChannel_GAT(nn.Module):
    def __init__(self, kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout):
        super(MutiChannel_GAT, self).__init__()
        
        print('MutiChannel_GAT', n_heads, num_nodes, dropout)

        self.gat_layer1 = BatchMultiHeadGraphAttention(
            n_heads*2, num_nodes, dropout
        )
        
        self.gat_layer4 = BatchMultiHeadGraphAttention(
            n_heads*2, num_nodes, dropout
        )
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.mlp1 = (nn.Conv2d(32,16,(2, 1),dilation=(1,1))) 
        self.mlp4 = (nn.Conv2d(32,16,(5, 1),dilation=(1,1))) 
 
    def forward(self,x_input):

        x_input_cpy = x_input
        
        x_input1 = self.mlp1(x_input)
        x_input4 = self.mlp4(x_input)

        #-------------GAT-------------#
        x_input1, attn = self.gat_layer1(x_input1)
        x_input4, attn = self.gat_layer4(x_input4)
        #-------------GAT-------------#

        x_input = torch.cat([x_input1[:,:,-x_input4.size(2):],x_input4[:,:,-x_input4.size(2):]], dim=1)
        
        x_input = ((x_input_cpy)[:,:,-x_input.size(2):] + (x_input)).permute(0,1,3,2)

        return x_input
