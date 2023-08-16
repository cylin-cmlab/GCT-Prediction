

# this efficient implementation comes from https://github.com/xptree/DeepInf/
class F_BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_heads, num_nodes, dropout, bias=True):
        super(F_BatchMultiHeadGraphAttention, self).__init__()

        print('F_BatchMultiHeadGraphAttention', n_heads, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        #self.w = nn.Parameter(torch.Tensor(self.n_head, num_nodes, num_nodes))
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))

        self.mlp_convs = (nn.Conv2d(n_heads, n_heads, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, ch, n, dim = h.size()
        h_prime = (h)
        attn_src = torch.matmul(torch.tanh(h), self.a_src)
        attn_dst = torch.matmul(torch.tanh(h), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        return output, attn
        

class Fusion(nn.Module):
    def __init__(self, kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout):
        super(Fusion, self).__init__()

        self.gat_layer = F_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )
        self.gat_layer2 = F_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )
        
        self.mlp1 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        self.mlp2 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        self.mlp3 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        self.mlp4 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        self.mlp5 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        self.mlp6 = (nn.Conv2d(32,8,(1, 1),dilation=(1,1))) 
        

        self.mlp7 = (nn.Conv2d(16,32,(1, 1),dilation=(1,1))) 
    def forward(self,x_input, diff1, diff2, x1, x2):

        bs,c,n,t = x_input.shape
        x_input_cpy = x_input

        x_input1 = self.mlp1(x_input)
        diff1 = self.mlp2(diff1)
        diff2 = self.mlp3(diff2)
        
        x_input2 = self.mlp4(x_input)
        x1 = self.mlp5(x1)
        x2 = self.mlp6(x2)
      
        bs,c,n,t = x1.shape
        x_input_new = []
        x1_all = []
        x2_all = []
        for i in range(t):
          x_t = (torch.cat([x_input1[:,:,:,[i]], diff1[:,:,:,[i]], diff2[:,:,:,[i]]], dim=3)).permute(0,1,3,2)
          x_all, attn = self.gat_layer( x_t )
          
          
          x_t = (torch.cat([x_input2[:,:,:,[i]], x1[:,:,:,[i]], x2[:,:,:,[i]]], dim=3)).permute(0,1,3,2)
          x_all2, attn = self.gat_layer2( x_t )

          x1_all.append(x_all[:,:,[0]].permute(0,1,3,2))
          x2_all.append(x_all2[:,:,[0]].permute(0,1,3,2))
          
        x1 = torch.cat(x1_all,dim=3)
        x2 = torch.cat(x2_all,dim=3)

        x_out = self.mlp7( torch.cat([x1,x2],dim=1) )+x_input_cpy
        

        return x_out