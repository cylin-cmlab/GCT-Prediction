class MFGM(nn.Module):
    def __init__(self, model_type, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None,kernel_set=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(MFGM, self).__init__()

        self.model_type = model_type

        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.layers = layers
        self.seq_length = seq_length

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        #----------------------#
        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        #----------------------#


        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        #----------------------#
        self.gconv1_1 = nn.ModuleList()
        self.gconv1_2 = nn.ModuleList()

        self.gconv2_1 = nn.ModuleList()
        self.gconv2_2 = nn.ModuleList()
        
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()

        self.fusion = nn.ModuleList()
        #----------------------#
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        
        self.start_conv3 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv4 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        
        dilation_factor = 1
        n_heads = 8
        kern = 6
        self.fusion = (Fusion(kern, dilation_factor, n_heads, 21, [24,16,8], [24,32], dropout))

        # Paepr eq 11: R=1+(c-1)(q^m -1)/(q -1).
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1
        
        print("# Model Type", self.model_type)
        print("# receptive_field", self.receptive_field)
        i=0
        if dilation_exponential>1:
            rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            rf_size_i = i*layers*(kernel_size-1)+1
        new_dilation = 1

        self.receptive_field = 13
        temporal_len = self.receptive_field
        for j in range(1,layers+1):
           
            if dilation_exponential > 1:
                rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
            else:
                rf_size_j = rf_size_i+j*(kernel_size-1)

            kern = 5
            dilation_factor = 1
            n_heads = 8
            #num_nodes = temporal_len
            print('temporal_len', temporal_len)
            self.filter_convs.append(MutiChannel_GAT(kern, dilation_factor, n_heads, num_nodes, [24,16,8], [32,32], dropout))
            self.gate_convs.append(MutiChannel_GAT(kern, dilation_factor, n_heads, num_nodes, [24,16,8], [32,32], dropout))

             
            temporal_len = temporal_len-(kern-1)

            '''
            # skip_convs #
            (0): Conv2d(32, 64, kernel_size=(1, 13), stride=(1, 1))
            (1): Conv2d(32, 64, kernel_size=(1, 7), stride=(1, 1))
            (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            '''
            if self.seq_length>self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, temporal_len)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, temporal_len)))
            dilation_factor = 1
            n_heads = 8
            
            self.gconv1.append(S_MutiChannel_GAT(kern, dilation_factor, n_heads, temporal_len, [24,16,8], [16,24,32], dropout))
            self.gconv2.append(S_MutiChannel_GAT(kern, dilation_factor, n_heads, temporal_len, [24,16,8], [16,24,32], dropout))
            
             

            
            #####   Normalization   ##### START
            if self.seq_length>self.receptive_field:
                print('1', self.seq_length - rf_size_j + 1)
                self.norm.append(LayerNorm((residual_channels, num_nodes, temporal_len),elementwise_affine=layer_norm_affline))
                  
            else:
                print('2', self.receptive_field - rf_size_j + 1)
                self.norm.append(LayerNorm((residual_channels, num_nodes, temporal_len),elementwise_affine=layer_norm_affline))
                     
            #####   Normalization   ##### END

            
            

            new_dilation *= dilation_exponential
    

        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        
        #####   SKIP layer   ##### START
        '''
        (skip0): Conv2d(2, 64, kernel_size=(1, 19), stride=(1, 1))
        (skipE): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        '''
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        #####   SKIP layer   ##### END

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, input_1, input_2, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'


        # Step0: 檢查receptive_field, 不足則padding0
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
            input_1 = nn.functional.pad(input_1,(self.receptive_field-self.seq_length,0,0,0))
            input_2 = nn.functional.pad(input_2,(self.receptive_field-self.seq_length,0,0,0))

        # Step1: turn([64, 2, 207, 19]) to ([64, 32, 207, 19])
        x = self.start_conv(input)  

        diff_1 = torch.cat([(input[:,[0]]-input_1[:,[0]]),input[:,[1]]], dim=1)
        diff_2 = torch.cat([(input[:,[0]]-input_2[:,[0]]),input[:,[1]]], dim=1)

        diff_1 = self.start_conv1(diff_1)  
        diff_2 = self.start_conv2(diff_2) 

        x1 = self.start_conv3(input_1)  
        x2 = self.start_conv4(input_2) 

        x = self.fusion(x,diff_1,diff_2,x1,x2)

        # Step1-1: original input skip =>(skip0)
        # (skip0): Conv2d(2, 64, kernel_size=(1, 19), stride=(1, 1))
        # ([64, 32, 207, 19]) -> ([64, 64, 207, 1])
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        

        # Layers : 3層 : 19->13->7->1 (取決於TCN取的維度)
        for i in range(self.layers):
            
            # Step2: Temporal Model --START #
            # 為上一層輸出, ex:  [64, 32, 207, 19] -> [64, 32, 207, 13] -> [64, 32, 207, 7]-> [64, 32, 207, 1]
            residual = x    
            

            x = x.permute(0,1,3,2)
            # Tanh
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)

            # Sigmoid
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)

            # Fusion
            x = filter * gate

            #-----------------#


            x = F.dropout(x, self.dropout, training=self.training)

            s = x
            #s = self.fusion[i](x,x1,x2)
            s = self.skip_convs[i](s)    

            skip = s + skip
            
            # Step3: Skip after TCN --END #

            x = torch.cat([self.gconv1[i](x[:,:16], self.predefined_A[0], self.predefined_A[1]), self.gconv2[i](x[:,16:32], self.predefined_A[0], self.predefined_A[1])], dim=1 )
            
            x = x + residual[:, :, :, -x.size(3):]

            
            if idx is None:
                x = self.norm[i](x,self.idx)

            else:
                x = self.norm[i](x,idx)
 
            # Step4: GCN --END #

        #(skipE): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        skip = self.skipE(x) + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x