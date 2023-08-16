

def main(runid):
   

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = MFGM(args.model_type, args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                   device, predefined_A=adj_mx, kernel_set=args.kernel_set, dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print(model)
    print(args)

    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])       # model參數量!
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, data['scaler'], device, args.cl)
    
    print("start training...",flush=True)	
    his_loss =[]	
    val_time = []	
    train_time = []	
    minl = 1e5	
    start_epoch=0	
    SAVE_PATH = ""	
    train_loss_epoch = []  # 紀錄train在epoch收斂	
    valid_loss_epoch = []  # 紀錄valid在epoch收斂	
    """
    ########------------------- 讀取檔案要處理的code -----------------------#############	
    ### loading model ###	
    SAVE_PATH = args.save + "exp"+str(args.expid)+"_0.pth"	
    	
    checkpoint = torch.load(SAVE_PATH)	
    engine.model.load_state_dict(checkpoint['model_state_dict'])	
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])	
    engine.task_level = checkpoint['task_level']	
    start_epoch = checkpoint['epoch']	
    #minl = checkpoint['loss']	
    train_loss_epoch = checkpoint['train_loss']	
    valid_loss_epoch = checkpoint['valid_loss']	
    ### 測試讀取出的model ###	
    valid_loss = []	
    valid_mape = []	
    valid_rmse = []	
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):	
        testx = torch.Tensor(x).to(device)	
        testx = testx.transpose(1, 3)	
        testy = torch.Tensor(y).to(device)	
        testy = testy.transpose(1, 3)	
        metrics = engine.eval(testx, testy[:,0,:,:])	
        valid_loss.append(metrics[0])	
        valid_mape.append(metrics[1])	
        valid_rmse.append(metrics[2])	
    	
    mvalid_loss = np.mean(valid_loss)	
    mvalid_mape = np.mean(valid_mape)	
    mvalid_rmse = np.mean(valid_rmse)	
    print("### 2-The valid loss on loding model is", str(round(mvalid_mape,4)))	
    minl= mvalid_mape	
    print("### minl:",minl, "checkpoint['loss']:",checkpoint['loss'])	
    ### 測試讀取出的model ###	
    ########------------------- 讀取檔案要處理的code -----------------------#############	
    """
    '''
    #####
    SAVE_PATH = args.save + "exp202111182244_0.pth"	
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.task_level = checkpoint['task_level']
    start_epoch = checkpoint['epoch']
    train_loss_epoch = checkpoint['train_loss']
    valid_loss_epoch = checkpoint['valid_loss']
    #####
    '''
    for i in range(start_epoch,start_epoch+args.epochs+1):
    
        train_loss = []
        train_mape = []
        train_rmse = []
        train_smape = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()  # 為了檢視資料先拿掉
        for iter, (x, y, x_1, x_2) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            trainx_1 = torch.Tensor(x_1).to(device)
            trainx_1= trainx_1.transpose(1, 3)

            trainx_2 = torch.Tensor(x_2).to(device)
            trainx_2= trainx_2.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:,0,:,:], trainx_1, trainx_2)
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_smape.append(metrics[3])

            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_smape = []

        s1 = time.time()
        for iter, (x, y, x_1, x_2) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            testx_1 = torch.Tensor(x_1).to(device)
            testx_1 = testx_1.transpose(1, 3)
            testx_2 = torch.Tensor(x_2).to(device)
            testx_2 = testx_2.transpose(1, 3)

            metrics = engine.eval(testx, testy[:,0,:,:], testx_1, testx_2)

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_smape = np.mean(train_smape)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_smape = np.mean(valid_smape)
        #his_loss.append(mvalid_loss)
        his_loss.append(mvalid_smape)

        #writer.add_scalar("train_loss", mtrain_loss, i)
        #writer.add_scalar("valid_loss", mvalid_loss, i)

        writer.add_scalar("train_loss", mvalid_loss, i)
        writer.add_scalar("valid_loss", mvalid_loss, i)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        # 紀錄每個epoch的loss
        train_loss_epoch.append(mtrain_loss)
        valid_loss_epoch.append(mvalid_loss)
        
        '''
        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            minl = mvalid_loss
        '''
        if mvalid_loss<minl:
            target_best_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            print("### Update Best Model:",target_best_model, 'Loss:', mvalid_mape, " ###")
            #torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            torch.save({
              'epoch': i,
              'task_level': engine.task_level,
              'model_state_dict': engine.model.state_dict(),
              'optimizer_state_dict': engine.optimizer.state_dict(),
              'loss': mvalid_mape,
              'train_loss': train_loss_epoch,
              'valid_loss': valid_loss_epoch
            }, SAVE_PATH)
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)
    
    writer.close()
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    #target_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
    SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
    print("### loading model is:",SAVE_PATH ,'###')
    #engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"))
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    print("### Loading Model finished ###")
    print("### The valid loss on loding model is", str(round(loss,4)))
    
    
    #----------------- 此內不能刪掉，不然會crash -----------------------#
    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    print('#realy', realy.shape)
    
    for iter, (x, y, x_1, x_2) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)

        testx_1 = torch.Tensor(x_1).to(device)
        testx_1 = testx_1.transpose(1, 3)
        testx_2 = torch.Tensor(x_2).to(device)
        testx_2 = testx_2.transpose(1, 3)

        with torch.no_grad():
            preds = engine.model(testx,testx_1,testx_2)
            preds = preds.transpose(1,3)  # 64,1,6,12

        outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]  # 5240,6,12
    print('# cat valid preds', yhat.shape)
    

    #pred = scaler.inverse_transform(yhat)
    
    '''
    pred = yhat      # 5240,6,12])
    for i in range(args.num_nodes):
      pred[:,i,:] = scaler_list[i].inverse_transform(pred[:,i,:])
    '''
    pred = scaler.inverse_transform(yhat)
    
    vmae, vmape, vrmse,vsmape = metric(pred,realy)
    print("valid mape",vmape)
    #----------------- 此內不能刪掉，不然會crash -----------------------#


    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, x_1, x_2) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)

        testx_1 = torch.Tensor(x_1).to(device)
        testx_1 = testx_1.transpose(1, 3)
        testx_2 = torch.Tensor(x_2).to(device)
        testx_2 = testx_2.transpose(1, 3)

        with torch.no_grad():
            preds = engine.model(testx, testx_1, testx_2)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  #10478, 6, 12
    print('# cat test preds', yhat.shape)
    
    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.seq_out_len):
        #pred = scaler.inverse_transform(yhat[:, :, i])
        '''
        pred = yhat             # 10478, 6, 12
        for j in range(args.num_nodes):
          pred[:,j,i] = scaler_list[j].inverse_transform(pred[:,j,i])

        real = realy[:, :, i]
        real = realy[:, :, i]
        metrics = metric(pred[:,:,i], real)
        #print("#Predict:",i,", test maps", metrics[1])
        '''
        pred = scaler.inverse_transform(yhat[:, :, i])
        #pred = yhat             # 10478, 6, 12
        #for j in range(args.num_nodes):
        #  pred[:,j,i] = scaler_list[j].inverse_transform(pred[:,j,i])

        real = realy[:, :, i]
        #real = realy[:, :, i]
        #metrics = metric(pred[:,:,i], real)
        #print("#Predict:",i,", test maps", metrics[1])
        metrics = metric(pred, real)
        
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        smape.append(metrics[3])
        
    log = '{:.2f}	{:.2f}	{:.4f}	{:.4f}	'
    print( "##### exp" + str(args.expid) + "_" + str(runid)+'	', 
          log.format(mae[0], rmse[0], smape[0], mape[0]),
          log.format(mae[2], rmse[2], smape[2], mape[2]),
          log.format(mae[5], rmse[5], smape[5], mape[5]),
          log.format(mae[11], rmse[11], smape[11], mape[11]),
         )
    
    ### Drawing Loss Diagram ###
    fig = plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(checkpoint['train_loss'], label="train loss")
    plt.plot(checkpoint['valid_loss'], label="valid loss")
    plt.legend(loc="upper right")
    plt.title('#Loss of Training', fontsize=20)
    plt.ylabel("MAPE", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.show()

    return vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    vsmape = []
    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.runs):
        vm1, vm2, vm3,vm4, m1, m2, m3, m4 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        vsmape.append(vm4)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        smape.append(m4)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)
    smape = np.array(smape)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)
    asmape = np.mean(smape,0)

    smae = np.std(mae,0)
    s_mape = np.std(mape,0)
    srmse = np.std(rmse,0)
    s_smape = np.std(smape,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], s_mape[i]))
