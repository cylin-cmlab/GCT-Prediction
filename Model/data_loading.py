
batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data = {}

type_list = ['vehicular', 'stationary', 'pedestrian']

for i in range(len(type_list)):
  
  for category in ['train', 'val', 'test']:
      
      # Loading npz 
      cat_data = np.load(os.path.join(args.data, category+"_"+type_list[i] + '.npz'))
      print("loading:", category+"_"+type_list[i] + '.npz')

      if i == 0:
        key = ""
      else:
        key = "_"+str(i)
      data['x_' + category+key] = cat_data['x']     # (?, 12, 207, 2)
      data['y_' + category+key] = cat_data['y']     # (?, 12, 207, 2)

print(data.keys())


# 使用train的mean/std來正規化valid/test #
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

# 將欲訓練特徵改成正規化
for i in range(len(type_list)):
  if i == 0:
    key = ""
  else:
    key = "_"+str(i)

  for category in ['train', 'val', 'test']:
    data['x_' + category+key][..., 0] = scaler.transform(data['x_' + category+key][..., 0])


data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], data['x_train_1'], data['x_train_2'], batch_size)
data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], data['x_val_1'], data['x_val_2'], valid_batch_size)
data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], data['x_test_1'], data['x_test_2'], test_batch_size)
data['scaler'] = scaler

'''
adj_mx: 根據distances_la_2012.csv, 找出每個sensor與其他sensor距離並建立距離矩陣, 再進行正規化
'''
sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_data,args.adjtype)   # adjtype: default='doubletransition'

adj_mx = [torch.tensor(i).to(device) for i in adj_mx]

dataloader = data.copy()
