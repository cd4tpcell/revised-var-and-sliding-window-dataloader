from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import datetime
import time
class MyDataset(Dataset):
    def __init__(self, dir, files, seq_lens):
        self.files = files
        self.dir = dir
        self.seq_lens = seq_lens
        self.valid_indices = self._get_valid_indices()


    def _get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.files)):
            if self._is_valid_sample(idx):
                valid_indices.append(idx)
        return valid_indices

    def _is_valid_sample(self, idx):
        # 检查样本是否有效
        time = self.files[idx][:10]
        time_data = []
        year = int(time[:4])
        month = int(time[4:6])
        day = int(time[6:8])
        hour = int(time[8:10])
        timestamps = datetime.datetime(year, month, day, hour)
        time_data.append(timestamps)
        j = 1
        min_time_gap = datetime.timedelta(hours=1)
        while j < self.seq_lens + 1 and idx + j < len(self.files):
            time1 = self.files[idx + j][:10]
            year1 = int(time1[:4])
            month1 = int(time1[4:6])
            day1 = int(time1[6:8])
            hour1 = int(time1[8:10])
            timestampsx = datetime.datetime(year1, month1, day1, hour1)
            time_gap = timestampsx - time_data[-1]
            if time_gap == min_time_gap:
                time_data.append(timestampsx)
                j += 1
            else:
                break
            if len(time_data) == self.seq_lens:
                break
        return len(time_data) == self.seq_lens

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        zancundata = []
        time_data = []
        for j in range(self.seq_lens):
            time = self.files[actual_idx+j][:10]

            year = int(time[:4])
            month = int(time[4:6])
            day = int(time[6:8])
            hour = int(time[8:10])
            timestamps = datetime.datetime(year, month, day, hour)
            time_data.append(timestamps)
            zancundata.append(np.load(self.dir + self.files[actual_idx+j]))
        # j = 1
        # min_time_gap = datetime.timedelta(hours=1)
        # while j < self.seq_lens + 1 and actual_idx + j < len(self.files):
        #     time1 = self.files[actual_idx + j][:10]
        #     year1 = int(time1[:4])
        #     month1 = int(time1[4:6])
        #     day1 = int(time1[6:8])
        #     hour1 = int(time1[8:10])
        #     timestampsx = datetime.datetime(year1, month1, day1, hour1)
        #     time_gap = timestampsx - time_data[-1]
        #     if time_gap == min_time_gap:
        #         time_data.append(timestampsx)
        #         mydatax = np.load(self.dir + self.files[actual_idx + j])
        #         zancundata.append(mydatax)
        #         j += 1
        #     else:
        #         break
        #     if len(time_data) == self.seq_lens:
        #         break
        zancundata1 = np.stack(zancundata, axis=0)
        # print(zancundata1.shape)
        # exit()
        return {
            'pre': torch.from_numpy(zancundata1).reshape(self.seq_lens, 1, 1024, 1024).float(),
            'time': str(time_data[0]) + " " + str(time_data[-1]),
        }

# # 使用预处理后的数据集
# train_dataset = MyDataset(dir, myfiles, 3)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
if __name__ == '__main__':
    dir='/Workspace_II/chenhm/wok/dataset/satetrain/'
    myfiles=sorted(os.listdir(dir))
    dataset = MyDataset(dir,myfiles,16)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)
    # ll=len(dataloader)

    # anadir='/Workspace_II/chenhm/wok/pre/testdata/'
    # anmyfiles=sorted(os.listdir(anadir))
    # andataset = MyDataset(anadir,anmyfiles,12)

    # andataloader = DataLoader(andataset, batch_size=4, shuffle=True)
    # anll=len(andataloader)
    # eff=0
    for it, batch in enumerate(dataloader):
        print(it,batch['pre'].shape,batch['time'])

    # aneff=0
    # for it, batch in enumerate(andataloader):
    #     if torch.max(batch['pre'])<=1:
    #         aneff=aneff+1
    # print(eff,aneff)
    

    # start_time = time.time()  # 记录循环开始的时间

    # for it, batch in enumerate(dataloader):
    #     print(ll,it)

    # end_time = time.time()  # 记录循环结束的时间

    # elapsed_time = end_time - start_time  # 计算耗时

    # print(elapsed_time/ll)
        
