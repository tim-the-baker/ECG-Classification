import help_code_demo
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sig_key = "IEGM_seg"
lab_key = "label"

def get_dataloader():
    path_data = "./tinyml_contest_data_training/"
    path_indices = "./data_indices"
    size = 1250
    batch_size = 32

    trainset = help_code_demo.IEGM_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=size,
                        transform=transforms.Compose([help_code_demo.ToTensor()]))

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


if __name__ == '__main__':
    dataloader = get_dataloader()
    for data in dataloader:
        sig = data[sig_key][0,0,:,0]
        plt.plot(sig)
        plt.title(data[lab_key])
        plt.show(block=True)
        break

