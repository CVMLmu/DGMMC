import os
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dir_path):
        self.path = dir_path
        self.filenames = os.listdir(dir_path)
        
    def __getitem__(self, index):
        audio_path = os.path.join(self.path, self.filenames[index])
        #feature = data.load_and_transform_audio_data([audio_path], 'cpu')
        return audio_path, self.filenames[index]

    def __len__(self):
        return len(self.filenames)
    
class ImageBind(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = imagebind_model.imagebind_huge(pretrained=True)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        x = self.encoder(x)[ModalityType.AUDIO]

        return x

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_PATH = os.path.join('data', 'ESC', 'audio')

FEATURES_PATH = os.path.join('data', 'ESC', 'features')
if os.path.isdir(DATA_PATH) is False:
    os.mkdir(DATA_PATH)

model = ImageBind()
model.to(device)

batch_size = 4
mytrainset = MyDataset(DATA_PATH)
mytrainloader = DataLoader(mytrainset, batch_size=batch_size, shuffle=False, num_workers=2)

for i, d in enumerate(mytrainloader, 0):

    # get the inputs; data is a list of [inputs, labels]
    inputs, filenames = d
    #inputs = inputs.to(device)

    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(inputs, device)
    }
    
    features = model(inputs).cpu()

    for j in range(0, len(features)):

        feature_path = os.path.join(FEATURES_PATH, '{}.pt'.format(filenames[j]))
        torch.save(features[j], feature_path)
