import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import SwinModel,SwinConfig
from transformers.models.swin.modeling_swin import SwinPatchMerging

# from torchvision.datasets import VOCDet"ection

from common_utils import get_train_dataset, get_transformations
from config import Config
import sys
import gc


class SwinModelModified(torch.nn.Module):
    def __init__(self, config, final_input_resolution=None, dim=None, downsample_last=False, decoder=False):
        super().__init__()
        
        
        self.final_input_resolution = final_input_resolution
        if decoder:
            self.model = SwinModel(config)
        else:
            self.model  = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            
        self.downsample = None
        if downsample_last:
            self.downsample = SwinPatchMerging(final_input_resolution, dim=dim, norm_layer=nn.LayerNorm)
            
    def forward(self, x):
        out = self.model(x, output_hidden_states=True)
        out_downsampled = None
        if self.downsample is not None:
            out_downsampled = self.downsample(out.hidden_states[-1], self.final_input_resolution)
        return out, out_downsampled
    
class SwinEncoderDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = 96
        self.layers = [96, 192, 384, 768, 1536]
        self.config = config
        
        final_encoder_resolution = [shape//(4*2**3) for shape in config.image_size]
        final_encoder_dim = int(self.embed_dim * 2**3)
        self.encoder = SwinModelModified(
            config=None, 
            final_input_resolution=final_encoder_resolution, 
            dim=final_encoder_dim,
            downsample_last=True
            )
        
        tasks = []
        for task in config.tasks.keys():
            if config.tasks[task]:
                tasks.append(task)
        self.task_to_id = {task: i for i, task in enumerate(tasks)}
        self.task_decoder = nn.ModuleList([nn.ModuleList() for _ in range(len(config.tasks))])
        self.task_upsampler = nn.ModuleList([nn.ModuleList() for _ in range(len(config.tasks))])
        self.task_head = [[] for _ in range(len(config.tasks))]
        
        for task, include_task in config.tasks.items():
            task_id = self.task_to_id[task]
            if not include_task:
                continue
            for i in range(4):
                if i==0:
                    num_channels = self.layers[-1]
                    embed_dim = self.layers[-1]
                else:
                    num_channels = self.layers[-(i)]
                    embed_dim = self.layers[-(i+1)]
                    
                decoder_config = SwinConfig(
                    patch_size=1,  
                    depths=[config.decoder_depth], 
                    num_heads=[config.decoder_n_head[i]], 
                    embed_dim=embed_dim, 
                    num_channels=num_channels,
                    mlp_ratio=4,  
                    window_size=config.decoder_window_size[i],
                    dropout=0.1,
                    out_indices=[0],
                    use_absolute_embeddings=False,
                    image_size=config.image_size[0]//(4*2**(4-i))
                )
                self.task_decoder[task_id].append(SwinModelModified(decoder_config, decoder=True))
                
                upsampler = nn.ConvTranspose2d(
                    in_channels=embed_dim,
                    out_channels=embed_dim//2,
                    kernel_size=2,
                    stride=2
                )
                self.task_upsampler[task_id].append(upsampler)
        
    def forward(self, x):
        batch_size, in_channels, _, _ = x.shape
        
        out, out_downsampled = self.encoder(x) #out_downsampled=(batch_size, h/64*w/64, 1536)
        encoder_out_image_size = [shape//(2**6) for shape in self.config.image_size]
        encoder_out_temp = out_downsampled.view(batch_size, *encoder_out_image_size, self.layers[-1]).permute(0, 3, 1, 2)
        del out_downsampled
        
        temp = {}
        for task, task_id in self.task_to_id.items():
            for i in range(4):
                if i==0:
                    encoder_out = encoder_out_temp
                    decoder_inp = encoder_out
                else:
                    encoder_out = out.hidden_states[-(i+1)] # (batch_size, h/(4*2**(4-i))*w/(4*2**(4-i)), layers[-(i+1)])
                    encoder_out_image_size = [shape//(2**(4-i+2)) for shape in self.config.image_size]
                    encoder_out = encoder_out.view(batch_size, *encoder_out_image_size, self.layers[-(i+1)]).permute(0, 3, 1, 2)
                    decoder_inp = torch.cat((decoder_out, encoder_out), dim=1)

                decoder_out, _ = self.task_decoder[task_id][i](decoder_inp)
                decoder_out = decoder_out.hidden_states[-1]
                del _
                torch.cuda.empty_cache()
                gc.collect()
                
                encoder_out_image_size = [shape//(2**(4-i+2)) for shape in self.config.image_size]
                upsampler_inp = decoder_out.view(batch_size, *encoder_out_image_size,-1).permute(0, 3, 1, 2)
                decoder_out = self.task_upsampler[task_id][i](upsampler_inp)
                del encoder_out, decoder_inp, upsampler_inp
                
            temp[task] = decoder_out
        
        return temp

class BottleNeck(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dataset = config.dataset
        
        if self.dataset=="PASCAL_MT":
            self.seq = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=config.embed_dim,
                    out_channels=config.embed_dim//2,
                    kernel_size=2,
                    stride=2
                ),
                nn.Conv2d(
                    in_channels=config.embed_dim//2,
                    out_channels=config.embed_dim//2,
                    kernel_size=1
                ),
                nn.ConvTranspose2d(
                    in_channels=config.embed_dim//2,
                    out_channels=config.embed_dim//4,
                    kernel_size=2,
                    stride=2
                )
            )
        else:
            self.seq = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=config.embed_dim,
                    out_channels=config.embed_dim//2,
                    kernel_size=2,
                    stride=2
                ),
                nn.Conv2d(
                    in_channels=config.embed_dim//2,
                    out_channels=config.embed_dim//2,
                    kernel_size=1
                ),
                nn.ConvTranspose2d(
                    in_channels=config.embed_dim//2,
                    out_channels=config.embed_dim//4,
                    kernel_size=2,
                    stride=2
                )
            )
        
    def forward(self, x):
        x = self.seq(x)
        return x # (batch_size, embed_dim//4, h, w)
    
class MLPHead(nn.Module):
    def __init__(self, config, num_classes) -> None:
        super().__init__()
        self.config = config
        
        if config.dataset=="PASCAL_MT":
            self.projection = nn.Conv2d(
                in_channels=config.embed_dim//4,
                out_channels=num_classes,
                kernel_size=1
            )
        else:
            self.projection = nn.Conv2d(
                in_channels=config.embed_dim//4,
                out_channels=num_classes,
                kernel_size=1
            )
    
    def forward(self, x):
        x = self.projection(x)
        return x
    
class MultiTaskModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
        self.swin_encoder_decoder = SwinEncoderDecoder(config)
        self.bottleneck = nn.ModuleDict()
        # self.bottleneck = BottleNeck(config)
        self.heads = nn.ModuleDict()
        for task, include_task in config.tasks.items():
            if include_task:
                self.heads[task] = MLPHead(config, config.num_classes[task])
                self.bottleneck[task] = BottleNeck(config)
        
    def forward(self, x):
        x = self.swin_encoder_decoder(x)
        
        out = {}
        for task, include_task in self.config.tasks.items():
            if include_task:
                out_task = self.bottleneck[task](x[task])
                out[task] = self.heads[task](out_task)
                
        return out
        
        


def get_model(config):
    model = MultiTaskModel(config)
    return model

if __name__ == "__main__":
    config = Config()
    train_transforms, val_transforms = get_transformations()
    train_dataset = get_train_dataset(config, train_transforms)
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    config = Config()
    model = MultiTaskModel(config=config).to("cuda:5")
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_million = total_params / 1e6
    print()
    print("----------------")
    print(f"Total parameters: {total_params_in_million:.2f} million")

    
    for i, data in enumerate(trainloader):
        out = model(data['image'].to("cuda:5"))
        for task in config.tasks.keys():
            print(task, out[task].shape)
            print(task, "original", data[task].shape)
            print()
            
        break
            

