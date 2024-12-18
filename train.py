from MicroRTS_NB import GameState, UnitTypeTable
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
#from ai.rush.CombatRush import CombatRush
#from synthesis.extent1DSL.featureExtractor.SMFE import SMFE
from featureExtractor2 import FeatureExtractor2
from vqvae import VQVAE
from dataSetFeatureExtractor import DataSetFeatureExtractor, ReadDataFeatureExtractor
import torch.nn.functional as F
#from utils import get_data_loader, count_parameters, save_img_tensors_as_grid
import uuid
from torch.utils.data import DataLoader

class FeatureExtractorTrain():
    #https://github.com/K3dA2/Muse/blob/main/model.py
    @staticmethod
    def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, valid_loader,
                    max_grad_norm=1.0, epoch_start=0, save_img=True, show_img=False,reset = False,
                    ema_alpha=0.99,usage_threshold=1.0):
        with open("trainDatas/losses.txt", "w") as file:
            pass
        model.train()
        ema_loss = None
        scheduler = None
        previous_loss = None

        for epoch in range(epoch_start, n_epochs):
            loss_train = 0.0
            mse_loss_train = 0.0
            vq_loss_train = 0.0

            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
            for batch_idx, imgs  in enumerate(progress_bar):
                imgs = imgs.to(device)

                outputs, vq_loss = model(imgs)
                mse_loss = loss_fn(outputs, imgs)
                loss = mse_loss + vq_loss

                loss_train += loss.item()
                mse_loss_train += mse_loss.item()
                vq_loss_train += vq_loss.item()

                loss.backward()

                
                #utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.set_postfix(loss=loss.item(), mse_loss=mse_loss.item(), vq_loss=vq_loss.item())

            avg_loss_train = loss_train / len(data_loader)
            avg_mse_loss_train = mse_loss_train / len(data_loader)
            avg_vq_loss_train = vq_loss_train / len(data_loader)

            
            
            

            print('{} Epoch {}, Training loss {:.4f}, MSE loss {:.4f}, VQ loss {:.4f}'.format(
                datetime.datetime.now(), epoch, avg_loss_train, avg_mse_loss_train, avg_vq_loss_train))

            if epoch % 1 == 0:
                # Validation phase
                model.eval()
                loss_val = 0.0
                with torch.no_grad():
                    for imgs in valid_loader:
                        imgs = imgs.to(device)
                        outputs, vq_loss = model(imgs)
                        mse_loss = loss_fn(outputs, imgs)
                        loss = mse_loss + vq_loss
                        loss_val += loss.item()

                avg_loss_val = loss_val / len(valid_loader)
                print(f'Val loss: {avg_loss_val}')
            with open("trainDatas/losses.txt", "a") as file:
                file.write(f"{avg_loss_train}\t{avg_mse_loss_train}\t{avg_vq_loss_train}\t{avg_loss_val}\n")

            '''
            if epoch % 1 == 0:
                

                model_path = os.path.join('weights/', 'waifu-vqvae_epoch.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
                # Reset underused embeddings
            
            # Reset underused embeddings conditionally
            if reset:
                if epoch > 5 and previous_loss is not None and avg_loss_train > previous_loss * 1.25:
                    print("reseting")
                    with torch.no_grad():
                        for batch_imgs, _ in data_loader:
                            model.reset_underused_embeddings(batch_imgs.to(device), threshold=usage_threshold)
                            break
            '''
            previous_loss = avg_loss_train



    @staticmethod
    def test0():
        path = 'synthesis/extent1DSL/featureExtractor/model/'
        val_path = 'synthesis/extent1DSL/featureExtractor/datas/'
        model_path = 'synthesis/extent1DSL/featureExtractor/model/models/'
        epoch = 0

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        print(f"using device: {device}")

        model = VQVAE()  # Assuming Unet is correctly imported and defined
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        loss_fn = nn.MSELoss().to(device)

        #print(count_parameters(model))
        datas = ReadDataFeatureExtractor.read(device)
        datasX = datas[:9]
        datasY = datas[:3] 
        print("n_train",len(datasX))
        print("n_test",len(datasY))
        data_loader = DataLoader(datasX, batch_size=3, shuffle=True)
        val_loader = DataLoader(datasY, batch_size=3, shuffle=False)

        
        # Optionally load model weights if needed
        '''
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        '''
        '''
        with torch.no_grad():
            for valid_tensors, _ in val_loader:
                break

            save_img_tensors_as_grid(valid_tensors, 4, "true")
            val_img, _ = model(valid_tensors.to(device))
            save_img_tensors_as_grid(val_img, 4, "recon")

        with torch.no_grad():
            for valid_tensors, _ in data_loader:
                break

            save_img_tensors_as_grid(valid_tensors, 4, "true1")
            val_img, _ = model(valid_tensors.to(device))
            save_img_tensors_as_grid(val_img, 4, "recon1")
        '''
        
        FeatureExtractorTrain.training_loop(
            n_epochs=25,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            device=device,
            data_loader=data_loader,
            valid_loader=val_loader,
            epoch_start=epoch + 1,
        )
        
        
        img : torch.tensor= datas[4]
        featureExtractor =FeatureExtractor2()
        y = model(img.unsqueeze(0))[0]
        example = y.view(5,3,16,16)
        for i in range(5):
            img = example.detach().numpy()
        
            featureExtractor.viewFeature(img[i])
            
    @staticmethod
    def test1():
        path = 'model/'
        val_path = 'datas/'
        model_path = 'models/'
        epoch = 0

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        print(f"using device: {device}")

        model = VQVAE()  # Assuming Unet is correctly imported and defined
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4)
        loss_fn = nn.MSELoss().to(device)

        #print(count_parameters(model))
        datas = ReadDataFeatureExtractor.read(device)
        datasX , datasY = train_test_split(datas, test_size=0.2, random_state=42)
        print("n_train",len(datasX))
        print("n_test",len(datasY))
        input()
        data_loader = DataLoader(datasX, batch_size=32, shuffle=True)
        val_loader = DataLoader(datasY, batch_size=32, shuffle=False)

        
        FeatureExtractorTrain.training_loop(
            n_epochs=40,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            device=device,
            data_loader=data_loader,
            valid_loader=val_loader,
            epoch_start=epoch + 1,
        )
        torch.save(model.state_dict(),model_path+ 'modeloTrained.pth')
    
    '''
    @staticmethod
    def test2():
        model = VQVAE()  
        model_path = 'synthesis/extent1DSL/featureExtractor/model/models/'
        model.load_state_dict(torch.load(model_path+ 'modeloTrained.pth'))  # Carregue os pesos
        model.eval()
        map = "./maps/basesWorkers16x16A.xml"
        utt = UnitTypeTable(2)
      
        gs = GameState(map,utt)
        pgs = gs.getPhysicalGameState()
        ai0 = CombatRush(pgs,utt,"Heavy")
        ai1 = CombatRush(pgs,utt,"Ranged")
        sm = SMFE()
        featureExtractor =FeatureExtractor2()
        r,fes = sm.playout(gs,ai0,ai1,0,3000,False)
        inp = torch.tensor(np.array(fes)).reshape(15,16,16).unsqueeze(0).to(torch.float32)
        print(inp.shape)
        
        img = model(inp)[0].reshape(5,3,16,16)
        inp=inp.reshape(5,3,16,16)
        print(img.shape)
        for i in range(5):
            featureExtractor.viewFeature(inp[i])
            featureExtractor.viewFeature(img[i].detach().numpy())
        
        
        
    @staticmethod
    def test3():
        model = VQVAE()  
        model_path = 'synthesis/extent1DSL/featureExtractor/model/models/'
        model.load_state_dict(torch.load(model_path+ 'modeloTrained.pth'))  # Carregue os pesos
        model.eval()
        map = "./maps/basesWorkers16x16A.xml"
        utt = UnitTypeTable(2)
      
        gs = GameState(map,utt)
        pgs = gs.getPhysicalGameState()
        ai0 = CombatRush(pgs,utt,"Light")
        ai1 = CombatRush(pgs,utt,"Ranged")
        sm = SMFE()
        featureExtractor =FeatureExtractor2()
        
        r,fes = sm.playout(gs,ai0,ai1,0,3000,False)
        with torch.no_grad():
            inp = torch.tensor(np.array(fes)).reshape(15,16,16).unsqueeze(0).to(torch.float32)
            codebook_entries = model.codebook.weight.unsqueeze(0).detach()

            codebook_entries
            z  = model.encoder(inp)
            _,n,d = codebook_entries.shape
            distances = torch.cdist(z, codebook_entries, p=2)
            min_distances, indices = torch.min(distances, dim=-1)
            carac = [ 1, 6 ,8, 5,
                     8,6,2,4,
                     1,3,3,5,
                     9,4,2,1]
            soma = 0
            for i in range(16):

                cos_sim = F.cosine_similarity(codebook_entries[0][carac[i]], z[0][i], dim=0)
                soma+=cos_sim.item()
            print(soma/16)]
    '''
                
     
           
        

        