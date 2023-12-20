import torch
import torchvision.models as TorchModels
import torch.nn.init as init
from .refine_net import RefineNetLSTM
from .sbd import SBD
import random
import scipy.optimize
from torch import nn
import math
import os
import numpy as np
import sys

class IODINE(torch.nn.Module):

        def __init__(self,
                T,
                K,
                a_dim,
                resolution,
                beta=1.,
                use_feature_extractor=True
                ):
            
            super(IODINE, self).__init__()
            self.lmbda0 = torch.nn.Parameter(torch.rand(1,2*a_dim)-0.5,requires_grad=True)
            self.decoder = SBD(a_dim, resolution, out_channels=4)
            self.refine_net = RefineNetLSTM(a_dim,32)
            
            self.layer_norms = torch.nn.ModuleList([
                    torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                    torch.nn.LayerNorm((3,64,96),elementwise_affine=False),
                    torch.nn.LayerNorm((1,64,96),elementwise_affine=False),
                    torch.nn.LayerNorm((2*a_dim,),elementwise_affine=False),
                    ])

            self.use_feature_extractor = use_feature_extractor 
            if self.use_feature_extractor:
                feature_extractor = TorchModels.squeezenet1_1(pretrained=True).features[:5] 
                self.feature_extractor = torch.nn.Sequential(
                        feature_extractor,
                        torch.nn.Conv2d(128,64,3,stride=1,padding=1),
                        torch.nn.ELU(),
                        torch.nn.Conv2d(64,32,3,stride=1,padding=1),
                        torch.nn.ELU(),
                        torch.nn.Conv2d(32,16,3,stride=1,padding=1),
                        torch.nn.ELU())
                for param in self.feature_extractor[0]:
                        param.requires_grad = False
            
            self.register_buffer('T', torch.tensor(T))
            self.register_buffer('K', torch.tensor(K))
            self.register_buffer('a_dim', torch.tensor(a_dim))
            self.register_buffer('var_x', torch.tensor(0.3))
            self.register_buffer('h0',torch.zeros((1,128)))
            self.register_buffer('base_loss',torch.zeros(1,1))
            self.register_buffer('b', torch.tensor(beta)) ## Weight on NLL component of loss
        
        def forward(self, x):
            N,C,H,W = x.shape
            K, T, a_dim = self.K, self.T, self.a_dim
            assert not torch.isnan(self.lmbda0).any().item(), 'lmbda0 has nan'
            
            ## Initialize parameters for latents' distribution
            lmbda = self.lmbda0.expand((N*K,)+self.lmbda0.shape[1:])
            total_loss = torch.zeros_like(self.base_loss.expand((N,1)))
            ## Initialize LSTMCell hidden states
            h = self.h0.expand((N*K,)+self.h0.shape[1:]).clone().detach()
            c = torch.zeros_like(h)
            assert h.max().item()==0. and h.min().item()==0.

            for it in range(T):
                ## Sample latent code
                mu_a, logvar_a = lmbda.chunk(2,dim=1)
                mu_a, logvar_a = mu_a.contiguous(), logvar_a.contiguous()
                a = self._sample(mu_a,logvar_a) 
                ## Get means and masks 
                dec_out = self.decoder(a) 
                mu_x, mask_logits = dec_out[:,:C,:,:], dec_out[:,C,:,:] 
                mask_logits = mask_logits.view((N,K,)+mask_logits.shape[1:]) 
                mu_x = mu_x.view((N,K,)+mu_x.shape[1:]) 
                ## Process masks
                masks = torch.nn.functional.softmax(mask_logits,dim=1).unsqueeze(dim=2) 
                mask_logits = mask_logits.unsqueeze(dim=2) 
                ## Calculate loss: reconstruction (nll) & KL divergence
                _x = x.unsqueeze(dim=1).expand((N,K,)+x.shape[1:])
                deviation = -1.*(mu_x - _x)**2
                ll_pxl_channels = ((masks*(deviation/(2.*self.var_x)).exp()).sum(dim=1,keepdim=True)).log()
                assert ll_pxl_channels.min().item()>-math.inf
                ll_pxl = ll_pxl_channels.sum(dim=2,keepdim=True) 
                ll_pxl_flat = ll_pxl.view(N,-1)
                nll = -1.*(ll_pxl_flat.sum(dim=-1).mean())
                div_a = self._get_div(mu_a,logvar_a,N,K)
                loss = self.b * nll + div_a
                ## Accumulate loss
                scaled_loss = ((float(it)+1)/float(T)) * loss
                total_loss += scaled_loss

                assert not torch.isnan(loss).any().item(), 'Loss at t={} is nan. (nll,div): ({},{})'.format(nll,div)
                if it==T-1: continue

                ## Refine lambda
                refine_inp_rec = self.get_refine_inputs([_x],[mu_x],[masks],[mask_logits],[ll_pxl],lmbda,loss,[deviation],norm_ind=3)

                ## Potentially add additional features from pretrained model (scaled down to appropriate size)
                if self.use_feature_extractor:
                        x_resized = torch.nn.functional.interpolate(x,(257,385)) ## Upscale to desired input size for squeezenet
                        additional_features = self.feature_extractor(x_resized).unsqueeze(dim=1)
                        additional_features = additional_features.expand((N,K,16,64,96)).contiguous()
                        additional_features = additional_features.view((N*K,16,64,96))
                        refine_inp_rec['img'] = torch.cat((refine_inp_rec['img'],additional_features),dim=1)

                delta, h, c = self.refine_net(refine_inp_rec, h, c)
                lmbda = lmbda + delta
                    
            rec = (mu_x * masks).sum(dim=1)
            mse = torch.nn.functional.mse_loss(rec,x)

            return {'loss':total_loss, 'reconstruction':rec, 'mse':mse}


        def get_refine_inputs(self, _x_all,mu_x_all,masks_all,mask_logits_all,ll_pxl_all,lmbda,loss,deviation_all,norm_ind):
            N,K,C,H,W = mu_x_all[0].shape
            
            ## Calculate additional non-gradient inputs
            ll_pxl_all = [ll_pxl.expand((N,K,) + ll_pxl.shape[2:]) for ll_pxl in ll_pxl_all] 
            p_mask_individual_all =[(deviation/(2.*self.var_x)).exp().prod(dim=2,keepdim=True) for deviation in deviation_all] 
            p_masks_all = [torch.nn.functional.softmax(p_mask_individual, dim=1) for p_mask_individual in p_mask_individual_all] 
            
            ## Calculate gradient inputs
            dmu_x_all = [torch.autograd.grad(loss,mu_x,retain_graph=True,only_inputs=True)[0] for mu_x in mu_x_all]
            dmasks_all = [torch.autograd.grad(loss,masks,retain_graph=True,only_inputs=True)[0] for masks in masks_all] 
            dlmbda = torch.autograd.grad(loss,lmbda,retain_graph=True,only_inputs=True)[0] 

            ## Apply layer norm
            ll_pxl_stable_all = [self.layer_norms[0](ll_pxl).detach() for ll_pxl in ll_pxl_all]
            dmu_x_stable_all = [self.layer_norms[1](dmu_x).detach() for dmu_x in dmu_x_all]
            dmasks_stable_all = [self.layer_norms[2](dmasks).detach() for dmasks in dmasks_all]
            dlmbda_stable = self.layer_norms[norm_ind](dlmbda).detach()
            
            ## Generate coordinate channels
            H,W = (64,96)
            x_range = torch.linspace(-1.,1.,H)
            y_range = torch.linspace(-1.,1.,W)
            x_grid, y_grid = torch.meshgrid([x_range,y_range])
            x_grid =  x_grid.view((1, 1) + x_grid.shape).cuda()
            y_grid = y_grid.view((1, 1) + y_grid.shape).cuda()
            x_mesh = x_grid.expand(N,K,-1,-1,-1).contiguous()
            y_mesh = y_grid.expand(N,K,-1,-1,-1).contiguous()
            #### cat xxx_all
            _x_all = torch.cat(_x_all,dim=2)
            mu_x_all = torch.cat(mu_x_all, dim=2)
            masks_all = torch.cat(masks_all, dim=2)
            mask_logits_all = torch.cat(mask_logits_all, dim=2)
            dmu_x_stable_all = torch.cat(dmu_x_stable_all,dim=2)
            dmasks_stable_all = torch.cat(dmasks_stable_all, dim=2)
            p_masks_all = torch.cat(p_masks_all, dim=2)
            ll_pxl_stable_all = torch.cat(ll_pxl_stable_all,dim=2)


            ## Concatenate into vec and mat inputs
            img_args = (_x_all, mu_x_all,masks_all,mask_logits_all,dmu_x_stable_all,dmasks_stable_all,
                    p_masks_all,ll_pxl_stable_all,x_mesh,y_mesh)
            vec_args = (lmbda, dlmbda_stable)
            

            img_inp = torch.cat(img_args,dim=2)
            vec_inp = torch.cat(vec_args,dim=1)
            img_inp = img_inp.view((N*K,)+img_inp.shape[2:])

            return {'img':img_inp, 'vec':vec_inp}

        """
        Computes the KL-divergence between an isotropic Gaussian distribution over latents
        parameterized by mu_a and logvar_a and the standard normal
        """
        def _get_div(self,mu_a,logvar_a,N,K):
                kl = ( -0.5*((1.+logvar_a-logvar_a.exp()-mu_a.pow(2)).sum(dim=1)) ).view((N,K))
                return (kl.sum(dim=1)).mean()

        """
        Implements the reparameterization trick
        Samples from standard normal and then scales and shifts by var and mu
        """
        def _sample(self,mu,logvar):
                std = torch.exp(0.5*logvar)
                return mu + torch.randn_like(std)*std

        """
        Save the current IODINE model
        """
        def save(self,save_path,epoch=None):
                print('Saving model at epoch {}'.format(epoch))
                suffix = 'epoch_{}.th'.format(epoch)
                model_save_path = os.path.join(save_path, suffix)
                torch.save(self.state_dict(),model_save_path)

        """
        Loads weights for the IODINE model
        """
        def load(self,load_path,map_location='cpu'):
            model_dict = self.state_dict()
            state_dict = torch.load(load_path,map_location='cpu')
            new_state_dict = {key : state_dict[key] for key in state_dict}
            model_dict.update(new_state_dict)
            self.load_state_dict(model_dict)
            print('load ckpt successfully')

        """
        Checks if any of the model's weights are NaN
        """
        def has_nan(self):
                for name,param in self.named_parameters():
                        if torch.isnan(param).any().item():
                                print(param)
                                assert False, '{} has nan'.format(name)

        """
        Checks if any of the model's weight's gradients are NaNs
        """
        def grad_has_nan(self):
                for name,param in self.named_parameters():
                        if torch.isnan(param.grad).any().item():
                                print(param)
                                print('---------')
                                print(param.grad)
                                assert False, '{}.grad has nan'.format(name)

