import yaml
import wandb
import argparse
from src import build_dataset, build_model
import torch
import os

def main(config):
    
    wandb.init(project="EXPHY", name=config['exp']['name'])
    
    # Checkpoint dir
    save_path = os.path.join('checkpoints','stage_{}'.format(config['exp']['stage']),config['exp']['name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create dataset
    train_set,val_set = build_dataset(config['dataset'])

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config['training']['batch_size'],
            shuffle=True, 
            num_workers=config['training']['num_workers'],
            drop_last=True
            )
    
    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['training']['val_batch_size'],
            shuffle=True, 
            num_workers=config['training']['num_workers'],
            drop_last=True
            )

    ## Create  model
    model = build_model(config['model'])
    pretrained_path = config['training']['pretrained_path']
    if pretrained_path:
        model.load(pretrained_path)

    ## Will use all visible GPUs if parallel=True
    parallel = config['training']['parallel']
    if parallel and torch.cuda.device_count() > 1:
      print('Using {} GPUs'.format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
      module = model.module

    ## Set up optimizer and data logger
    optimizer = torch.optim.Adam(model.parameters(),lr=config['optimizer']['lr'], weight_decay=config['optimizer']['regularization'])
    model = model.to('cuda:0')
    for epoch in range(config['training']['epochs']):
            
            print('On epoch {}'.format(epoch))
            mbatch_cnt = 0
            for i,mbatch in enumerate(train_loader):
    
                x = mbatch.to('cuda:0')		
                ## Forward pass 
                ret = model.forward(x)	
                loss = ret['loss']
                mse = ret['mse']


                ## Backwards Pass
                if parallel:
                    loss = loss.mean()
                    mse = mse.mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
                optimizer.step()

                ## Print outputs & log
                if i%50==0:
                    wandb.log({"train_mse": mse.cpu().item()})
                    print('epoch:{}    mbatch:{}    loss:{:.0f}    MSE:{:.5f}'.format(epoch, mbatch_cnt, loss.item(), mse.item()))
                mbatch_cnt += 1

            ## Validation
            val_mse = 0
            if epoch % config['training']['val_interval'] == 0:
                model.eval()
                for mbatch in val_loader:
                    x = mbatch.to('cuda:0')
                    ret = model.forward(x)
                    val_mse += ret['mse'].cpu().item()
                val_mse = val_mse / len(val_loader.dataset)
                wandb.log({"val_mse": val_mse})
                wandb.log({"input_image": wandb.Image(mbatch[0])})
                rec = ret['reconstruction']
                wandb.log({"reconstructed_image": wandb.Image(rec[0].cpu())})
                
            if epoch % config['training']['save_interval'] == 0: 
                module.save(save_path,epoch=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EXPHY training')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config,'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    main(config)
