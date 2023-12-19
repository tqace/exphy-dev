from .data import build_comphyImg_dataset 

def build_dataset(config):
    args = {
        'data_root':config['data_root']
        }
    if config['name'] == 'comphyImg':
        return build_comphyImg_dataset(args)
    else:
        raise NotImplementedError("This dataset is not implemented yet.")



