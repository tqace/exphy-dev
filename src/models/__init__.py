from .stage1 import IODINE

def build_model(params):
    if params['name'] == 'ExphyS1Base':
        return IODINE(
                T = params['T'],
                K = params['K'],
                beta = params['beta'], 
                a_dim = params['a_dim'],
                resolution = params['resolution'],
                use_feature_extractor = params['use_feature_extractor']
                )
    else:
        raise NotImplementedError("This model is not implemented yes.")

