from utils.general import *


def strip_optimizer_yolor(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    #x['model'].half()  # to FP16
    #for p in x['model'].parameters():
    #    p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


not_optimized = Path.cwd() / 'optimizer' / 'not_optimized'
optimized = Path.cwd() / 'optimizer' / 'optimized'
name_optimized = set(path.name for path in optimized.glob('*.pt'))
for model_path in (path for path in not_optimized.glob('*.pt') if path.name not in name_optimized):
    if 'yolor' in str(model_path):
        strip_optimizer_yolor(model_path, optimized / model_path.name)
    else:
        strip_optimizer(model_path, optimized / model_path.name)
