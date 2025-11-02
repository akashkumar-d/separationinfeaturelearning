import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import vmap, jacrev
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utilities import *

torch.autograd.set_detect_anomaly(True)

import numpy as np

class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ThreeLayerNet, self).__init__()
        # Define three linear layers without biases.
        self.fc1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.fc3 = nn.Linear(hidden_size2, output_size, bias=False)
        self.d = input_size
        self.h1 = hidden_size1
        self.h2 = hidden_size2
        
        self.body = nn.Sequential(*[self.fc1, nn.ReLU(),
                                  self.fc2, nn.ReLU(),])
        self.head =  self.fc3


    def forward(self, x):
        return self.head(self.body(x))
    
    def initialize(self):
        self.fc1.weight.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 1)
        self.fc3.weight.data.normal_(0, 1)
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.d = input_size
        self.h = hidden_size
        self.body = nn.Sequential(*[self.fc1, nn.ReLU(),])
        self.head =  self.fc2

    def forward(self, x):
        return self.head(self.body(x))

    def initialize(self):
        self.fc1.weight.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 1)  

def run_optimizer_experiment(
    optimizer_cls,         
    optimizer_kwargs,      
    batch_size,            
    num_repeats,           
    num_epochs=2000,       
    seed=37,               
    output_dir="optimizers_vs_alignment",
    early_stop=False,
    three_layer_net=False,
    two_layer_net=False,
    link_function=None,
):


    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if optimizer_cls.__name__ == 'SGD' and 'weight_decay' in optimizer_kwargs:
        exp_name  = f"{optimizer_cls.__name__}_WD_{timestamp}"
    else:
        exp_name  = f"{optimizer_cls.__name__}_{timestamp}"
    exp_path  = os.path.join(output_dir, exp_name)
    os.makedirs(exp_path, exist_ok=False)
    print(f'Created folder {exp_path}')
    
    #collect call arguments
    call_info = dict(
    batch_size=batch_size,
    num_epochs=num_epochs,
    num_repeats=num_repeats,
    seed=seed,
    early_stop=early_stop,
    **optimizer_kwargs,          # lr, momentum, …
    )

    # build two strings: one for the title, one for a small foot-note
    title_str = f"{optimizer_cls.__name__}"
    param_str = ", ".join(f"{k}={v}" for k, v in call_info.items())

    # write config
    cfg = {
        'optimizer': optimizer_cls.__name__,
        **optimizer_kwargs,
        'batch_size': batch_size,
        'num_repeats': num_repeats,
        'num_epochs': num_epochs,
        'seed': seed,
    }
    print(f"[INFO] Writing config to {exp_path}/config.txt")
    with open(os.path.join(exp_path, "config.txt"), "w") as f:
        for k, v in cfg.items():
            if k == 'SGD' and ('momentum' in optimizer_kwargs):
                k = 'SGD_WD' if optimizer_kwargs['momentum'] > 0. else 'SGD'
            f.write(f"{k}: {v}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

    print(f"[INFO] Starting {num_repeats} repetitions of {optimizer_cls.__name__}")
    for rep in range(num_repeats):
        torch.manual_seed(seed + rep)
        np.random.seed(seed + rep)

#         # --- data setup ---
#         d, n, nval = 100, 1024, 5000
#         EGOP = torch.zeros((d, d), device=device)
#         EGOP[0, 0] = EGOP[1, 1] = 1.
#         EGOP_np = EGOP.cpu().numpy()

#         X = torch.randn(n + nval, d, device=device)
#         Y = (X[:, 0] * X[:, 1] )\
#                 .unsqueeze(1)

#         Xval, Yval = X[n:], Y[n:]
#         Xtrain, Ytrain = X[:n], Y[:n]
#         Ytrain += 0.1 * torch.randn(n, device=device)
                
#         model = ThreeLayerNet(d, 512, 512, 1).to(device)

        # --- data setup ---
        d, n, nval = 100, 2048, 5000
        # EGOP = torch.zeros((d, d), device=device)
        # EGOP[0, 0] = EGOP[1, 1] = 1.
        # EGOP_np = np.zeros((d,d))
        # EGOP_np[:5,:5] = np.array([   [2, 1, 1, 0, 0],
        #                     [1, 2, 1, 0, 0],
        #                     [1, 1, 1, 0, 0],
        #                     [0, 0, 0, 3, 0],
        #                     [0, 0, 0, 0, 3]] )
        EGOP_np = np.zeros((d,d))
        EGOP_np[:3,:3] = np.array([   [1, 1, 0,],
                            [1, 1, 0,],
                            [0, 0, 0,],
                                  ])
        #
        # # EGOP_np = EGOP.cpu().numpy()

        X = torch.randn(n + nval, d, device=device)
        if link_function is not None:
            Y = link_function(X)
            Y = Y.unsqueeze(1)
            print(Y.shape)

            X.requires_grad = True
            # Y = Y.detach()
            print(Y.shape)
            def f_single(x1d):           # x1d: (d,) -> (m,)
                return link_function(x1d.unsqueeze(0)).squeeze(0)  # (m,)
            J = vmap(jacrev(f_single))(X)    # (B, m, d)
            X.requires_grad = False
            J = J.detach()
            # print(J.shape)
            J_flat = J.reshape(-1, J.shape[-1])   # (B*m, d)
            EGOP = (J_flat.T @ J_flat) / J_flat.shape[0]
            # nabla_XofY = torch.autograd.grad(Y, X, create_graph=True)[0]
            # EGOP = nabla_XofY.T @ nabla_XofY
            # nabla_XofY = vmap(jacrev(link_function))(X)
            # EGOP = nabla_XofY.T @ nabla_XofY
            EGOP_np = EGOP.cpu().numpy()    
            # print(EGOP_np.shape)
            Xval, Yval = X[n:], Y[n:]
            Xtrain, Ytrain = X[:n], Y[:n]
            print(f"Ytrain shape: {Ytrain.shape}")
            noise = 0.1 * torch.randn((n,1), device=device)
            print(f'noise shape: {noise.shape}')
            Ytrain += noise
            print(EGOP_np[:10,:10])
        else:
            Y = torch.relu(X[:, 0]  + X[:, 1]) # + ( X[:,0] + X[:,1] + X[:,2] )*X[:,3]*X[:,4]
            Y = Y.unsqueeze(1)
            Xval, Yval = X[n:], Y[n:]
            Xtrain, Ytrain = X[:n], Y[:n]
            Ytrain += 0.1 * torch.randn(n, device=device).unsqueeze(1)

        assert three_layer_net or two_layer_net, "Either three_layer_net or two_layer_net must be True"

        if three_layer_net:
            model = ThreeLayerNet(d, 1024, 1024, 1).to(device)
        elif two_layer_net:
            model = TwoLayerNet(d, 1024, 1).to(device)
        else:
            raise ValueError("Invalid model type")
            X.requires_grad = False

        #####################
        # model.initialize()
        
        if not 'Muon' in optimizer_cls.__name__ :
            optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        else:
            hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
            nonhidden_params = [*model.head.parameters(), ]
            # param_groups = [
            #     dict(params=hidden_weights, use_muon=True,
            #          lr=0.02, weight_decay=0.01),
            #     dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
            #          lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
            # ]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                     lr=optimizer_kwargs['lr'], weight_decay=optimizer_kwargs['weight_decay'], momentum=0.),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                     betas=(0., 0.), lr=optimizer_kwargs['lr'], weight_decay=optimizer_kwargs['weight_decay'], momentum=0.),
            ]
            optimizer = optimizer_cls(param_groups, )
        print('*'*15)
        print(optimizer)
        print(optimizer_kwargs)
        print('*'*15)

        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(Xtrain, Ytrain),
            batch_size=batch_size,
            shuffle=True
        )

        train_losses = []
        val_losses   = []
        agop_align   = []


        # epoch loop with progress bar
        for epoch in range(num_epochs):
            batch_losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_losses.append(np.mean(batch_losses))
            
            with torch.no_grad():
                agop_tensor = get_AGOP(model, Xtrain)
                agop_np     = agop_tensor
                aa = mat_cos(agop_np, EGOP_np)
                agop_align.append(aa)
                val_losses.append(criterion(model(Xval), Yval).item())
                
            # print percentage progress (overwrites same line)
            pct = (epoch + 1) / num_epochs * 100
            print(f"  Epoch {epoch+1}/{num_epochs} ({pct:5.1f}%)", end='\r', flush=True)
            
            if early_stop and np.mean(batch_losses) <= 1e-8:
                print(f"Loss <=1e-8 achieved at iter. {epoch}. Stopping training.")
                break
            if early_stop and aa >= 0.95 and np.mean(batch_losses) <= 1e-4:
                break
        # save .npy files
        print(f"[INFO] Saving .npy for repeat {rep+1}")
        np.save(os.path.join(exp_path, f"train_losses_rep{rep}.npy"), np.array(train_losses))
        np.save(os.path.join(exp_path, f"val_losses_rep{rep}.npy"),   np.array(val_losses))
        np.save(os.path.join(exp_path, f"agop_align_rep{rep}.npy"),   np.array(agop_align))

        all_results.append({
            'train': train_losses,
            'val':   val_losses,
            'align': agop_align
        })

    print(f"[INFO] Aggregating results and plotting")
    
    max_len = max( map(len, [r['train'] for r in all_results] ) )
    # np.vstack([np.pad(a, (0, max_len - len(a)), mode='edge') for a in arrays])
    arr_train = np.stack([ np.pad(r['train'], (0, max_len - len(r['train'])), mode='edge') for r in all_results])
    arr_val   = np.stack([ np.pad(r['val'], (0, max_len - len(r['val'])), mode='edge') for r in all_results])
    arr_align = np.stack([ np.pad(r['align'], (0, max_len - len(r['align'])), mode='edge') for r in all_results]) 
    # arr_train = np.stack([r['train'] for r in all_results])
    # arr_val   = np.stack([r['val']   for r in all_results])
    # arr_align = np.stack([r['align'] for r in all_results])

    mean_train = arr_train.mean(axis=0)
    mean_val   = arr_val.mean(axis=0)
    mean_align = arr_align.mean(axis=0)

    ci_train = 1.96 * arr_train.std(axis=0) / np.sqrt(num_repeats)
    ci_val   = 1.96 * arr_val.std(axis=0)   / np.sqrt(num_repeats)
    ci_align = 1.96 * arr_align.std(axis=0) / np.sqrt(num_repeats)

    epochs = np.arange(1, min([max_len, num_epochs]) + 1)

    # build a human‐readable title
    params_str = ", ".join(f"{k}={v}" for k, v in optimizer_kwargs.items())
    title_str  = f"{optimizer_cls.__name__} ({params_str})"

    # 1) create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=250)

    # 2) plot your train & val losses on ax1
    ax1.plot(epochs, mean_train, label='train')
    ax1.set_yscale('log')
    ax1.fill_between(epochs,
                     mean_train - ci_train,
                     mean_train + ci_train,
                     alpha=0.3)
    ax1.plot(epochs, mean_val, label='val')
    ax1.fill_between(epochs,
                     mean_val - ci_val,
                     mean_val + ci_val,
                     alpha=0.3)
    ax1.set_title('MSE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 3) plot your AGOP alignment on ax2
    ax2.plot(epochs, mean_align, label='align', color="tab:green")
    ax2.fill_between(epochs,
                     mean_align - ci_align,
                     mean_align + ci_align,
                     alpha=0.3, color="tab:green")
    ax2.set_title('AGOP Alignment')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Align')

    # 4) add the overall suptitle
    # fig.suptitle(title_str, fontsize=16, y=1.02)
    fig.suptitle(title_str, fontsize=16, y=1.05)        # big headline
    fig.text(0.5, 0.01, param_str, ha='center',          # small footer
             fontsize=9, color="tab:purple", wrap=True)

    # 5) layout, save, close
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "results.png"), bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Experiment complete. All outputs in: {exp_path}")
    all_results.append(fig)
    return all_results