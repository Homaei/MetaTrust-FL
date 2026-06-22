import torch
import torch.nn as nn

def get_tcn_params(in_channels=8, num_channels=[64, 64, 64, 64], k=3, use_wn=True, use_res=True):
    total = 0
    for i, out_c in enumerate(num_channels):
        in_c = in_channels if i == 0 else num_channels[i-1]
        
        # Conv
        w = out_c * in_c * k
        if use_wn:
            total += w + out_c + out_c  # v, g, b
        else:
            total += w + out_c
            
        # Residual
        if use_res and in_c != out_c:
            w = out_c * in_c * 1
            if use_wn:
                total += w + out_c + out_c
            else:
                total += w + out_c
    return total

print("Standard TCN 1-conv per block, WN, Res:", get_tcn_params(use_wn=True, use_res=True))
print("Standard TCN 1-conv per block, NO WN, Res:", get_tcn_params(use_wn=False, use_res=True))
print("Standard TCN 1-conv per block, NO WN, NO Res:", get_tcn_params(use_wn=False, use_res=False))

# what if 2 convs per block?
def get_tcn_2_params(in_channels=8, num_channels=[64, 64, 64, 64], k=3, use_wn=True):
    total = 0
    for i, out_c in enumerate(num_channels):
        in_c = in_channels if i == 0 else num_channels[i-1]
        # conv1
        total += out_c * in_c * k + (2*out_c if use_wn else out_c)
        # conv2
        total += out_c * out_c * k + (2*out_c if use_wn else out_c)
        # res
        if in_c != out_c:
            total += out_c * in_c * 1 + (2*out_c if use_wn else out_c)
    return total

print("Standard TCN 2-conv per block, WN:", get_tcn_2_params(use_wn=True))
print("Standard TCN 2-conv per block, NO WN:", get_tcn_2_params(use_wn=False))

# What if kernel size is different? Or one layer is 32?
for k in [2, 3]:
    for c1 in [32, 64]:
        for wn in [True, False]:
            for res in [True, False]:
                p = get_tcn_params(in_channels=8, num_channels=[c1, 64, 64, 64], k=k, use_wn=wn, use_res=res)
                if p == 33088:
                    print(f"FOUND! 1-conv: k={k}, channels=[{c1},64,64,64], wn={wn}, res={res}")
                p2 = get_tcn_2_params(in_channels=8, num_channels=[c1, 64, 64, 64], k=k, use_wn=wn)
                if p2 == 33088:
                    print(f"FOUND! 2-conv: k={k}, channels=[{c1},64,64,64], wn={wn}")

