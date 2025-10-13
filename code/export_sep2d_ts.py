import torch
from train_tcn import TCN

# adjust if you trained a different tiny config
W = 256
model = TCN(in_ch=2, ch=64, out_ch=2, k=5, blocks=6,
            dropout=0.05, use_bn=False, residual=True,
            separable=True, sep2d=True).eval()

ck = torch.load("tcn_denoiser_ema_model.pt", map_location="cpu")
sd = ck["model"] if "model" in ck else ck["ema_state"]
model.load_state_dict(sd, strict=True)

class OnlyY(torch.nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, x):
        y, _ = self.core(x)
        return y

wrapped = OnlyY(model).eval()
example = torch.randn(1, W, 2)

# Minimal, robust TS export (no freeze/optimize to avoid JIT IR attrs)
ts = torch.jit.trace(wrapped, example, strict=False)
ts.save("tcn_sep2d64x6_W256.ts")
print("saved -> tcn_sep2d64x6_W256.ts")
