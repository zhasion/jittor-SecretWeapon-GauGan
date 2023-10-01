# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import jittor as jt
import jittor.nn as nn
import jittor.nn as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def execute(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def execute(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: t.reshpae(), h = self.heads), qkv)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = jt.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = jt.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def execute(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size_w, image_size_h, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., nc = 1):
        super().__init__()
        image_height, image_width = image_size_h, image_size_w
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size
        self.nc = nc

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(jt.randn(1, num_patches + 1, dim))
        # self.pos_embedding = jt.zeros((1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(jt.randn(1, 1, dim))
        # self.cls_token = jt.zeros((1, 1, dim))
        self.cls_token = jt.randn((1, nc, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def execute(self, img):
        p = self.patch_size
        B, C, H, W = img.shape
        img = img.reshape(B, C, H//p, p, W//p, p).transpose(0, 2, 4, 3, 5, 1).reshape(B, H//p*W//p, p*p*C)
        # img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', ),
        x = self.to_patch_embedding(img)

        # random shuffle
        b, n, _ = x.shape
        temp = []
        for i in range(b):
            idx = jt.randperm(n)
            temp.append(x[i,idx,:])
        x = jt.stack(temp, dim = 0)

        cls_tokens = jt.repeat(self.cls_token, (b, 1, 1))
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = jt.concat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, :self.nc]

        return x
        # x = self.to_latent(x)
        # return self.mlp_head(x)
