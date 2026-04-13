class EASA(nn.Module):
    """改进1: 降低下采样率 改进2: 使用双线性插值 改进3: 添加最小尺寸保护.
    """

    def __init__(self, dim=36, down_scale=1):  #  添加可配置参数
        super().__init__()
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = down_scale  #  从8改为4（可配置）
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, X):
        _b, _c, h, w = X.shape

        #  改进：设置最小下采样尺寸为8
        down_h = max(8, h // self.down_scale)  # 原本是 max(1, ...)
        down_w = max(8, w // self.down_scale)

        pooled = F.adaptive_max_pool2d(X, (down_h, down_w))  # 最大池化
        x_s = self.dw_conv(pooled)
        x_v = torch.var(X, dim=(-2, -1), keepdim=True)
        Temp = x_s * self.alpha + x_v * self.belt

        # 改进：使用双线性插值替代最近邻
        attn = F.interpolate(
            self.gelu(self.linear_1(Temp)),
            size=(h, w),
            mode="bilinear",  # 从 'nearest' 改为 'bilinear'
            align_corners=False,
        )

        x_l = X * attn
        out = self.linear_2(x_l) + X

        return out


class EGConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, use_easa=True):
        super().__init__()

        assert c1 > 0 and c2 > 0, f"Invalid channels: c1={c1}, c2={c2}"

        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)

        self.use_easa = use_easa
        if use_easa:
            self.easa = EASA(dim=c_)

        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)

        if self.use_easa:
            y = self.easa(y)

        y2 = self.cv2(y)
        out = torch.cat((y, y2), 1)

        return out
