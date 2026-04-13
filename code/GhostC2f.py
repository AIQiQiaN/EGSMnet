class GhostC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数

        # 输入卷积：使用GhostConv
        self.cv1 = GhostConv(c1, 2 * self.c, 1, 1)

        # Bottleneck序列
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))

        # 输出卷积：标准Conv用于特征融合
        self.cv2 = Conv(2 * self.c + n * self.c, c2, 1)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))  # Split
        y.extend(m(y[-1]) for m in self.m)  # 逐层添加Bottleneck输出
        return self.cv2(torch.cat(y, 1))  # Concat + 融合
