class SAGE(nn.Module):
    def __init__(self, c1, num_heads=8, topk_ratio=0.7, e_lambda=1e-4, dropout=0.0):
        super().__init__()
        assert c1 % num_heads == 0, f"c1={c1} must be divisible by num_heads={num_heads}"
        self.c = c1
        self.num_heads = num_heads
        self.dim_head = c1 // num_heads
        self.topk_ratio = topk_ratio
        self.e_lambda = e_lambda

        self.qkv = nn.Linear(c1, c1 * 3, bias=False)
        self.proj = nn.Linear(c1, c1)
        self.router = nn.Sequential(nn.Linear(c1, c1 // 4), nn.ReLU(inplace=True), nn.Linear(c1 // 4, num_heads))
        self.norm = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)

        # ============ 1. 计算空间自适应权重（基于原始特征）============
        # 只计算一次统计量，用于空间权重
        x_mu = x_flat.mean(dim=1, keepdim=True)
        x_var = ((x_flat - x_mu) ** 2).mean(dim=1, keepdim=True)
        x_sigma = (x_var + self.e_lambda).sqrt()

        # 空间重要性分数 (B, N, 1)
        spatial_weight = torch.sigmoid((x_flat - x_mu) / (x_sigma + 1e-8))

        # ============ 2. LayerNorm归一化（不破坏后续使用空间权重）============
        x_norm = self.norm(x_flat)

        # ============ 3. Router评分（融合空间信息）============
        router_scores = self.router(x_norm)  # (B, N, num_heads)

        # 将空间权重融入Router评分
        spatial_score = spatial_weight.mean(dim=-1, keepdim=True)  # (B, N, 1)
        router_scores = router_scores * spatial_score  # (B, N, num_heads)

        # ============ 4. QKV投影 ============
        qkv = self.qkv(x_norm).view(B, N, 3, self.num_heads, self.dim_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, dim_head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # ============ 5. Top-K稀疏选择（只选K和V，保留完整Q）============
        scores = router_scores.permute(0, 2, 1)  # (B, num_heads, N)
        topk = max(1, int(self.topk_ratio * N))
        _, indices = scores.topk(topk, dim=-1)  # (B, num_heads, topk)

        # 只对K和V进行稀疏选择
        idx_exp = indices.unsqueeze(-1).expand(-1, -1, -1, self.dim_head)
        k_selected = torch.gather(k, 2, idx_exp)  # (B, num_heads, topk, dim_head)
        v_selected = torch.gather(v, 2, idx_exp)

        # Q保持完整，与选中的K计算注意力
        attn_scores = torch.matmul(q, k_selected.transpose(-2, -1)) * (self.dim_head**-0.5)
        # attn_scores: (B, num_heads, N, topk)

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # 所有位置都有输出
        out = torch.matmul(attn, v_selected)  # (B, num_heads, N, dim_head)

        # ============ 6. 输出投影 ============
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return x + out
