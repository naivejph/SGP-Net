import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  Module 1 -- Spectral Prototype Bank
# =============================================================================
class SpectralPrototypeBank(nn.Module):
    def __init__(self, dim: int = 512, n_bands: int = 3):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        self._raw_r1 = nn.Parameter(torch.tensor(-1.2567))      # softplus -> ~0.25
        self._raw_r2gap = nn.Parameter(torch.tensor(-0.7253))   # softplus -> ~0.30
        # Softness of the transitions (higher = sharper). Init gives a gentle
        # roll-off that lets gradients flow even at the boundary.
        self._raw_sharp = nn.Parameter(torch.tensor(2.5))

    @staticmethod
    def _masked_avg(fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        num = (fmap * mask.unsqueeze(1)).sum(dim=(-2, -1))
        den = mask.sum(dim=(-2, -1)).clamp_min(1e-5).unsqueeze(1)
        return num / den

    def _make_radial_masks(self, h: int, w: int, device, dtype):
        fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)       # [h]
        fx = torch.fft.rfftfreq(w, d=1.0).to(device=device, dtype=dtype)      # [w//2+1]
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        rho = torch.sqrt(gy * gy + gx * gx)                                   # [h, w//2+1]

        r1 = F.softplus(self._raw_r1)
        r2 = r1 + F.softplus(self._raw_r2gap)
        sharp = F.softplus(self._raw_sharp) + 1.0   # ensure > 1 for useful roll-off

        # Smooth transitions via sigmoids (differentiable everywhere).
        # pass-below-r1 : sigmoid(sharp * (r1 - rho))
        # pass-above-r  : sigmoid(sharp * (rho - r))
        below_r1 = torch.sigmoid(sharp * (r1 - rho))
        below_r2 = torch.sigmoid(sharp * (r2 - rho))
        m_low = below_r1
        m_mid = below_r2 - below_r1          # non-negative because r2 > r1
        m_high = 1.0 - below_r2

        # Stack as [n_bands, h, w//2+1].
        return torch.stack([m_low, m_mid, m_high], dim=0).clamp_min(0.0)
       
    def forward(self,
                sup_fts: torch.Tensor,
                sup_mask: torch.Tensor,
                qry_fts: torch.Tensor):

        if qry_fts.dim() == 3:
            qry_fts = qry_fts.unsqueeze(0)          # -> [1, C, h, w]

        b, c, h, w = sup_fts.shape
        assert b == 1 and c == self.dim

        # Downsample support mask to feature resolution -> [1, h, w]
        if sup_mask.dim() == 3 and sup_mask.shape[-2:] != (h, w):
            mask_hw = F.interpolate(sup_mask.unsqueeze(1).float(),
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=True).squeeze(1)
        else:
            mask_hw = sup_mask.float()
            if mask_hw.dim() == 2:
                mask_hw = mask_hw.unsqueeze(0)

        # 1) FFT of support and query.
        F_s = torch.fft.rfft2(sup_fts, norm='ortho')      # complex, [1,C,h,w//2+1]
        F_q = torch.fft.rfft2(qry_fts, norm='ortho')      # complex, [1,C,h,w//2+1]

        # 2) Build radial band masks.
        masks = self._make_radial_masks(h, w, sup_fts.device, sup_fts.dtype)  # [3,h,w//2+1]

        bands_s, bands_q, protos = [], [], []
        for i in range(self.n_bands):
            mi = masks[i].unsqueeze(0).unsqueeze(0)        # [1,1,h,w//2+1]
            bs = torch.fft.irfft2(F_s * mi, s=(h, w), norm='ortho')   # [1,C,h,w]
            bq = torch.fft.irfft2(F_q * mi, s=(h, w), norm='ortho')   # [1,C,h,w]

            bands_s.append(bs)
            bands_q.append(bq)
            protos.append(self._masked_avg(bs, mask_hw))              # [1, C]

        return bands_s, bands_q, protos


# =============================================================================
#  Module 2 -- Geodesic Matcher (Heat-Method-based, fully differentiable)
# =============================================================================
class GeodesicMatcher(nn.Module):
    def __init__(self,
                 dim: int = 512,
                 n_bands: int = 3,
                 k_steps: int = 5,
                 aff_sigma: float = 0.5,
                 scaler: float = 20.0,
                 seed_quantile: float = 0.85):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        self.k_steps = k_steps
        self.aff_sigma = aff_sigma
        self.scaler = scaler
        self.seed_quantile = seed_quantile
        self.alpha_geo = nn.Parameter(torch.zeros(n_bands))
        self.band_logits = nn.Parameter(torch.ones(n_bands))
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    @staticmethod
    def _neighbour_affinity(fmap: torch.Tensor, sigma: float):
        """
        Build 8-neighbour affinity of a feature map.
        fmap: [1, C, h, w]. Returns aff: [1, 8, h, w] and a mask of
        in-bounds neighbours (the 8 shifts centred on each pixel).
        Affinity = exp( - (1 - cos_sim) / sigma ) in [0, 1].
        """
        _, c, h, w = fmap.shape
        fmap_n = F.normalize(fmap, dim=1)

        # 3x3 shifts excluding the centre -> 8 neighbour patches.
        # Use padding=1 so shifted copies have identical spatial size.
        padded = F.pad(fmap_n, (1, 1, 1, 1), mode='replicate')

        shifts = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

        affs = []
        for dy, dx in shifts:
            sh = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            cos = (fmap_n * sh).sum(dim=1, keepdim=True)                 # [1,1,h,w]
            a = torch.exp(-(1.0 - cos) / max(sigma, 1e-3))               # [1,1,h,w]
            affs.append(a)
        aff = torch.cat(affs, dim=1)                                     # [1,8,h,w]

        # Out-of-bounds mask: shifts that leave the grid are killed.
        mask = torch.ones_like(aff)
        # Build a [1,8,h,w] boundary mask.
        y_idx = torch.arange(h, device=fmap.device).view(1, 1, h, 1)
        x_idx = torch.arange(w, device=fmap.device).view(1, 1, 1, w)
        for i, (dy, dx) in enumerate(shifts):
            valid_y = (y_idx + dy >= 0) & (y_idx + dy < h)
            valid_x = (x_idx + dx >= 0) & (x_idx + dx < w)
            mask[:, i:i + 1] = (valid_y & valid_x).float()

        aff = aff * mask
        return aff, shifts

    @staticmethod
    def _diffuse_once(u: torch.Tensor,
                      aff: torch.Tensor,
                      shifts) -> torch.Tensor:
        _, _, h, w = u.shape
        padded = F.pad(u, (1, 1, 1, 1), mode='replicate')

        num = u.clone()                     # self-contribution (aff_self = 1)
        den = torch.ones_like(u)
        for i, (dy, dx) in enumerate(shifts):
            neigh = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            a = aff[:, i:i + 1]             # [1,1,h,w]
            num = num + a * neigh
            den = den + a
        return num / den.clamp_min(1e-6)

    def forward(self,
                qry_ft_raw: torch.Tensor,        # [1, C, h, w] or [C, h, w]
                bands_q: list,                    # list of 3 tensors [1, C, h, w]
                protos: list) -> torch.Tensor:    # list of 3 tensors [1, C]
        if qry_ft_raw.dim() == 3:
            qry_ft_raw = qry_ft_raw.unsqueeze(0)
        _, C, h, w = qry_ft_raw.shape

        cos_maps = []
        for i in range(self.n_bands):
            bq = F.normalize(bands_q[i], dim=1)                       # [1,C,h,w]
            p  = F.normalize(protos[i],  dim=1)                       # [1,C]
            cos = (bq * p.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)  # [1,1,h,w]
            cos_maps.append(cos)

        # ----- Step 2. per-band geodesic refinement via Heat Method.
        geo_maps = []
        for i in range(self.n_bands):
            # Build affinity on the *band* feature map (so the geodesic
            # respects the anatomical scale captured by this band).
            aff, shifts = self._neighbour_affinity(bands_q[i],
                                                   sigma=self.aff_sigma)

            cos = cos_maps[i]                                          # [1,1,h,w]
            pivot = torch.quantile(cos.detach().reshape(-1),
                                   self.seed_quantile).to(cos.dtype)
            u0 = torch.sigmoid(self.scaler * (cos - pivot))            # [1,1,h,w]

            # K diffusion steps.
            u = u0
            for _ in range(self.k_steps):
                u = self._diffuse_once(u, aff, shifts)

            geo_maps.append(u)

        # ----- Step 3. fuse cosine + geodesic per band
        band_scores = []
        for i in range(self.n_bands):
            g = torch.sigmoid(self.alpha_geo[i])                       # scalar in (0,1)
            s = (1.0 - g) * cos_maps[i] + g * geo_maps[i]              # [1,1,h,w]
            band_scores.append(s)
        band_score_stack = torch.cat(band_scores, dim=1)               # [1,3,h,w]

        # ----- Step 4. build per-pixel prototype via weighted combination
        #               of the 3 band prototypes, weights = softmax over bands
        #               of (band_score * band_logit).
        band_logits = self.band_logits.view(1, self.n_bands, 1, 1)
        logits = band_score_stack * self.scaler * band_logits          # [1,3,h,w]
        weights = torch.softmax(logits, dim=1)                         # [1,3,h,w]

        protos_stack = torch.stack(protos, dim=1)                      # [1,3,C]
        # Broadcast & sum: [1,3,C,1,1] * [1,3,1,h,w] -> sum over band
        blended = (protos_stack.unsqueeze(-1).unsqueeze(-1)
                   * weights.unsqueeze(2)).sum(dim=1)                  # [1,C,h,w]

        blended = self.refine(blended)                                  # [1,C,h,w]

        # ----- Step 5. assemble Decoder-compatible tensor.
        matched = torch.cat([qry_ft_raw, blended, band_score_stack],
                            dim=1)                                      # [1, 2C+3, h, w]
        return matched, band_score_stack


