import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.Decoders import Decoder
from boundary_loss import BoundaryLoss

from models.moudles import SpectralPrototypeBank, GeodesicMatcher


class FewShotSeg(nn.Module):

    _DATASET_HPARAMS = {
        'Abd-CT':  dict(n_bands=3, k_steps=5, aff_sigma=0.5,
                        seed_quantile=0.85, my_weight=(0.08, 1.0)),
        'Abd-MRI': dict(n_bands=3, k_steps=5, aff_sigma=0.5,
                        seed_quantile=0.85, my_weight=(0.10, 1.0)),
        'CMR':     dict(n_bands=3, k_steps=4, aff_sigma=0.4,
                        seed_quantile=0.90, my_weight=(0.04, 1.0)),

    }

    def __init__(self,
                 pretrained_weights: str = "deeplabv3",
                 dataset: str = None):
        super().__init__()

        # Resolve hyper-parameters (dataset-aware, with CT defaults).
        hp = self._DATASET_HPARAMS.get(dataset,
                                       self._DATASET_HPARAMS['Abd-CT'])

        self.encoder = Res101Encoder(
            replace_stride_with_dilation=[True, True, False],
            pretrained_weights=pretrained_weights,
        )
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.my_weight = torch.FloatTensor(list(hp['my_weight'])).cuda()
        self.criterion = nn.NLLLoss(ignore_index=255, weight=self.my_weight)
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)

        self.spectral = SpectralPrototypeBank(dim=512,
                                              n_bands=hp['n_bands'])
        self.geo = GeodesicMatcher(dim=512,
                                   n_bands=hp['n_bands'],
                                   k_steps=hp['k_steps'],
                                   aff_sigma=hp['aff_sigma'],
                                   scaler=self.scaler,
                                   seed_quantile=hp['seed_quantile'])
        # =================================================================

        self.decoder1 = Decoder()      # foreground branch
        self.decoder2 = Decoder()      # background branch

    # ===================================================================== #
    #                           FORWARD                                     #
    # ===================================================================== #
    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs : way x shot x [B x 3 x H x W]
            supp_mask : way x shot x [B x H x W]
            qry_imgs  : N        x [B x 3 x H x W]
        Returns:
            output      : [B, 1+n_ways, H, W]  (softmax probs, bg then fg)
            align_loss  : scalar tensor
            b_loss      : scalar tensor
        """
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack(
            [torch.stack(way, dim=0) for way in supp_mask], dim=0
        ).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        # -------- encoder --------
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [torch.cat(qry_imgs, dim=0)], dim=0,
        )
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [
            img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
                supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]
            )
            for _, dic in enumerate(img_fts)
        ]
        qry_fts = [
            img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
                qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]
            )
            for _, dic in enumerate(img_fts)
        ]

        self.thresh_pred = tao  # kept for API compatibility; unused

        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        outputs = []

        for epi in range(supp_bs):
            sup_fts_epi = supp_fts[0][[epi], 0, 0]        # [1, C, h, w]
            qry_fts_epi = qry_fts[0][epi, 0]              # [C, h, w]
            sup_mask_fg = supp_mask[[epi], 0, 0]          # [1, H, W] in {0,1}
            sup_mask_bg = 1.0 - sup_mask_fg

            # ---- Foreground branch: Spectral + Geodesic ----
            bands_s_fg, bands_q_fg, protos_fg = self.spectral(
                sup_fts_epi, sup_mask_fg, qry_fts_epi)
            fg_matched, _ = self.geo(qry_fts_epi, bands_q_fg, protos_fg)
            fg_preds = self.decoder1(fg_matched)          # [1, 1, h, w]

            # ---- Background branch: same modules, inverted mask ----
            bands_s_bg, bands_q_bg, protos_bg = self.spectral(
                sup_fts_epi, sup_mask_bg, qry_fts_epi)
            bg_matched, _ = self.geo(qry_fts_epi, bands_q_bg, protos_bg)
            bg_preds = self.decoder2(bg_matched)          # [1, 1, h, w]

            fg_preds = F.interpolate(fg_preds, size=img_size,
                                     mode='bilinear', align_corners=True)
            bg_preds = F.interpolate(bg_preds, size=img_size,
                                     mode='bilinear', align_corners=True)

            preds = torch.cat([bg_preds, fg_preds], dim=1)
            preds = torch.softmax(preds, dim=1)
            outputs.append(preds)

            if train:
                align_loss_epi, b_loss_epi = self.alignLoss(
                    [supp_fts[n][epi] for n in range(len(supp_fts))],
                    [qry_fts[n][epi] for n in range(len(qry_fts))],
                    preds, supp_mask[epi],
                )
                align_loss += align_loss_epi
                b_loss += b_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / supp_bs, b_loss / supp_bs

    # ===================================================================== #
    #                         ALIGNMENT LOSS                                #
    # ===================================================================== #
    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
        Role-swapped auxiliary: the predicted query mask is used as a
        pseudo-support; the original support features are segmented
        against it and supervised by the real support mask.

        Design note (stable cold-start + no-crash version):
        ---------------------------------------------------
        Previously we switched between a "warm" branch and a "bg-only"
        branch based on whether the pseudo-fg had >= 4 pixels.  That
        caused the loss magnitude to jump whenever training crossed the
        threshold back and forth.  We now use a SINGLE unified branch:

          * The fg branch ALWAYS runs.  If the model's pred contains no
            fg pixels at all (extreme cold start), we fall back to a
            uniform pseudo-fg mask so the masked-average-pooling stage
            produces a global-mean prototype rather than collapsing.
          * The bg branch ALWAYS runs.
          * Supervision uses supp_label unchanged (no ignore-index swaps
            that would crash BoundaryLoss).

        Result: align_loss and align_b_loss are both non-zero and stay
        in a consistent magnitude from iteration 0, matching DIFD's
        original training dynamics.
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)    # [1, H, W]
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        pred_mask = torch.stack(binary_masks, dim=0).float()       # [(1+Wa), 1, H, W]

        loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)

        for way in range(n_ways):
            for shot in range(n_shots):
                # Swap roles: query acts as support, support acts as query.
                q_as_sup_fts = qry_fts[0]                     # [1, C, h, w]
                s_as_qry_fts = supp_fts[0][way, [shot]]       # [1, C, h, w]

                pseudo_fg = pred_mask[way + 1].float()        # [1, H, W]
                # Extreme cold-start safety: if the pred contains ZERO
                # fg pixels, fall back to a uniform pseudo-fg so the
                # masked average pool inside SpectralPrototypeBank does
                # not degenerate.  This gives the fg branch a weak but
                # non-pathological signal instead of a hard skip.
                if pseudo_fg.sum().item() < 1:
                    pseudo_fg = torch.ones_like(pseudo_fg)
                pseudo_bg = 1.0 - pred_mask[way + 1].float()
                if pseudo_bg.sum().item() < 1:
                    pseudo_bg = torch.ones_like(pseudo_bg)

                # ===== fg branch (always runs) =====
                bs_fg, bq_fg, p_fg = self.spectral(
                    q_as_sup_fts, pseudo_fg, s_as_qry_fts)
                fg_matched, _ = self.geo(s_as_qry_fts, bq_fg, p_fg)
                fg_preds_raw = self.decoder1(fg_matched)

                # ===== bg branch (always runs) =====
                bs_bg, bq_bg, p_bg = self.spectral(
                    q_as_sup_fts, pseudo_bg, s_as_qry_fts)
                bg_matched, _ = self.geo(s_as_qry_fts, bq_bg, p_bg)
                bg_preds_raw = self.decoder2(bg_matched)

                fg_preds = F.interpolate(fg_preds_raw,
                                         size=fore_mask.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)
                bg_preds = F.interpolate(bg_preds_raw,
                                         size=fore_mask.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)

                pred_ups = torch.softmax(
                    torch.cat([bg_preds, fg_preds], dim=1), dim=1)

                # Supervision target = real support mask.
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss   += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways
                b_loss += self.criterion_b(pred_ups, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, b_loss

        return loss, b_loss

    # ===================================================================== #
    #               LEGACY HELPERS (kept for external API)                  #
    # ===================================================================== #
    def get_masked_fts(self, fts, mask):
        masked = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                 / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        return masked

    def getPrototype(self, fg_fts):
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [
            torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
            for way in fg_fts
        ]
        return fg_prototypes

