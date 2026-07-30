"""Propagation-ON DIS inverse search: one Triton launch per cv2 raster anti-diagonal."""
import math

import numpy as np
import torch
import triton
import triton.language as tl

from .torch_dis_batched import EPS, TorchDISBatched


def _build_wavefront_schedule(hs, ws, nstripes=8):
    """Per pass, the anti-diagonal patch groups and each patch's two prior neighbors."""
    stripe_sz = int(math.ceil(hs / float(nstripes)))
    iss = np.arange(hs)
    stripe = iss // stripe_sz                       # (hs,)
    local = iss - stripe * stripe_sz                # local row within stripe
    stripe_top = stripe * stripe_sz                 # first global row of stripe
    # bottom valid global row of each patch's stripe (exclusive upper bound):
    stripe_bot = np.minimum((stripe + 1) * stripe_sz, hs)

    # broadcast to full grid
    IS = np.repeat(iss, ws)                          # (P,) global row per patch
    JS = np.tile(np.arange(ws), hs)                  # (P,) col per patch
    LOCAL = np.repeat(local, ws)
    S_TOP = np.repeat(stripe_top, ws)
    S_BOT = np.repeat(stripe_bot, ws)
    P = hs * ws
    p_index = np.arange(P)

    def group(keys):
        """Return list of index arrays grouped by ascending key value."""
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        bounds = np.where(np.diff(ks) != 0)[0] + 1
        return [order[s:e] for s, e in zip([0, *bounds.tolist()],
                                           [*bounds.tolist(), len(order)])]

    # ---- forward pass: key = local_row + js, neighbors LEFT (js-1), UP (is-1)
    key_fwd = LOCAL + JS
    left_ok = JS > 0
    up_ok = IS > S_TOP                               # strictly inside stripe
    left_idx = np.where(left_ok, p_index - 1, 0)
    up_idx = np.where(up_ok, p_index - ws, 0)
    fwd_groups = group(key_fwd)

    # ---- backward pass: key = (rows_below) + (cols_right); nbrs RIGHT/DOWN
    rows_below = (S_BOT - 1) - IS                    # 0 at stripe bottom
    cols_right = (ws - 1) - JS
    key_bwd = rows_below + cols_right
    right_ok = JS < (ws - 1)
    down_ok = IS < (S_BOT - 1)                       # strictly inside stripe
    right_idx = np.where(right_ok, p_index + 1, 0)
    down_idx = np.where(down_ok, p_index + ws, 0)
    bwd_groups = group(key_bwd)

    def pack(groups, nA_all, mA_all, nB_all, mB_all):
        return [dict(idx=g, nA=nA_all[g], mA=mA_all[g], nB=nB_all[g], mB=mB_all[g]) for g in groups]

    fwd = pack(fwd_groups, left_idx, left_ok, up_idx, up_ok)
    bwd = pack(bwd_groups, right_idx, right_ok, down_idx, down_ok)
    return {"fwd": fwd, "bwd": bwd}


# Mean-normalized-SSD sums (sd, sdsq, six, siy) of one patch at flow (cx, cy).
@triton.jit
def _patch_sums(cx, cy, Itl, Jtl, I1_ptr, I1base, We, roff, coff,
                I0p, I0xp, I0yp, bsz, i_low, i_high, j_low, j_high):
    i_I1 = tl.minimum(tl.maximum(Itl + cy + bsz, i_low), i_high)
    j_I1 = tl.minimum(tl.maximum(Jtl + cx + bsz, j_low), j_high)
    bi = tl.floor(i_I1)
    bj = tl.floor(j_I1)
    di = i_I1 - bi
    dj = j_I1 - bj
    bi_i = bi.to(tl.int64)
    bj_i = bj.to(tl.int64)
    w00 = (1.0 - di) * (1.0 - dj)
    w01 = (1.0 - di) * dj
    w10 = di * (1.0 - dj)
    w11 = di * dj
    base = I1base + bi_i * We + bj_i
    off = roff * We + coff
    A = tl.load(I1_ptr + base + off)
    Bt = tl.load(I1_ptr + base + off + 1)
    C = tl.load(I1_ptr + base + off + We)
    D = tl.load(I1_ptr + base + off + We + 1)
    I1p = w00 * A + w01 * Bt + w10 * C + w11 * D
    diff = I1p - I0p
    sd = tl.sum(diff)
    sdsq = tl.sum(diff * diff)
    six = tl.sum(diff * I0xp)
    siy = tl.sum(diff * I0yp)
    return sd, sdsq, six, siy


@triton.jit
def _diag_kernel(
    Sx_ptr, Sy_ptr,                       # (B,P) flow, in/out (in-place across diagonals)
    idx_ptr, nA_ptr, mA_ptr, nB_ptr, mB_ptr,   # (K,) int32 schedule for this diagonal
    Itl_ptr, Jtl_ptr,                     # (P,) float patch top-left
    I0_ptr, I0x_ptr, I0y_ptr,             # (B,h*w) float
    I1_ptr,                               # (B,He*We) float (bordered)
    iH11_ptr, iH12_ptr, iH22_ptr, xgs_ptr, ygs_ptr,   # (B,P) float
    P, hw, HeWe, w, We,
    bsz, n, ps_f, i_low, i_high, j_low, j_high,
    num_inner: tl.constexpr, BLOCK: tl.constexpr,
):
    pk = tl.program_id(0)
    pb = tl.program_id(1)
    p = tl.load(idx_ptr + pk)
    p64 = p.to(tl.int64)
    pb64 = pb.to(tl.int64)
    bp = pb64 * P + p64

    Itl = tl.load(Itl_ptr + p)
    Jtl = tl.load(Jtl_ptr + p)

    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    roff = rows[:, None].to(tl.int64)
    coff = cols[None, :].to(tl.int64)

    Itl_i = Itl.to(tl.int64)
    Jtl_i = Jtl.to(tl.int64)
    I0base = pb64 * hw + Itl_i * w + Jtl_i
    poff = roff * w + coff
    I0p = tl.load(I0_ptr + I0base + poff)
    I0xp = tl.load(I0x_ptr + I0base + poff)
    I0yp = tl.load(I0y_ptr + I0base + poff)

    iH11 = tl.load(iH11_ptr + bp)
    iH12 = tl.load(iH12_ptr + bp)
    iH22 = tl.load(iH22_ptr + bp)
    xg = tl.load(xgs_ptr + bp)
    yg = tl.load(ygs_ptr + bp)

    I1base = pb64 * HeWe

    # ---- candidate c0 = self seed
    bx = tl.load(Sx_ptr + bp)
    by = tl.load(Sy_ptr + bp)
    sd, sdsq, six, siy = _patch_sums(bx, by, Itl, Jtl, I1_ptr, I1base, We,
                                     roff, coff, I0p, I0xp, I0yp,
                                     bsz, i_low, i_high, j_low, j_high)
    best = sdsq - sd * sd / n

    # ---- candidate A (left on fwd / right on bwd)
    nA = tl.load(nA_ptr + pk)
    mA = tl.load(mA_ptr + pk)
    bpA = pb64 * P + nA.to(tl.int64)
    ax = tl.load(Sx_ptr + bpA)
    ay = tl.load(Sy_ptr + bpA)
    sd, sdsq, six, siy = _patch_sums(ax, ay, Itl, Jtl, I1_ptr, I1base, We,
                                     roff, coff, I0p, I0xp, I0yp,
                                     bsz, i_low, i_high, j_low, j_high)
    assd = sdsq - sd * sd / n
    takeA = (mA != 0) & (assd < best)
    bx = tl.where(takeA, ax, bx)
    by = tl.where(takeA, ay, by)
    best = tl.where(takeA, assd, best)

    # ---- candidate B (up on fwd / down on bwd)
    nB = tl.load(nB_ptr + pk)
    mB = tl.load(mB_ptr + pk)
    bpB = pb64 * P + nB.to(tl.int64)
    cx0 = tl.load(Sx_ptr + bpB)
    cy0 = tl.load(Sy_ptr + bpB)
    sd, sdsq, six, siy = _patch_sums(cx0, cy0, Itl, Jtl, I1_ptr, I1base, We,
                                     roff, coff, I0p, I0xp, I0yp,
                                     bsz, i_low, i_high, j_low, j_high)
    cssd = sdsq - sd * sd / n
    takeB = (mB != 0) & (cssd < best)
    bx = tl.where(takeB, cx0, bx)
    by = tl.where(takeB, cy0, by)

    # ---- gradient descent from best candidate
    cx = bx
    cy = by
    prev = 1e10
    active = 1.0
    for _i in tl.static_range(num_inner):
        sd, sdsq, six, siy = _patch_sums(cx, cy, Itl, Jtl, I1_ptr, I1base, We,
                                         roff, coff, I0p, I0xp, I0yp,
                                         bsz, i_low, i_high, j_low, j_high)
        dUx = six - sd * xg / n
        dUy = siy - sd * yg / n
        SSD = sdsq - sd * sd / n
        dx = iH11 * dUx + iH12 * dUy
        dy = iH12 * dUx + iH22 * dUy
        act = active > 0.5
        cx = tl.where(act, cx - dx, cx)
        cy = tl.where(act, cy - dy, cy)
        broke = act & (SSD >= prev)
        active = tl.where(broke, 0.0, active)
        prev = tl.where(active > 0.5, SSD, prev)

    moved = tl.sqrt((cx - bx) * (cx - bx) + (cy - by) * (cy - by))
    acc = moved <= ps_f
    ox = tl.where(acc, cx, bx)
    oy = tl.where(acc, cy, by)
    tl.store(Sx_ptr + bp, ox)
    tl.store(Sy_ptr + bp, oy)


class TorchDISBatchedPropFast(TorchDISBatched):
    """Spatial propagation in cv2's wavefront order, one Triton launch per diagonal."""

    def __init__(self, **kw):
        super().__init__(**kw)
        # A forward and a backward pass split gd_iter between them.
        self.num_iter = 2
        self._fsched_cache = {}

    def _fschedule(self, hs, ws):
        key = (hs, ws)
        sch = self._fsched_cache.get(key)
        if sch is not None:
            return sch
        raw = _build_wavefront_schedule(hs, ws, nstripes=8)
        out = {pas: [{name: torch.as_tensor(a, dtype=torch.int32, device=self.device) for name, a in g.items()} for g in raw[pas]] for pas in ("fwd", "bwd")}
        self._fsched_cache[key] = out
        return out

    def _inverse_search(self, I0, I1ext, I0x, I0y, Ux, Uy):
        ps, pstr, bsz = self.ps, self.pstr, self.bsz
        dev = self.device
        B = I0.shape[0]
        h, w = I0.shape[-2:]
        He, We = I1ext.shape[-2:]
        hs = 1 + (h - ps) // pstr
        ws = 1 + (w - ps) // pstr
        psz2 = ps // 2
        P = hs * ws
        n = float(ps * ps)

        iis = torch.arange(hs, device=dev) * pstr
        jjs = torch.arange(ws, device=dev) * pstr
        I_tl = iis.view(hs, 1).expand(hs, ws).reshape(P).float().contiguous()
        J_tl = jjs.view(1, ws).expand(hs, ws).reshape(P).float().contiguous()

        Sxx, Syy, Sxy, Sxg, Syg = self._structure_tensor(I0x, I0y, hs, ws)
        Sxx = Sxx.reshape(B, P); Syy = Syy.reshape(B, P); Sxy = Sxy.reshape(B, P)
        xgs = Sxg.reshape(B, P).contiguous()
        ygs = Syg.reshape(B, P).contiguous()
        detH = Sxx * Syy - Sxy * Sxy
        detH = torch.where(detH.abs() < EPS, torch.full_like(detH, EPS), detH)
        invH11 = (Syy / detH).contiguous()
        invH12 = (-Sxy / detH).contiguous()
        invH22 = (Sxx / detH).contiguous()

        cen_i = (iis + psz2).view(hs, 1).expand(hs, ws).reshape(P)
        cen_j = (jjs + psz2).view(1, ws).expand(hs, ws).reshape(P)
        Sx = Ux[:, 0][:, cen_i, cen_j].clone().contiguous()      # (B,P)
        Sy = Uy[:, 0][:, cen_i, cen_j].clone().contiguous()

        I0flat = I0.reshape(B, h * w).contiguous()
        I0xflat = I0x.reshape(B, h * w).contiguous()
        I0yflat = I0y.reshape(B, h * w).contiguous()
        I1flat = I1ext.reshape(B, He * We).contiguous()

        i_low = float(bsz - ps + 1)
        i_high = float(bsz + h - 1)
        j_low = float(bsz - ps + 1)
        j_high = float(bsz + w - 1)
        num_inner = int(math.floor(self.gd_iter / float(self.num_iter)))

        sched = self._fschedule(hs, ws)
        hw = h * w
        HeWe = He * We
        bsz_f = float(bsz)
        ps_f = float(ps)

        def run_pass(steps):
            for g in steps:
                K = g["idx"].shape[0]
                _diag_kernel[(K, B)](
                    Sx, Sy,
                    g["idx"], g["nA"], g["mA"], g["nB"], g["mB"],
                    I_tl, J_tl,
                    I0flat, I0xflat, I0yflat, I1flat,
                    invH11, invH12, invH22, xgs, ygs,
                    P, hw, HeWe, w, We,
                    bsz_f, n, ps_f, i_low, i_high, j_low, j_high,
                    num_inner=num_inner, BLOCK=ps,
                    num_warps=2,
                )

        run_pass(sched["fwd"])
        run_pass(sched["bwd"])
        return Sx.view(B, hs, ws), Sy.view(B, hs, ws)
