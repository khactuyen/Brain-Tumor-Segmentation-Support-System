# -*- coding: utf-8 -*-
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"D:\SIC_Capstone 2026\report_work\generated")
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = r"C:\Windows\Fonts\cambria.ttc"


def math_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size, index=1)
    except OSError:
        return ImageFont.truetype(FONT_PATH, size)


def render(name, lines, size=64):
    font = math_font(size)
    probe = Image.new("RGB", (20, 20), "white")
    draw = ImageDraw.Draw(probe)
    widths, heights = [], []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    gap = 30
    width = max(widths) + 180
    height = sum(heights) + gap * (len(lines) - 1) + 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 60
    for line, w, h in zip(lines, widths, heights):
        draw.text(((width - w) / 2, y), line, font=font, fill="#171717")
        y += h + gap
    image.save(OUT / name, dpi=(300, 300))
    print(name, image.size)


render("eq_conv3d.png", [
    "y_cₒ(i,j,k) = b_cₒ + Σ_cᵢ Σ_u Σ_v Σ_w  W_cₒ,cᵢ(u,v,w) · x_cᵢ(i+u,j+v,k+w)"
], 58)

render("eq_residual_norm.png", [
    "y = F(x; W) + x",
    "IN(x_n,c) = γ_c · (x_n,c − μ_n,c) / √(σ²_n,c + ε) + β_c",
], 62)

render("eq_dicece.png", [
    "L_Dice = 1 − (1/C) Σ_c  [ (2 Σ_i p_i,c g_i,c + ε) / (Σ_i p²_i,c + Σ_i g²_i,c + ε) ]",
    "L_CE = − (1/N) Σ_i Σ_c  g_i,c log(p_i,c + ε)",
    "L_DiceCE = λ_D · L_Dice + λ_CE · L_CE",
], 54)

render("eq_adamw.png", [
    "m_t = β₁m_t₋₁ + (1−β₁)g_t          v_t = β₂v_t₋₁ + (1−β₂)g²_t",
    "m̂_t = m_t / (1−β₁ᵗ)                 v̂_t = v_t / (1−β₂ᵗ)",
    "θ_t₊₁ = (1−η_tλ)θ_t − η_t · m̂_t / (√v̂_t + ε)",
], 58)

render("eq_cosine.png", [
    "η_t = η_min + ½(η_max − η_min) · [1 + cos(πt / T_max)]"
], 66)

render("eq_attention.png", [
    "Q = XW_Q          K = XW_K          V = XW_V",
    "Attention(Q,K,V) = Softmax(QKᵀ / √d_k + B) · V",
], 64)

render("eq_swin_block.png", [
    "ẑˡ = W-MSA(LN(zˡ⁻¹)) + zˡ⁻¹          zˡ = MLP(LN(ẑˡ)) + ẑˡ",
    "ẑˡ⁺¹ = SW-MSA(LN(zˡ)) + zˡ          zˡ⁺¹ = MLP(LN(ẑˡ⁺¹)) + ẑˡ⁺¹",
], 55)

render("eq_sliding_window.png", [
    "p̂_c(v) = [Σ_r∈R(v) w_r(v)p_r,c(v)] / [Σ_r∈R(v) w_r(v) + ε]",
    "ŷ(v) = argmax_c p̂_c(v)",
], 62)

render("eq_metrics.png", [
    "Dice(P,G) = 2|P∩G| / (|P|+|G|)          IoU(P,G) = |P∩G| / |P∪G|",
    "HD₉₅(P,G) = max{ Q₀.₉₅[d(∂P,∂G)], Q₀.₉₅[d(∂G,∂P)] }",
], 60)

render("eq_backprop.png", [
    "g_t = ∇_θ L = (∂L/∂z_L) · Π_ℓ₌L→1 (∂z_ℓ/∂z_ℓ₋₁) · (∂z₀/∂θ)"
], 60)
