"use strict";

import { createRequire } from "module";
import path from "path";

const require = createRequire(import.meta.url);
const modules = String.raw`C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules`;
const { mathjax } = require(path.join(modules, "mathjax-full/js/mathjax.js"));
const { TeX } = require(path.join(modules, "mathjax-full/js/input/tex.js"));
const { SVG } = require(path.join(modules, "mathjax-full/js/output/svg.js"));
const { liteAdaptor } = require(path.join(modules, "mathjax-full/js/adaptors/liteAdaptor.js"));
const { RegisterHTMLHandler } = require(path.join(modules, "mathjax-full/js/handlers/html.js"));
const { AllPackages } = require(path.join(modules, "mathjax-full/js/input/tex/AllPackages.js"));
const sharp = require(path.join(modules, "sharp"));

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svgOut = new SVG({ fontCache: "local" });
const document = mathjax.document("", { InputJax: tex, OutputJax: svgOut });

const outputDir = String.raw`D:\SIC_Capstone 2026\report_work\generated`;

function latexToSvg(latex) {
  const html = adaptor.outerHTML(document.convert(latex, { display: true }));
  const start = html.indexOf("<svg");
  const end = html.indexOf("</svg>");
  let svg = html.slice(start, end + 6).replace(/<\?xml[^>]*>/g, "");
  if (!/xmlns="http:\/\/www\.w3\.org\/2000\/svg"/.test(svg)) {
    svg = svg.replace(/<svg /, '<svg xmlns="http://www.w3.org/2000/svg" ');
  }
  return svg.replace(/currentColor/g, "#171717");
}

const equations = {
  "eq_conv3d.png": String.raw`
    y_{c_o}(i,j,k)=b_{c_o}+\sum_{c_i=1}^{C_{in}}\sum_{u=0}^{K_d-1}\sum_{v=0}^{K_h-1}\sum_{w=0}^{K_w-1}
    W_{c_o,c_i}(u,v,w)\,x_{c_i}(i+u,j+v,k+w)`,
  "eq_residual_norm.png": String.raw`
    \begin{aligned}
    \mathbf{y}&=\mathcal{F}(\mathbf{x};\mathbf{W})+\mathbf{x},\\
    \operatorname{IN}(x_{n,c})&=\gamma_c\frac{x_{n,c}-\mu_{n,c}}{\sqrt{\sigma_{n,c}^{2}+\varepsilon}}+\beta_c
    \end{aligned}`,
  "eq_dicece.png": String.raw`
    \begin{aligned}
    \mathcal{L}_{Dice}&=1-\frac{1}{C}\sum_{c=1}^{C}\frac{2\sum_i p_{i,c}g_{i,c}+\epsilon}{\sum_i p_{i,c}^{2}+\sum_i g_{i,c}^{2}+\epsilon},\\
    \mathcal{L}_{CE}&=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}g_{i,c}\log(p_{i,c}+\epsilon),\\
    \mathcal{L}_{DiceCE}&=\lambda_D\mathcal{L}_{Dice}+\lambda_{CE}\mathcal{L}_{CE}
    \end{aligned}`,
  "eq_adamw.png": String.raw`
    \begin{aligned}
    \mathbf{m}_t&=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t,\qquad
    \mathbf{v}_t=\beta_2\mathbf{v}_{t-1}+(1-\beta_2)\mathbf{g}_t^2,\\
    \widehat{\mathbf{m}}_t&=\frac{\mathbf{m}_t}{1-\beta_1^t},\qquad
    \widehat{\mathbf{v}}_t=\frac{\mathbf{v}_t}{1-\beta_2^t},\\
    \boldsymbol{\theta}_{t+1}&=(1-\eta_t\lambda)\boldsymbol{\theta}_t-
    \eta_t\frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t}+\epsilon}
    \end{aligned}`,
  "eq_cosine.png": String.raw`
    \eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
    \left(1+\cos\left(\frac{\pi t}{T_{max}}\right)\right)`,
  "eq_attention.png": String.raw`
    \begin{aligned}
    Q&=XW_Q,\quad K=XW_K,\quad V=XW_V,\\
    \operatorname{Attention}(Q,K,V)&=\operatorname{Softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+B\right)V
    \end{aligned}`,
  "eq_swin_block.png": String.raw`
    \begin{aligned}
    \hat z^{\ell}&=\operatorname{W\!\!\!-MSA}(\operatorname{LN}(z^{\ell-1}))+z^{\ell-1},\\
    z^{\ell}&=\operatorname{MLP}(\operatorname{LN}(\hat z^{\ell}))+\hat z^{\ell},\\
    \hat z^{\ell+1}&=\operatorname{SW\!\!\!-MSA}(\operatorname{LN}(z^{\ell}))+z^{\ell}
    \end{aligned}`,
  "eq_sliding_window.png": String.raw`
    \hat p_c(v)=\frac{\sum_{r\in\mathcal{R}(v)}w_r(v)\,p_{r,c}(v)}
    {\sum_{r\in\mathcal{R}(v)}w_r(v)+\epsilon},\qquad
    \hat y(v)=\arg\max_c\hat p_c(v)`,
  "eq_metrics.png": String.raw`
    \begin{aligned}
    Dice(P,G)&=\frac{2|P\cap G|}{|P|+|G|},\qquad
    IoU(P,G)=\frac{|P\cap G|}{|P\cup G|},\\
    HD_{95}(P,G)&=\max\left\{Q_{0.95}[d(\partial P,\partial G)],
    Q_{0.95}[d(\partial G,\partial P)]\right\}
    \end{aligned}`,
  "eq_backprop.png": String.raw`
    \mathbf{g}_t=\nabla_{\boldsymbol{\theta}}\mathcal{L}
    =\frac{\partial\mathcal{L}}{\partial\mathbf{z}_L}
    \prod_{\ell=L}^{1}\frac{\partial\mathbf{z}_{\ell}}{\partial\mathbf{z}_{\ell-1}}
    \frac{\partial\mathbf{z}_0}{\partial\boldsymbol{\theta}}`,
};

for (const [name, latex] of Object.entries(equations)) {
  const svg = Buffer.from(latexToSvg(latex));
  await sharp(svg, { density: 300 })
    .flatten({ background: "#FFFFFF" })
    .trim({ background: "#FFFFFF" })
    .extend({ top: 35, bottom: 35, left: 55, right: 55, background: "#FFFFFF" })
    .png()
    .toFile(path.join(outputDir, name));
  console.log(name);
}
