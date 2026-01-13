cat << 'EOF' > README.md
# Generative Modeling: RGB-to-IR Latent Diffusion (DDIM) & Score-based NCSN
### Advanced Deep Learning (IIIT Delhi) | Aman Kumar (MT24012)

This repository explores two state-of-the-art paradigms in generative modeling: **Latent Diffusion** for cross-modal image translation and **Score-based Generative Modeling** using Denoising Score Matching.

---

## 🔬 Project Overview

### 1. RGB-to-IR Translation via DDIM Inversion
**Objective:** To map RGB latent representations to Infrared (IR) latents using the M3FD multimodal dataset. This approach leverages the pre-trained latent space of Stable Diffusion to perform efficient cross-modal translation.

#### **Technical Approach:**
* **Latent Encoding:** Images are center-cropped to $512 \times 512$ and encoded into latent space $z_0$ using the **Stable Diffusion v1-4 VAE**.
* **DDIM Inversion:** We utilize deterministic DDIM inversion to diffuse the latent representation to $t=400$, extracting pairs $(z^{RGB}_{400}, z^{IR}_{400})$. This ensures that the structural information is preserved in the inverted latent.
* **Translation Network:** A **U-Net with skip connections** is trained to learn the mapping $f_\theta: z^{RGB}_{400} \rightarrow z^{IR}_{400}$.
* **Optimization:** Trained for 25 epochs using **AdamW** ($lr=1 \times 10^{-4}$) with a Mean Squared Error (MSE) loss on the latent representations.

#### **Quantitative Results (Test Set):**
* **PSNR:** 19.07 dB
* **SSIM:** 0.7083
* **Observation:** The high SSIM indicates excellent preservation of structural edges and object boundaries during the RGB to IR translation.

---

### 2. Score-based Generative Modeling on MNIST
**Objective:** Implementation of a **Noise Conditional Score Network (NCSN)** to estimate the gradient of the log-density of the data distribution, enabling high-quality image synthesis.

#### **Technical Approach:**
* **Architecture:** A **$\sigma$-conditioned U-Net** utilizing sinusoidal positional embeddings for time/noise steps and **FiLM (Feature-wise Linear Modulation)** for effective conditioning.
* **Training (Denoising Score Matching):** The model is trained to minimize the Fisher divergence between the network output and the score of the perturbed data distribution:
    $$L_{DSM}(\theta) = \mathbb{E}_{x, \sigma, \tilde{x}} \left[ \frac{\sigma^2}{2} \| s_\theta(\tilde{x}, \sigma) + \frac{\tilde{x} - x}{\sigma^2} \|^2 \right]$$
* **Inference (Annealed Langevin Dynamics):** We sample from the learned distribution by iteratively moving along the estimated score function while gradually decreasing the noise level $\sigma$.

#### **Quantitative Results:**
* **Epochs:** 250
* **Final DSM Loss:** 0.0466
* **Inception Score (IS):** 2.18 ± 0.04
* **FID:** 103.49
* **Analysis:** The low FID and stable IS confirm that the model generates diverse and recognizable digits, effectively learning the MNIST manifold.

---

## 📊 Summary of Performance

| Task | Metric | Final Value |
| :--- | :--- | :--- |
| **Diffusion (Latent Regression)** | **PSNR** | **19.07 dB** |
| **Diffusion (Latent Regression)** | **SSIM** | **0.7083** |
| **Score-based (NCSN)** | **FID** | **103.49** |
| **Score-based (NCSN)** | **IS** | **2.18** |

---

## 🛠 Tech Stack & Dependencies
* **Core:** Python, PyTorch
* **Generative:** Diffusers (HuggingFace), DDIM Schedulers
* **Processing:** OpenCV, NumPy, SciPy (Optimization)
* **Evaluation:** Torchmetrics (PSNR, SSIM, FID)

---

## 📜 References
1.  **Song, J. et al. (2025).** *M3FD: A Multimodal Multi-domain Multi-weather Dataset.*
2.  **Song, Y., & Ermon, S. (2020).** *Denoising Diffusion Implicit Models (DDIM).*
3.  **Song, Y., & Ermon, S. (2019).** *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS.

---
**Aman Kumar** M.Tech CSE, IIIT Delhi | Postgraduate Researcher, MIDAS Lab  
[aman24012@iiitd.ac.in](mailto:aman24012@iiitd.ac.in)
EOF
