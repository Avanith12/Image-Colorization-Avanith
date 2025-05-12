# <h1 align="center"><span>  🟦 Enhancing Image Colorization Using Conditional GANs</span>

<p align="center"><em>Bring grayscale images to life with deep learning!</em></p>


## <h2 align="center"><span> 🟦 What’s This Project About?</span>

Our project transforms black-and-white images into vibrant, colorized versions using a **Conditional Generative Adversarial Network (cGAN)**. We combine a **ResNet18 encoder**, **Dynamic UNet decoder**, and **PatchGAN discriminator** to learn realistic mappings from grayscale to color. Our work automates a process that once required manual skill — making it instant, scalable, and accessible.

<p align="center">
  <a href="Image_Colorization_Avanith_Rami_Pranathi.pdf" target="_blank">View our paper</a> |
  <a href="https://docs.google.com/presentation/d/1ugwfzaby_SkdIb8dxpUI79GYZOD1xhxJ/edit?usp=sharing&ouid=117579044537130000857&rtpof=true&sd=true" target="_blank">Presentation</a>
</p>



## <h2 align="center"><span> 🟦 Problem & Solution</span>

### Problem

Grayscale images are emotionally powerful, but lack visual richness. Manual recoloring is slow, costly, and inconsistent.

### Solution

We train a deep learning model to **learn color patterns from real-world images** and apply them to black-and-white inputs — offering fast, accurate, and beautiful colorization.

## <h2 align="center"> 🟦 Real-World Applications</span>

- Revive old family photos  
- Recolor historic archives  
- Help artists and content creators  
- Restore vintage films  
- Create colorful educational materials

---

## <h2 align="center"><span> 🟦 Model Architecture</span>

    - Encoder: ResNet18 (pretrained)
    - Decoder: Dynamic UNet
    - Discriminator: 70x70 PatchGAN
    - Loss: L1 Loss + GAN Loss
    - Image Size: 256x256

## <h2 align="center"><span> 🟦 Training Information</span>

| **Setting**        | **Details**                        |
|--------------------|------------------------------------|
| Dataset            | Subset of ImageNet (~10k images)   |
| Libraries          | PyTorch, FastAI, OpenCV, NumPy     |
| Training Time      | ~10 hours                          |
| Optimizer          | Adam (β₁ = 0.5, β₂ = 0.999)         |
| Evaluation Metrics | MSE, SSIM, Visual Comparison       |

---

<p align="center">
  <img src="top10prediction.png" width="1000" />
  <br><em>Top 10 good predictions based on Peak Signal-to-Noise Ratio Metric </em>
</p>

<p align="center">
  <img src="top10worseprediction.png" width="1000" />
  <br><em>Top 10 worse predictions based on Peak Signal-to-Noise Ratio Metric </em>
</p>


## <h2 align="center"><span> 🟦 Future Improvements</span>

- Real-time demo  
- Mobile + web support  
- Adjustable color warmth/vibrancy  
- Add Vision Transformers + Diffusion models  
- Tune for specific photo types (portraits, landscapes)

## <h2 align="center"><span> 🟦 Meet the Team</span>

<div align="center">

<table>
<td align="center">
  <a href="https://www.linkedin.com/in/avanith-kanamarlapudi-8aa081204/">
    <img src="teamimages/Avanith.png" width="100" height="100" style="border-radius: 50%;"/><br>
    <strong>Avanith Kanamarlapudi</strong>
  </a>
</td>
<td align="center">
  <a href="https://www.linkedin.com/in/raminguyen/">
    <img src="teamimages/ramihuunguyen.png" width="100" height="100" style="border-radius: 50%;"/><br>
    <strong>Rami Huu Nguyen</strong>
  </a>
</td>
<td align="center">
  <a href="https://www.linkedin.com/in/lakshmi-pranathi-vutla30/">
    <img src="static/Pranathi.jpg" width="100" height="100" style="border-radius: 50%;"/><br>
    <strong>Lakshmi Pranathi Vutla</strong>
  </a>
</td>

</table>

</div>

## <h2 align="center"><span> Acknowledgements</span>

<p align="center">

#UMassBoston #ImageColorization #DeepLearning #CGAN #Rami #Avanith #Pranathi

</p>
