# Enhancing Image Colorization Using Conditional Generative Adversarial Network

> Grayscale to Vibrant Transformation  
> By **Rami Huu Nguyen**, **Lakshmi Pranathi Vutla**, and **Avanith Kanamarlapudi**

---

## About Our Project

Grayscale images are beautiful but limited in visual expression. Manually colorizing them is time-consuming and requires expert skills. Our project uses **deep learning** to automatically convert grayscale images to colorful versions using a Conditional GAN.

### Model Architecture

- **Encoder:** ResNet18  
- **Decoder:** Dynamic UNet  
- **Discriminator:** PatchGAN (70x70)  
- **Loss Functions:** L1 Loss (structure) + GAN Loss (realism)

## How It Works

Our pipeline is powered by:

- **Conditional GAN (cGAN)** for learning a mapping from grayscale to color
- **L1 loss** to ensure pixel accuracy
- **GAN loss** to improve realism
- **PatchGAN discriminator** to focus on local structure and textures


## Meet the Team

| Member | LinkedIn | GitHub | Email |
|--------|----------|--------|-------|
| **Avanith Kanamarlapudi** | [LinkedIn](https://www.linkedin.com/in/avanith-kanamarlapudi-8aa081204/) | [GitHub](https://github.com/Avanith12) | [Email](mailto:A.Kanamarlapudi001@umb.edu) |
| **Lakshmi Pranathi Vutla** | [LinkedIn](https://www.linkedin.com/in/lakshmi-pranathi-vutla30/) | [GitHub](https://github.com/Pranathivutla30) | [Email](mailto:L.Vutla001@umb.edu) |
| **Rami Huu Nguyen** | [LinkedIn](https://www.linkedin.com/in/raminguyen/) | [GitHub](https://github.com/raminguyen) | [Email](mailto:huuthanhvy.nguyen001@umb.edu) |

---

> 🌈 *“Old to Bold – Make History Colorful!”*
