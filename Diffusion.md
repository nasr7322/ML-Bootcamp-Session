# Session 08: Diffusion Models for Image Generation

### Introduction: Revisiting Transformers and Embeddings

- Quick recap on **Transformers**, embedding spaces, and encoder-decoder architectures.

<img src="images/transformer-architecture.jpg" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>


### Today’s Topic

- **Diffusion Models for Image Generation** [(Diffusion Demo)](https://www.kaggle.com/code/nasr73222/diffusion-session)

<img src="images\Image Generation-architecture.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

### Key Differences from Transformers

1. Move from understanding text to understanding images.
2. Focus on generating new images from learned data.

***

## Part 1: Understanding Text as Images

#### Historical Context

- In 2020, AI made massive leaps with GPT-3 in text. Researchers aimed to bring these ideas to images.


#### CLIP: Connecting Text and Vision

- **OpenAI’s CLIP (Contrastive Language-Image Pre-Training)** [(paper link)](https://arxiv.org/pdf/2103.00020), [(open AI)](https://openai.com/index/clip/)
- CLIP is trained on 400M internet-crawled image-caption pairs.

<img src="images\CLIP.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

**How was CLIP trained?**

- For each image-caption pair:
    - Minimize loss for matching pairs
    - Maximize loss for mismatched ones
    - Uses _cosine similarity_ between image and text embeddings.

<img src="images\overview-a.svg" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

**Cosine Similarity Function:**

$$
\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{ \| \vec{A} \| \| \vec{B} \| }
$$

**Contrastive Loss Function:**

**Goal:** Make the embedding of a matching image-text pair $$(\mathbf{e}_I, \mathbf{e}_T)$$ close together, and push apart non-matching pairs.

$$
\mathcal{L}_{contrastive} = -\log \frac{\exp(\cos(\mathbf{e}_I, \mathbf{e}_T) / \tau)}{\sum_{j=1}^{N} \exp(\cos(\mathbf{e}_I, \mathbf{e}_{T_j}) / \tau)}
$$

- **$\tau$**: Hyperparameter that scales the cosine similarity scores before they are passed into the softmax function.

**Shared embedding space:**

- Both pictures and captions are represented as vectors close together if paired.

<img src="images\overview-b.svg" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

**Demo 1:**

- Applications of the shared embedding space.

**Demo 2:**

- Zero-shot classification using only textual prompts.

***

## Part 2: Image Generation

#### Background

- Generating realistic images has been a problem for decades.
- Previous approaches:
    - **Autoencoders (AEs):** encode and reconstruct images via compression
    <img src="images\mona_lisa_AE.jpeg" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

    - **GANs (Generative Adversarial Networks):** generate sharp, realistic images using an adversarial approach
    [(This Person Does Not Exist)](https://thispersondoesnotexist.com)
    <img src="images\GAN.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

***

#### Enter Diffusion Models

- Named after the diffusion process from chemistry
- Might seem straight forward at the start.
- Denoise images and train a network to reverse the noise step right?
<img src="images\Diffusion Process.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>


- In 2020, the **DDPM** paper introduced a new approach [(paper link)](https://arxiv.org/pdf/2006.11239)
- Implementation from the paper suggests otherwise 
<img src="images\DDPMCode.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

**Explanation of Symbols in DDPM Training & Sampling Algorithms:**

- **$x_0$**: Original data sample (e.g., an image from the dataset).
- **$q(x_0)$**: Data distribution (how real images are distributed).
- **$t$**: Discrete time step (from $1$ to $T$), representing the amount of noise added.
- **$T$**: Total number of diffusion steps.
- **$\epsilon$**: Random noise sampled from a standard normal distribution $\mathcal{N}(0, I)$.
- **$\alpha_t$**: Noise schedule parameter at step $t$ (controls how much noise is added at each step).
- **$\bar{\alpha}_t$**: Cumulative product of $\alpha_t$ up to step $t$.
- **$\epsilon_\theta(x, t)$**: Neural network predicting the noise component at step $t$ for input $x$.
- **$z$**: Random noise for stochasticity in sampling (set to $0$ at the last step).
- **$\sigma_t$**: Standard deviation for added noise at step $t$ during sampling.

**Training (Algorithm 1):**
1. Sample a real image $x_0$ from the dataset.
2. Pick a random timestep $t$.
3. Sample random noise $\epsilon$.
4. Create a noisy image using $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
5. Train the model $\epsilon_\theta$ to predict the noise $\epsilon$ from the noisy image $x_t$ and timestep $t$.

**Sampling (Algorithm 2):**
1. Start from pure noise $x_T$.
2. For each timestep $t$ (from $T$ down to $1$):
    - Predict the noise $\epsilon_\theta(x_t, t)$.
    - Compute a less noisy image $x_{t-1}$ using the reverse process equation.
    - Add random noise $z$ (except at the final step).
3. Return $x_0$ as the generated image.

**The actual process involves 2 unusual steps:**
1. Predicting the total **added noise**, not just the previous denoised step.
<img src="images\training goal.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>
2. Adding noise with every generation step.
<img src="images\Generation noise.png" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>


***

### Visualizing Diffusion

- Points on a simple 2D grid represents images, they start in their original structure of readable images.

<img src="images\toy dataset.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

- As noise is added, they “random walk” away from their structure turning into noisy images.

<div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="images\toy dataset.png" style="width: 32%;"/>
    <img src="images\random toy dataset 2.png" style="width: 32%;"/>
    <img src="images\random toy dataset.png" style="width: 32%;"/>
</div>

- In general, diffusion models are **trained to reverse** this process.

***

#### Why Not Reverse Step by Step?

- Instead of predicting the small amount of noise to go from $x_t$ to $x_{t-1}$, the model is trained to predict the **total noise $ε$** that was added to the original image $x_0$ to create $x_t$
- **Why this works:** The noisy image $x_t$ is a direct combination of the original image $x_0$ and the total noise $ε$. If the model can predict $ε$ from $x_t$, it can also estimate the original $x_0$.
- **Equivalence:** Knowing the start $x_0$ and end $x_t$ points allows you to calculate any intermediate step (like $x_{t-1}$). Therefore, predicting the total noise $ε$ is mathematically equivalent to predicting the noise needed for a single step back.
- **Benefits:** This approach simplifies the model's objective. It has one consistent goal for every timestep `t`: find the original noise `ε`. This makes training more stable and efficient.

#### Score Function

- We now have a score function producing  a feild that guides the points to move to the original locations.

$$
x_{0} = \epsilon_\theta(x_{100})
$$

<img src="images\large t vector feild.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

#### Introducing **t**: The Time Parameter

- If noise steps are larger, the overall predicted feild for a pixel will point to the mean.
- if noise steps are smaller, the predicted feild will be refined pointing to the actuall structure of the data.
- Adding **t** as an input allows for **time-variant vector field** prediction — a flow-based model.
    - The vector field changes and becomes more detailed as $t (1 \rightarrow 0)$, restoring the original image structure.

$$
x_{0} = \epsilon_\theta(x_{100}, t=1.0)
$$

<div style="display: flex; justify-content: space-around; align-items: center;">
    <img src="images\large t vector feild.png" style="width: 45%;"/>
    <img src="images\small t vector feild.png" style="width: 45%;"/>
</div>

#### Randomness in Diffusion

- Why inject more randomness in the reverse step?
    - **Without random noise:** Images look blurry; all points move to the center (mean).
    - **With random noise:** Model produces diverse, high-quality images.
- The noise is gradually reduced as $ t \rightarrow 0 $ to allow for conversion.

<img src="images\result of no ransomness.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

***

#### Equations

**DDPM Forward/Noising Process:**

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Reverse/Sampling Process (Stochastic Differential Equation):**

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1- \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta (x_t, t) \right) + \sigma_t z
$$

- $ \epsilon_\theta $ is the model's prediction of the noise
- $ z \sim \mathcal{N}(0, I) $ for stochasticity
- $ Stochasticity refers to the quality of being random, unpredictable, or influenced by chance.

***

### Improvement: DDIM

- **DDIM** (Denoising Diffusion Implicit Models) provides a way to do faster, optionally deterministic sampling while keeping the same training objective as DDPM [(paper)](https://arxiv.org/abs/2010.02502).
- **Key change:**
    - Training stays identical: the model still learns $\epsilon_\theta(x_t t)$, the noise added to $x_0$ to produce $x_t$.
    - Sampling changes: compute an estimate of $x_0$ from $x_t$, then construct $x_{t-1}$ directly from that estimate.
    - You can remove sampling noise (deterministic sampling) or reintroduce a controlled amount via an η parameter.


- **DDIM Equation:**


$$
x_0 = \left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{ \sqrt{\alpha_t} } \right)
$$
.
$$
x_{t-1} = \sqrt{\alpha_{t-1}}\, x_0 \;+\; \sqrt{1 - \alpha_{t-1}}\, \epsilon_\theta(x_t, t)
$$

- **Controlled stochasticity:** introduce η ∈ [0,1] and set σ_t so that
  $$
  \sigma_t = \eta \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}\Big(1-\frac{\alpha_t}{\alpha_{t-1}}\Big)}.
  $$
  The general update becomes
  $$
  x_{t-1} = \sqrt{\alpha_{t-1}}\, x_0 \;+\; \sqrt{1 - \alpha_{t-1} - \sigma_t^2}\, \epsilon_\theta(x_t,t) \;+\; \sigma_t z,\quad z\sim\mathcal{N}(0,I).
  $$
  - η = 0 → deterministic DDIM (no added noise).
  - η = 1 → recovers DDPM-like stochasticity.



- **Practical benefits:**
    - Much fewer sampling steps are possible: you can sample on a subset of timesteps with good quality.
    - Deterministic sampling. yields reproducible outputs and often sharper results; adding noise to it introduces diversity.
    - Works seamlessly with previously trained models as only the sampler changes.


***

### Adding Guidance: CLIP \& Prompts

- **Diffusion models** alone can only generate objects similar to training data.
- **2022:** DALL-E2 (see [paper](https://cdn.openai.com/papers/dall-e-2.pdf)) combines CLIP embeddings with diffusion.
    - Embedding is added into the diffusion process, conditioning the output to come up with certain images based on certain prompts.

$$
f(x_{100}, t, \text{class})
$$

<img src="images\Dall-E-2-Architecture.jpg" style="width: 70%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

***

<img src="images\diffusion meme.jpg" style="width: 30%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

***

<img src="images\guidance.png" style="width: 50%; display: block; margin-left: auto; margin-right: auto; padding: 20px"/>

- But this initially introduces conflict (good images vs desired class).
- The need to generate a realistic image and the need to steer it into a specific class conflicted, leading to images not adhering to the desired classes very well.


#### Classifier-Free Guidance

- Use two flows:
    $$
    f(x_{100}, t, class), f(x_{100}, t, no class)
    $$

- Combine with weighting:

$$
\text{Guided Output} = f(\text{with class}) + \alpha [f(\text{with class}) - f(\text{no class})]
$$

- Higher $\alpha$ makes prompt guidance stronger.


#### Negative Prompts

- Further improved guidance:
    - Subtract unwanted class flows to actively repel certain outcomes.

$$
\text{Guided Output} = f(\text{with class}) + \alpha [f(\text{with class}) - f(\text{negative class})]
$$

- Visual demo with and without negative prompts

[(Stable Diffusion Test)](https://huggingface.co/spaces/stabilityai/stable-diffusion)

**Prompt:** `Keanu Reeves portrait photo of a asia old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes, 50mm portrait photography, hard rim lighting photography`

**Negative Prompt:** `disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal`

***

### Popular Image Generation Models Built on Diffusion Models

| Model | Developer | Core Architecture |
| :--- | :--- | :--- |
| **[Stable Diffusion](https://stability.ai/stable-image)** | Stability AI | **Latent Diffusion Model (LDM)** |
| **[Midjourney](https://www.midjourney.com/)** | Midjourney, Inc. (Independent Lab) | **Proprietary Diffusion Model** (U-Net based) |
| **[DALL·E 3](https://openai.com/dall-e-3/)** | OpenAI | **Cascaded Diffusion Model** (with Prior) |
| **[Imagen](https://deepmind.google/models/imagen/)** | Google DeepMind | **Cascaded Diffusion Model** (with T5 text encoder) |

### Key Architectural Concepts (brief)

1. **Latent Diffusion (Stable Diffusion):** run diffusion on compressed latents (VAE) to cut compute and memory.
2. **Cascaded / Super‑resolution:** generate low‑res images then progressively upscale with specialized diffusion models.
3. **Text conditioning:** quality of the text encoder (CLIP vs large LM like T5) strongly affects prompt fidelity.
4. **Styling & guidance:** model finetuning and guidance (classifier‑free, negative prompts) control aesthetics and adherence to prompts.


***

### Thank You!