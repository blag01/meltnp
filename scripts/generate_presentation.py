import collections 
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pathlib import Path
import os

prs = Presentation()

# Helper to add standard slide
def add_slide(title, content, image_paths=None):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = title
    
    body_shape = slide.placeholders[1]
    tf = body_shape.text_frame
    
    # Process text content
    if isinstance(content, list):
        for i, point in enumerate(content):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = point
            p.font.size = Pt(18)
    
    # Insert multiple images if provided
    if image_paths:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # Distribute horizontally if multiple
        num_images = len(image_paths)
        start_left = 5.0 if num_images == 1 else 1.0
        width = 4.5 if num_images == 1 else 4.0
        top = 2.5 if num_images == 1 else 4.0
        
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                left_pos = start_left + (i * width)
                slide.shapes.add_picture(path, Inches(left_pos), Inches(top), width=Inches(width))

# Slide 1: Title
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "Scaling Transformer Neural Processes"
subtitle.text = "Robustness Benchmarking and Test-Time Adaptation\nPrepared automatically."

# Slide 2: Core Problem
add_slide("The Core Problem", [
    "Neural Processes (NPs) are incredibly powerful at few-shot function regression, but they struggle heavily with Out-Of-Distribution (OOD) data.",
    "Small perturbations (noise, sensor failures, structural bias) often destroy regression coherence.",
    "Our goal: Create a framework capable of measuring robustness under distribution shift."
])

# Slide 3: TNP Architecture Upgrades
add_slide("TNP Architecture Upgrades", [
    "Moving from flat Cross-Attention to deep Transformer Neural Processes (TNP) (L=3 stacked Transformers with residual connections).",
    "Optional Latent TNP path via Latent Encoders.",
    "Targeted regularized ELBO sampling for stochastic uncertainty bands."
], ["results/tnp/10/gp_robust/weights.png"])

# Slide 4: Structured Benchmarking Suite
add_slide("Structured Benchmarking Suite", [
    "Developed an automated execution pipeline tracking three rigorous dimensions: (NLL, MSE, ECE).",
    "Introduced 6 structured mathematical corruptions:",
    " - Heteroskedastic scaling",
    " - Extreme Outlier injection",
    " - Covariate Shift",
    " - Warp Shift, Bias, and pure Noise"
], ["results/tnp/10/plots/outlier/sinusoid_nll.png"])

# Slide 5: Test-Time Adaptation Concepts
add_slide("The 'Magic' of Test-Time Adaptation", [
    "Adapting cleanly to corruption at inference time *without any labels*.",
    "Using Bidirectional Pseudo-Likelihood Optimization where the TNP acts as a structural prior.",
    "MLPs natively learn to 'subtract' structural noise until the manifold looks consistently predictable."
])

# Slide 6: Visualizing the Adaptation
add_slide("Visualizing Denoising Adaptation", [
    "Comparing the Neural Process's predictive uncertainty BEFORE and AFTER unsupervised gradient descent.",
    "Left: The noisy data causes total predictive collapse.",
    "Right: The MLP cleans the data; the NP collapses its variance back onto the precise sine wave."
], [
    "results/tnp/10/test_time_adaptation/before_mlp.png",
    "results/tnp/10/test_time_adaptation/after_mlp.png"
])

# Slide 7: Adaptation Descent Curve
add_slide("Adaptation Descent Profile", [
    "The Negative Log-Likelihood drops violently within the first 50 iterations.",
    "The NP pulls the small noise-mapping parameters down into the nearest high-probability manifold."
], ["results/tnp/10/test_time_adaptation/optimization_curve_mlp.png"])

# Slide 8: Langevin Dynamics (SGLD)
add_slide("Escaping Sinkholes with Langevin Dynamics (SGLD)", [
    "Maximum Likelihood (MLE) alone often gets stuck mapping noise into trivial flat geometries.",
    "By injecting parameterized Langevin noise (diffusion) directly into parameter updates, the adaptation easily escapes local minima!",
    "Comparing different intensities of diffusion prevents the manifold from folding."
], ["results/tnp/10/tta_budget/budget_sinusoid_Heteroskedastic_s1.0.png"])

Path("assets").mkdir(exist_ok=True)
output_path = "assets/Neural_Processes_Robustness_Presentation.pptx"
prs.save(output_path)
print(f"Presentation saved successfully to {output_path}")
