import collections 
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pathlib import Path
import os

prs = Presentation()

# Helper to add standard slide
def add_slide(title, content, image_path=None):
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
    
    # Insert image if provided and exists
    if image_path and os.path.exists(image_path):
        # We'll put the text on the left and image on the right if both exist
        # But for simplicity, we can just overlay it on the bottom or right.
        slide.shapes.add_picture(image_path, Inches(4.5), Inches(2.5), width=Inches(5.0))

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
], "results/tnp/10/gp_robust/weights.png")

# Slide 4: Structured Benchmarking Suite
add_slide("Structured Benchmarking Suite", [
    "Developed an automated execution pipeline tracking three rigorous dimensions: (NLL, MSE, ECE).",
    "Introduced 6 structured mathematical corruptions:",
    " - Heteroskedastic scaling",
    " - Extreme Outlier injection",
    " - Covariate Shift",
    " - Warp Shift, Bias, and pure Noise"
], "results/tnp/10/plots/outlier/sinusoid_nll.png")

# Slide 5: Test-Time Adaptation
add_slide("The 'Magic' of Test-Time Adaptation", [
    "Adapting cleanly to corruption at inference time *without any labels*.",
    "Using Bidirectional Pseudo-Likelihood Optimization where the TNP acts as a structural prior.",
    "MLPs natively learn to 'subtract' structural noise until the manifold looks consistently predictable."
], "results/tnp/10/test_time_adaptation/before_mlp.png")

# Slide 6: Langevin Dynamics (SGLD)
add_slide("Escaping Sinkholes with Langevin Dynamics (SGLD)", [
    "Maximum Likelihood (MLE) alone often gets stuck mapping noise into trivial flat geometries.",
    "By injecting parameterized Langevin noise (diffusion) directly into parameter updates, the adaptation easily escapes local minima!",
    "Comparing x5, x10 diffusion intensity dynamically prevents the manifold from folding."
], "results/tnp/10/tta_budget/budget_sinusoid_Heteroskedastic_s1.0.png")

output_path = "Neural_Processes_Robustness_Presentation.pptx"
prs.save(output_path)
print(f"Presentation saved successfully to {output_path}")
