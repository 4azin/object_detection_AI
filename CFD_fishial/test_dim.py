import open_clip

print("Loading bioclip-2...")
model_l, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
print("BioCLIP-2 dim:", model_l.text_projection.shape if hasattr(model_l, "text_projection") and model_l.text_projection is not None else "No projection")

print("Loading bioclip-2.5-vith14...")
model_h, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14")
print("BioCLIP-2.5 dim:", model_h.text_projection.shape if hasattr(model_h, "text_projection") and model_h.text_projection is not None else "No projection")
