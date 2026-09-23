# Multimodal image retrieval

Retrieving relevant from a database is a challenging task. This project focused on retrieving images using:
* only a textual description of the image (text-to-image retrieval)
* a reference image and a description of the wanted modification to it (composed image retrieval)

Several vision-language models were used, with a special focus on the model CLIP and contrastive learning. 
Zero-shot experiments were conducted using publicly available datasets, as well as a personally curated one (using the simple annotation tooling in `/annotator`). 
Finally, a model combining text and image representations in the problem of composed image retrieval was trained on the FashionIQ dataset.

More information can be found in my BSc thesis (in Croatian): https://repozitorij.fer.unizg.hr/object/fer:13946
