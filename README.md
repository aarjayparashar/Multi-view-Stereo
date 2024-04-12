# HW5

**Please first install the requirements.txt!** Then you could start with the notebooks in vscode.

## Hints

Common Mistakes in function `compute_disparity_map`:

- Directly copy my example code, forgetting to use the k_size arg passed in.
- When you initialize the buffer you want to use later in the iteration, please explicitly define it as float64. Initialize as other types but convert to float64 at the end of your code may not work.

Common Mistakes in function `compute_rectification_R`:

- Check your rotation matrix is a rotation!
- Be careful about the order of the cross product

Common Mistakes in function `ssd_kernel`:

- The two s correspond to the sum squared, please don’t use the sqrt.
- The most possible reason your code runs slow is because of your implementatio of the kernel fucntion, one hint for your to think about: check the numpy boardcasting, what’s the output shape of two array A-B with shape A: [M,1,P,3] and shape B: [1,N,P,3]?

Not sure why we need an L-R consistency check? You can visualize the consistency mask aligned with the right view (what you returned in the function is the one that aligns with the left view), and you will immediately realize the reason

