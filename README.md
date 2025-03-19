# General Super-Resolution with Plug-and-Play Diffusion Denoising

This project explores and experiments with the general super-resolution problem \( y = DHx + n \) using the plug-and-play framework, employing a diffusion model as a denoiser. The approach is inspired by the [Denoising Diffusion Models for Plug-and-Play Image Restoration](https://arxiv.org/abs/2204.11824), with its implementation available in the [DiffPIR](./DiffPIR/) directory.

The DiffPIR code builds on the [Deep Plug-and-Play Image Restoration (DPIR)](https://arxiv.org/abs/2008.09351) implementation, found in the [DPIR](./DPIR) directory. Both share a similar structure, with DiffPIR replacing the denoiser with a diffusion model.

In both repositories, the general super-resolution problem is addressed via an analytical solution in the data fidelity step:

`x = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)`

- **DiffPIR**: [Jump to line 317 of main_ddpir_sisr.py](./DiffPIR/main_ddpir_sisr.py#L317).

- **DPIR**: [Jump to line 207 of main_dpir_sisr.py](./DPIR/main_dpir_sisr.py#L207).

This analytical solution was first proposed in [Fast Single Image Super-Resolution Using a New Analytical Solution for l2-l2 Problems](https://hal.science/hal-01373784/). Implementations in :

- **Matlab** : [Matlab Fast Single Image Super-Resolution](./fast-single-super-resolution-analytical-solution/matlab/).
- **Python** : [Python Fast Single Image Super-Resolution](./fast-single-super-resolution-analytical-solution/python/).

## Repository Structure

- **`ddpm-training`**: Data and training sets for diffusion model training.
- **`DiffPIR`**: DiffPIR implementation with configs, guided diffusion model, kernels, and utilities.
- **`DPIR`**: DPIR implementation with models, kernels, and utilities.
- **`fast-single-super-resolution-analytical-solution`**: MATLAB and Python code for the analytical super-resolution solution.
