![PI-Fourier DeepONet architechture2](https://github.com/user-attachments/assets/fc86b050-152e-492a-855c-5fe9768414ad)
# Physics-informed-Fourier-DeepONet-for-seismic-traveltime
Seismic traveltime calculation based on the eikonal equation is fundamental to various geophysical applications. With advances in deep learning, **neural operators** now allow networks to learn general solutions to PDEs. In particular, **Physics-Informed Neural Networks (PINNs)** enable training supervised by physical laws rather than labeled data.

In this project, we propose **PI-Fourier-DeepONet**, a hybrid physics-informed neural operator that combines the **Deep Operator Network (DeepONet)** and the **Fourier Neural Operator (FNO)** to approximate seismic traveltimes in complex media. DeepONet captures the operator mapping between traveltime, the velocity model, and source location, while FNO enhances the model's ability to learn global patterns in the frequency domain。
# Requirements
jax 
jaxlib 
scikit-fmm
matplotlib
numpy

