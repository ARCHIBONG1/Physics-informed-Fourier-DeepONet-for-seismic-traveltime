![PI-Fourier DeepONet architechture2](https://github.com/user-attachments/assets/fc86b050-152e-492a-855c-5fe9768414ad)
# Physics-informed-Fourier-DeepONet-for-seismic-traveltime
Seismic traveltime calculation based on the eikonal equation is fundamental to various geophysical applications. With advances in deep learning, **neural operators** now allow networks to learn general solutions to PDEs. In particular, **Physics-Informed Neural Networks (PINNs)** enable training supervised by physical laws rather than labeled data.

In this project, we propose **PI-Fourier-DeepONet**, a hybrid physics-informed neural operator that combines the **Deep Operator Network (DeepONet)** and the **Fourier Neural Operator (FNO)** to approximate seismic traveltimes in complex media. DeepONet captures the operator mapping between traveltime, the velocity model, and source location, while FNO enhances the model's ability to learn global patterns in the frequency domain。

# Dataset
![4_velocity_families](https://github.com/user-attachments/assets/5662e93e-1c09-4d66-9e4d-638927abf401)

我们利用OpenFWI(https://github.com/lanl/OpenFWI)提供的数据集进行测试

# Results
![train_styleA](https://github.com/user-attachments/assets/7d7ffecc-4812-48d7-b202-93cdbfdd4d88)
PI-Fourier-DeepONet能够在eikonal方程的指导下，对不同速度模型进行地震走时模拟。我们利用scikit-fmm(https://github.com/scikit-fmm/scikit-fmm)提供的Fast Marching Method(FMM)作为作为比较的基准。

# Requirements

jax

jaxlib

scikit-fmm

matplotlib

numpy

jupyter notebook


