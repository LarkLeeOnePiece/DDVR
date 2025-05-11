# Computational scientific Imaging

## repo structure

- D2VR folder contains the FBP CUDA version

- scripts folder contains the FBP+Volume render python implementation


## Dataset 

you can get dataset from [Clean Implicit 3D Structure from Noisy 2D STEM Images](https://github.com/HannahKniesel/Implicit-Electron-Tomography)

## Instruction for code running

- 1. You need to set the data path in the coresponding place in py_render.py

- 2. For the FBP CUDA version, you need to set the path in D2VR\scr\main.cu

## Some visual result

### Render UI
We implment the interaction UI based on pygame
![Comparison](./assets/renderUI.gif "Magic Gardens")

### FBP reconstruction result

For the dataset, we used four dataset from  [Clean Implicit 3D Structure from Noisy 2D STEM Images](https://github.com/HannahKniesel/Implicit-Electron-Tomography)

1. Synthetic data with noise

![fbp](./assets/Syn_Noisy_wAO_vol_512x512x512.gif)


2. Synthetic data without noise

![fbp](./assets/Syn_Clean_wAO_vol_512x512x512.gif)

3. Real nanoparticles

![fbp](./assets/Real_NanoParticles_wAO_vol_512x512x512.gif)

4. Read covid infected cell

![fbp](./assets/Real_Denoised_BM3D_wAO_vol_512x512x512.gif)

### Some render result
This is one result showing the comparison between DVR and D2VR with AO

![Comparison](./assets/comparison_with_error.gif "Magic Gardens")
