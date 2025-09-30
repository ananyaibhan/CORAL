# **CORAL: Microplastic Segmentation using U-Net**

This project implements a **U-Net convolutional neural network** for binary segmentation of microplastic particles in aquatic sample images. The entire workflow is built in **Google Colab** with integration to **Google Drive** for data management and model checkpointing.

## **Project Overview**

- CORAL performs pixel-wise segmentation of microplastic particles from environmental water sample images.  
- Uses a standard U-Net architecture trained with binary cross-entropy loss.  
- The pipeline covers data preparation, model training, evaluation, and inference with visualization.  
- Dataset Source: https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision

## **Workflow**

### **Data Preparation**

- Input images and binary masks are resized to **256×256 pixels** using albumentations.  
- Masks are binarized using a threshold (>127) and converted to torch tensors for model input.

### **Model Architecture**

- The U-Net consists of **4 downsampling and 4 upsampling blocks** with batch normalization and skip connections.  
- Final output is a **1-channel sigmoid activation** representing pixel-wise microplastic presence.

### **Training and Evaluation**

- Model trained with **batch size 16** using binary cross-entropy loss for **20 epochs**.  
- Validation performance measured using **Intersection over Union (IoU)** metric.  
- Model checkpoints are saved after every epoch.

### **Inference and Visualization**

- On test images, the model outputs a probability mask that is thresholded and upscaled to original size.  
- Microplastic regions are detected via contour finding, and displayed with red bounding boxes on the original image.  
- Visualizations of original and annotated images are shown side-by-side using matplotlib.

## **Performance Metrics**

- Final validation IoU at epoch 20: **0.6851**  
- Validation IoU ranged from approximately **0.57 to 0.69** in late training epochs.  
- Final training and validation loss: approximately **0.1108**.  
- Outputs are binary masks with bounding box overlays; no multi-class detection or post-processing applied.

## **Usage Summary**

- Mount Google Drive in Colab and place image/mask folders as required.  
- Install dependencies (PyTorch, albumentations, matplotlib, tqdm, PIL, NumPy, OpenCV).  
- Run training and validation cells sequentially.  
- Upload image for inference and view annotated detection results inline.  
- Checkpoints are automatically handled, allowing training resumption.

## **Scope & Considerations**

- Current implementation uses a basic **U-Net** and **binary segmentation** only.  
- No object detection, multi-class segmentation, or advanced post-processing techniques are included.  
- Dataset loading and management require user setup; automated downloading is not provided.  
- Metrics tracked include loss and IoU; additional metrics and inference speed can be added by users if needed.
