
# ML-proj2

## Folder and Script Descriptions

- **training1 folder**: Stores the enlarged training images, used to match the test set.
- **Classification.py**: Implements logistic regression, CNN_simple, FCN8s_simple, and FCN8s models.
- **Classification2.py**: Implements FCN8S_stride16_simplified and FCN8S_stride16 models.
- **Classification3.py**: Calls the Segformer model.
- **Data_preprocess.py**: Handles data preprocessing, image segmentation, normalization, feature extraction for logistic regression, and data augmentation.
- **mask_to_submission.py**: Generates submission CSV files.
- **submission_to_mask.py**: Visualizes CSV files.
- **helper.py**: Uses interpolation to generate images in the training1 folder.
- **overlap.py**: Users can test the performance of the model by visuralizing the predicted roads on the original test images.
## How to Run

- Run `python linear.py` to execute the logistic regression model. Outputs ACC and saves results to `submision1.csv`.
- Run `python cnn_simple.py` to execute the CNN_simple model. Outputs ACC and MIou, saves results to `submision2.csv`.
- Run `python fcn8s_simple.py` to execute the FCN8s_simple model. Outputs ACC and MIou, saves results to `submision3.csv`.
- Run `python fcn8s.py` to execute the FCN8s model. Outputs ACC and MIou, saves results to `submision4.csv`.
- Run `python fcn8s_stride16.py` to execute the FCN8s_stride16 model. Outputs ACC and MIou, saves results to `submision5.csv`.
- Run `python fcn8s_stride16_simple.py` to execute the FCN8s_stride16_simplified model. Outputs ACC and MIou, saves results to `submision6.csv`.
- Run `python Segformer.py` to execute the Segformer model. Outputs ACC and MIou, saves results to `submision7.csv`.
- Run `python overlap.py` tand input the image index and model number for visualization. And the output is saved in `visualization.png`
The corresponding result images will be generated in paths like `/test_set_images/test_1/test_1.png_result1`.

We use the results from `submission7` for the final submission.