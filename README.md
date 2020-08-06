# COMP9517
## How to run the code to reimplement the results
### Cell Detection
We use the notebook `segmentation_and_detection.ipynb` to generate bounding box coordinates for all given frames. We used the latest version of opencv-python, sklearn and matplotlib library in this notebook.

First, run the second cell in `Data Preprocessing` section. It contains a function called `thres_preprocessing()` which would carry out the segmentation for each type of dataset. A sample output for each type of dataset is listed below the cell.

Then, go to `Method-2: Use Marching Squares Algorithm` section. It contains a function `label_contours()` which would call the segmentation method above and carry out the cell detection after images are segmented. For the input of this function, the first argument is the path of your input dataset, the second argument is the path of the output (you need to ensure the path exists before using the function), and the rest of the arguments are just used for segmentation processing (they are same with the examples at the bottom of `Data Preprocessing` section). The function would store the cell coordinate details in the given output path. A sample output is displayed in `Further usage` section. It re-loaded the coordinate outputs from the function and successfully re-drew the corresponding bounding boxes. Further detailed usage is in the notebook of `Tracking.ipynb`.

### Cell Tracking
We use the notebook `Tracking.ipynb` to track cells detected in the previous section. 
First go to the second cell and check the path is the file which contains the numpy file and images from the cell detection section.
Run the entire notebook in order of the cells.
### Motion Analysis
We use the notebook `Task3.ipynb` to do the motion analysis. 
First go to the second cell and check the path is the file which contains the pickel file and images from the cell tracking and mitosis section. 
Run the entire notebook in order.
