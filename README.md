# PathMNIST Classification using Deep Learning

In this homework, we will solve a medical image classification problem using deep learning. 
The objective is to build a model that can automatically classify pathology images from the PathMNIST dataset.

The PathMNIST is part of the MedMNIST collection, containing 107,180 RGB images of pathological specimens with 9 different classes. The dataset is already split into training, validation, and test sets.

Specific objectives for this homework are:
1. Practice working with medical image datasets
2. Implement a CNN architecture for classification
3. Use TensorBoard to monitor and analyze training
4. Evaluate model performance with appropriate metrics

### Submission Requirements
- All source code files
- Single report file in markdown format with the following sections:
    - Dataset Access and Loading (explain how you loaded the dataset) (10 pts)
    - Model Architecture (explain your model architecture and show your computational graph using TensorBoard) (10 pts)
    - Training Implementation (explain your training implementation and show your training and validation loss curves using TensorBoard) (10 pts)
    - Model Evaluation (explain your model evaluation and show your confusion matrix, per-class precision and recall, and at least 1 example of a correct and incorrect classifications) (10 pts)
- TensorBoard logs (only for the best performing model)

## Dataset Access and Loading
The PathMNIST dataset can be accessed from [MedMNIST v2](https://medmnist.com/). You can download the dataset directly or use the medmnist package:

```python
import medmnist

# Load the PathMNIST dataset
dataset = medmnist.dataset.PathMNIST(split='train')
```

The dataset has already been downloaded and saved in the `/home/osz09/DATA_SharedClasses/SharedDatasets/MedMNIST/pathmnist.npz` directory. You can load the dataset using the following code:

```python
data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedMNIST/"
data = np.load(join(data_dir, "pathmnist.npz"))
```
### Dataset Exploration 
Create a file called `analyze_data.py` that:
- Loads the PathMNIST dataset from the npz file
- Displays basic statistics (number of images per class, image dimensions)
- (Optional) Visualizes random samples from each class in a grid
- (Optional) Shows class distribution in training/validation/test sets

### Model Architecture 
Design and implement a CNN in `MyModels.py` that includes:
- Multiple convolutional layers with appropriate kernel sizes
- Batch normalization layers
- Max pooling layers
- Dense layers
- (Optional) Skip connections

Your model description should include:
- Layer-by-layer architecture specification
- Number of parameters
- Justification for architectural choices
- Visual representation of the model (you can use tensorboard  or tools like [NN-SVG](http://alexlenail.me/NN-SVG/index.html))

### Training Implementation 
Create a `Training.py` script that implements:
- Training loop with batches and epochs
- Validation after each epoch
- Early stopping (optional)
- Learning rate scheduling (optional)

TensorBoard logging should include:
- Training and validation loss curves
- Accuracy metric
- Model computational graph

### Model Evaluation 
In your main script (`PathMNIST.py`), implement:
- Model training
- Performance evaluation on validation set
- Visualization of results
