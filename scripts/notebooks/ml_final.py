# Converted from notebook: ml-final.ipynb

# %% [code cell]
! pip install gdown

# %% [code cell]
import os  
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Input, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

# %% [code cell]
# Paths 
dataset_dir = r'/content/drivee/Shareddrives/ML_Data/2D_datasets'  # Directory with class folders
output_dir = r'/content/drivee/Shareddrives/ML_Data/2D_datasets/Split_Data'  # Directory to store split dataset

# Create output folders
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# %% [code cell]
# Loop through each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)

    # Skip if not a directory or if it's the output directory
    if not os.path.isdir(class_path) or class_name == 'Split_Data':
        continue

    # Get all image filenames for the current class
    image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]

    # Split into train (80%) and test (20%)
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # From the training set, split into train (80% of train) and validation (20% of train)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    # Define destination paths for this class
    train_class_dir = os.path.join(output_dir, "train", class_name)
    val_class_dir = os.path.join(output_dir, "val", class_name)
    test_class_dir = os.path.join(output_dir, "test", class_name)

    # Create directories
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move files to their respective directories
    for f in train_files:
        shutil.copy(f, train_class_dir)
    for f in val_files:
        shutil.copy(f, val_class_dir)
    for f in test_files:
        shutil.copy(f, test_class_dir)

print("Dataset splitting completed.")

# %% [code cell]
# Define the Splitted dataset directory uploaded from the drive to be used in kaggle
dataset_dir = "/kaggle/input/2d-mri-splitteddatavol-2/Split_Data_vol2"

# Define directories for train, validation, and test  
train_dir = os.path.join(dataset_dir, 'train')  
val_dir = os.path.join(dataset_dir, 'val')  
test_dir = os.path.join(dataset_dir, 'test')  

# %% [code cell]
# Manually define label mapping  
label_mapping = {  
    'AD': 0,  # Alzheimer's Disease  
    'CN': 1,  # Cognitively Normal  
    'MCI': 2   # Mild Cognitive Impairment  
}  

# Normalization layer  
normalization_layer = layers.Rescaling(1./255)  

# Preprocess function  
def preprocess_data(images, labels):  
    images = normalization_layer(images)  # Normalize images to [0, 1]  
    return images, labels  

# Load datasets with processing  
def load_dataset(directory):  
    return tf.keras.utils.image_dataset_from_directory(  
        directory,  
        image_size=(224, 224),  
        color_mode="grayscale",  
        batch_size=32,  
        label_mode='categorical'  # this will map labels based on folder names  
    ).map(lambda x, y: preprocess_data(x, y))  

# Load datasets  
train_ds = load_dataset(train_dir)  
val_ds = load_dataset(val_dir)  
test_ds = load_dataset(test_dir)  

# To extract manual labels, create a custom labelling function  
def manual_label_extractor(labels):  
    return tf.one_hot(tf.argmax(labels, axis=-1), depth=len(label_mapping))  

# Set the labels manually  
train_ds = train_ds.map(lambda x, y: (x, manual_label_extractor(y)))  
val_ds = val_ds.map(lambda x, y: (x, manual_label_extractor(y)))  
test_ds = test_ds.map(lambda x, y: (x, manual_label_extractor(y)))  

# %% [code cell]
# Build CNN1 with regularization
def build_cnn1_with_regularization(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    #output = layers.Dense(3, activation='softmax')(x)  # Output layer size based on number of classes; used only if I will train that model alone
    return models.Model(inputs=inputs, outputs=x)

# Build and compile the model  
input_shape = (224, 224, 1)  # Grayscale images  
cnn_model1 = build_cnn1_with_regularization(input_shape) 

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.85  # Decay learning rate by 15% every epoch

# Compile the model
cnn_model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks, we could use early stopper but in our case the epochs already is relatively small so it's unnecessary 
callbacks = [ 
    LearningRateScheduler(lr_scheduler) 
]

cnn_model1.summary()

# %% [code cell]
# Train the model (Dropout 0.3)
history = cnn_model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# %% [code cell]
# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.tight_layout()
    plt.show()

# %% [code cell]
plot_history(history) #(Dropout 0.3)

# %% [code cell]
# Evaluate the CNN1 model on the test dataset (Dropout 0.3)
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model1, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
# Train the model (Dropout 0.4)
history = cnn_model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# %% [code cell]
plot_history(history) #(Dropout 0.4)

# %% [code cell]
# Evaluate the CNN1 model on the test dataset (Dropout 0.4)
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model1, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
# Train the model (Dropout 0.5)
history = cnn_model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# %% [code cell]
plot_history(history) #Dropout 0.5

# %% [code cell]
# Evaluate the CNN1 model on the test dataset (Dropout 0.5)
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model1, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
plot_history(history) #from last time run

# %% [code cell]
# Function to evaluate the model on the test dataset  
def evaluate_model(model, test_ds, num_classes):  
    test_loss, test_accuracy = model.evaluate(test_ds)  # Evaluate the model  
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')  

    # Extract test images and labels from the test dataset  
    test_images = []  
    test_labels = []  

    for images, labels in test_ds:  
        test_images.append(images.numpy())  # Collecting images  
        test_labels.append(labels.numpy())  # Collecting labels  

    test_images = np.concatenate(test_images)  # Combine all batches of images  
    test_labels = np.concatenate(test_labels)  # Combine all batches of labels  

    # Predict class probabilities  
    predictions = model.predict(test_images)  # Make predictions  
    predicted_classes = np.argmax(predictions, axis=1)  # Get predicted class indices  
    true_classes = np.argmax(test_labels, axis=1)  # Get true class indices  

    return test_loss, test_accuracy, true_classes, predicted_classes, predictions  

# Function to plot confusion matrix  
def plot_confusion_matrix(true_classes, predicted_classes, class_names):  
    cm = confusion_matrix(true_classes, predicted_classes)  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)  
    disp.plot(cmap='viridis', values_format='d')  
    plt.title('Confusion Matrix')  
    plt.show()  

# Function to plot ROC curve  
def plot_roc_curve(predictions, true_classes, num_classes):  
    # Binarize the test labels (for multi-class ROC)  
    y_test_bin = label_binarize(true_classes, classes=np.arange(num_classes))  
    
    fpr = {}  
    tpr = {}  
    roc_auc = {}  

    for i in range(num_classes):  
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])  
        roc_auc[i] = auc(fpr[i], tpr[i])  

    plt.figure(figsize=(8, 6))  
    for i in range(num_classes):  
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')  
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('ROC Curve')  
    plt.legend(loc='lower right')  
    plt.show()  

# Define number of classes and class names (like ['AD', 'CN', 'MCI'])  
num_classes = len(label_mapping)  # Assuming you've defined this earlier in your code  
class_names = list(label_mapping.keys())  # Use keys from your label mapping to get class names  

# %% [code cell]
# Evaluate the model on the test dataset  
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model1, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  

# %% [code cell]
print(f"Classification Report:\n{report}")

# %% [markdown]


# %% [code cell]
# Build CNN2 with regularization
def build_cnn2_with_regularization(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(512, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    #output = layers.Dense(3, activation='softmax')(x)  # Output layer size based on number of classes
    return models.Model(inputs=inputs, outputs=x)

# Build and compile the model  
input_shape = (224, 224, 1)  # Grayscale images  
cnn_model2 = build_cnn2_with_regularization(input_shape) 

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.85  # Decay learning rate by 10% every epoch

# Compile the model
cnn_model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    LearningRateScheduler(lr_scheduler)
]

# %% [code cell]
cnn_model2.summary()

# %% [code cell]
# Train the model (Dropout 0.3)
history2 = cnn_model2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# %% [code cell]
plot_history(history2) #Dropout 0.3

# %% [code cell]
# Evaluate the CNN1 model on the test dataset (Dropout 0.3)
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model2, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
# Train the model (Dropout 0.5)
history2 = cnn_model2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# %% [code cell]
plot_history(history2) #Dropout 0.5

# %% [code cell]
# Evaluate the CNN2 model on the test dataset (Dropout 0.5)
print("Evaluating the model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model2, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
# Create CNN1 and CNN2
input_shape = (224, 224, 1)  # Adjusted for consistency
cnn1 = build_cnn1_with_regularization(input_shape)
cnn2 = build_cnn2_with_regularization(input_shape)

# Combine the models without duplicate input
def combine_models(cnn1, cnn2):
    shared_input = layers.Input(shape=(224, 224, 1))  # Shared input layer
    output1 = cnn1(shared_input)  # Output from CNN1
    output2 = cnn2(shared_input)  # Output from CNN2
    combined_output = layers.concatenate([output1, output2])  # Concatenate outputs
    x = layers.Dense(128, activation='relu')(combined_output)  # Additional dense layer
    x = layers.Dropout(0.2)(x)  # Add Dropout layer before final output
    final_output = layers.Dense(3, activation='softmax')(x)
    return models.Model(inputs=shared_input, outputs=final_output)

# Build combined model
combined_cnn = combine_models(cnn1, cnn2)

# Compile the combined model
combined_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.9  # Decay learning rate by 5% every epoch

# Callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
]

# Summary of the combined model
combined_cnn.summary()


# %% [code cell]
# Train the combined model
history_comined = combined_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# Plot training history
plot_history(history)

# %% [code cell]
# Evaluate the combined model on the test dataset  
print("Evaluating combined model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(combined_cnn, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
plot_history(history2)#last run

# %% [code cell]
# Evaluate the model on the test dataset  
print("Evaluating CNN2 model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(cnn_model2, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")

# %% [code cell]
# Create CNN1 and CNN2
input_shape = (224, 224, 1)  # Adjusted for consistency
cnn1 = build_cnn1_with_regularization(input_shape)
cnn2 = build_cnn2_with_regularization(input_shape)

# Combine the models without duplicate input
def combine_models(cnn1, cnn2):
    shared_input = layers.Input(shape=(224, 224, 1))  # Shared input layer
    output1 = cnn1(shared_input)  # Output from CNN1
    output2 = cnn2(shared_input)  # Output from CNN2
    combined_output = layers.concatenate([output1, output2])  # Concatenate outputs
    x = layers.Dense(128, activation='relu')(combined_output)  # Additional dense layer
    #x = layers.Dropout(0.2)(x)  # Add Dropout layer before final output
    final_output = layers.Dense(3, activation='softmax')(x)
    return models.Model(inputs=shared_input, outputs=final_output)

# Build combined model
combined_cnn = combine_models(cnn1, cnn2)

# Compile the combined model
combined_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.95  # Decay learning rate by 10% every epoch

# Callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
]

# Summary of the combined model
combined_cnn.summary()


# %% [code cell]
# Train the combined model
history_comined = combined_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    batch_size=8,
    callbacks=callbacks
)

# Plot training history
plot_history(history)

# %% [code cell]
# Evaluate the combined model on the test dataset  
print("Evaluating CNN2 model on the test dataset...")  
test_loss, test_accuracy, true_classes, predicted_classes, predictions = evaluate_model(combined_cnn, test_ds, num_classes)  

# Plot confusion matrix  
plot_confusion_matrix(true_classes, predicted_classes, class_names)  

# Plot ROC curve  
plot_roc_curve(predictions, true_classes, num_classes)  

# Generate and print classification report  
report = classification_report(true_classes, predicted_classes, target_names=class_names)  
print(f"Classification Report:\n{report}")