#!/usr/bin/env python
# coding: utf-8

# # Digital Mirror
# 
# ### John S Butler and Cian McLoughlin
# 
# The __Digital Mirror__ is a collaboration between artist Cian McLoughlin and mathematician and neuroscientist John Butler of TU Dublin. Together, they explore the connections between art, machine learning, and neuroscience in the context of facial recognition.
# 
# This code creates the eigenfaces from a video recording by:
# 1. A 10 second video at 30 frames a second giving 300 images that are 640x480. 
# 2. These images are averaged to get the "meanface".
# 3. The "meanface" is subtracted from each image, Convert each subtracted image from a 2D matrix 640x480 to a vector 1x307200. Make a matrix from all the subtraced images which is 300x307200 
# 4. Calculate then eigenvalues and eigenvectors of subtracted images using the Principal Component Analysis (PCA)
# 5. Plot the Eigenfaces
# 6. Reconstruct an Image
# 7. How many eigenfaces do we need?

# ## LIBRARIES NEEDED

# In[3]:


import cv2  # Computer vision tasks
import matplotlib.pyplot as plt  # Plotting and visualization
import os  # Interacting with the operating system (file/directory management)
import numpy as np  # Numerical computing (arrays, matrices)
from sklearn.decomposition import PCA  # Dimensionality reduction (PCA)
import time  # Time-related functions (e.g., sleep, execution time)


# ## IMAGE RECORDING AND PLOTTING FUNCTIONS

# In[5]:


# Set the name of the folder where the files will be stored
name = "Test"

# Record the start time for measuring how long the script takes to run
start_time = time.time()

# Specify the directory name by concatenating the base directory with the specified name
# In this case, it's creating a path like "EigenFace/Test"
directory_name = "EigenFace/" + name

# Check if the directory already exists and if it is a directory (not a file with the same name)
if os.path.exists(directory_name) and os.path.isdir(directory_name):
    # If the directory exists, print a confirmation message
    print(f"The directory '{directory_name}' exists.")
else:
    # If the directory does not exist, create it using os.mkdir
    os.mkdir(directory_name)
    # Print a success message indicating the directory was created
    print(f"Directory '{directory_name}' created successfully.")


# In[6]:


### IMAGE RECORDING AND PLOTTING FUNCTIONS

### VIDEO RECORDING 
def record_video(output_path, frame_width=640, frame_height=480, fps=30, duration=10):
    """
    Records a video using the default camera and saves grayscale frames as images.

    Parameters:
    output_path (str): The path to save the output video.
    frame_width (int): Width of the video frame.
    frame_height (int): Height of the video frame.
    fps (int): Frames per second.
    duration (int): Duration of the video in seconds.

    Returns:
    images (list): List of grayscale images captured from the video.
    """

    # Open the default camera (usually the first camera)
    cap = cv2.VideoCapture(0)
    # List of facial expressions to show on the screen during recording
    FACES = ["Happy", "Angry", "Confused", "Puppy Dog", "Weirded Out", 
             "Blank", "Delighted", "Sick", "Shocked", "Sad", "Happy Again"]
    # Set the width and height of the frames to be captured
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 output
    output_video="EigenFace/"+output_path+"/"+output_path+".mp4"
    # Uncomment the following line to save video
    # out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    face_count=0

    # Calculate the number of frames to capture based on the duration and fps
    num_frames_to_capture = int(duration * fps)
    frame_count = 0
    images=[]
    while frame_count < num_frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            break

        # Uncomment to write the frame to the output file
        #   out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Look at the camera and move your head', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        images.append(gray_frame)
        # Save the grayscale frame as an image
        frame_filename = os.path.join('output_images_folder', f'frame_{frame_count:04d}.png')
#        cv2.imwrite(frame_filename, gray_frame)
        # Display the frame (optional)
        if frame_count%np.round(num_frames_to_capture/10)==0:
            face_count=face_count+1

        cv2.imshow(f"Make a {FACES[face_count]} face", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
    # Release the camera and file writer
    cap.release()
    # Uncomment the following line to save video
    # out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    print(f"Images recorded ")
    return images


# Function to plot the mean face and eigenfaces
def plot_gallery(images, titles, h, w,name, n_row=1, n_col=6):
    """
    Plots a gallery of images (e.g., mean face and eigenfaces).

    Parameters:
    images (list): List of images to display.
    titles (list): Titles for each subplot.
    h (int): Image height.
    w (int): Image width.
    name (str): Name for the output file.
    n_row (int): Number of rows in the plot.
    n_col (int): Number of columns in the plot.
    """

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    eigen_filename=name+"eigenfaces.png"
    eigen_filename="EigenFace/"+name+"/"+eigen_filename
    plt.suptitle("The Digital Mirror: AI and the Evolution of Portraiture",fontsize=24)
    plt.tight_layout() 
    plt.savefig(eigen_filename, dpi=300)
    plt.show()

# Function to plot the mean face and eigenfaces with titlte
def plot_gallery_print(images, titles, h, w,name, n_row=1, n_col=3):
    """
    Plots a gallery of images for printing purposes.

    Parameters:
    images (list): List of images to display.
    titles (list): Titles for each subplot.
    h (int): Image height.
    w (int): Image width.
    name (str): Name for the output file.
    n_row (int): Number of rows in the plot.
    n_col (int): Number of columns in the plot.
    """

    plt.figure(figsize=(1.8 *2* n_col, 2.4*2 * n_row),)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.suptitle("The Digital Mirror: AI and the Evolution of Portraiture",fontsize=24)
    file_name="EigenFace/"+name+"/"+name+".png"    
    plt.savefig(file_name, dpi=300)
    plt.tight_layout() 
    plt.show()


### FUNCTION TO COMPARE TWO FACES USING PCA
def compare_faces(face1, face2):
    """
    Compares two faces using PCA by calculating the Euclidean distance.

    Parameters:
    face1 (ndarray): First face image data.
    face2 (ndarray): Second face image data.

    Returns:
    distance (float): Euclidean distance between the two faces in PCA space.
    """
    # Transform faces to PCA space
    face1_pca = pca.transform([face1])
    face2_pca = pca.transform([face2])

    # Compute the Euclidean distance between the two faces
    distance = np.linalg.norm(face1_pca - face2_pca)
    print(distance)
    return distance


# ## 1. Video Recording 
# 
# The camera will record for 10 seconds. Look at the video and follow the instructions. 

# In[8]:


# Example usage
images=record_video(name, duration=10)  # Record for 10 seconds


# ## EIGENFACES

# ## 2. Meanface
# 

# In[11]:


# Convert the list of images to a NumPy array
images = np.array(images)

# Extract the number of samples, and the height (h) and width (w) of each image
n_samples, h, w = images.shape

# Reshape the images into a 2D data matrix where each row is a flattened image
# The shape of X will be (n_samples, h * w), making it suitable for PCA
X = np.reshape(images, (n_samples, h * w))

# Determine the number of features (pixels per image) in the dataset
n_features = X.shape[1]

# Normalize the dataset by computing the mean of each feature (pixel) across all images
# This step helps to standardize the data for PCA
X_mean = X.mean(axis=0)

# Compute the mean face by reshaping the mean vector back to the original image dimensions
# This shows the average appearance of all the images in the dataset
mean_face = X_mean.reshape((h, w))


# ## 3. Subtracted Meanface

# In[13]:


# Center the dataset by subtracting the mean image from each image
# This creates a dataset where the mean of each feature (pixel) is zero
X_centered = X - X_mean


# ## 4. Principal Component Analysis

# In[15]:


# Perform Principal Component Analysis (PCA) to find the main modes of variation (eigenfaces)
# n_components: Number of principal components (eigenfaces) to compute
# svd_solver='randomized': A faster algorithm for large datasets
# whiten=True: Normalize the principal components (useful for subsequent learning tasks)
n_components = 200  # Number of eigenfaces to retain for dimensionality reduction
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_centered)

# Extract the principal components (eigenfaces) from the PCA model
# Each component is reshaped back to the original image dimensions (h, w)
# The eigenfaces capture the most significant variations among the dataset images
eigenfaces = pca.components_.reshape((n_components, h, w))


# ## 5. Plot the Eigenfaces
# ### Example Eigenfaces

# In[17]:


# Titles for the eigenfaces
eigenface_titles = [f"Eigenface {i + 1}" for i in range(eigenfaces.shape[0])]
# Plot the mean face and the first 18 eigenfaces
plot_gallery(np.vstack((mean_face[np.newaxis], eigenfaces[0:44:10])), ["Meanface"] + eigenface_titles[0:44:10], h, w,name)
plt.show()


# In[18]:


# Plot the mean face and the first 18 eigenfaces
plot_gallery_print(np.vstack((mean_face[np.newaxis], eigenfaces[[0,30]])), ["Meanface"] + eigenface_titles[0:34:28], h, w,name)
plt.show()


# In[19]:


plot_gallery(np.vstack((mean_face[np.newaxis], eigenfaces[0:23])), ["Meanface"] + eigenface_titles[0:23], h, w,name,4)


# ## 6. Reconstruct an Image
# The reconstruction allows us to see how well the reduced representation captures the original face information. If the reconstruction is close to the original, it indicates that the most important features have been retained. This also helps in assessing the effectiveness of the PCA dimensionality reduction.

# In[21]:


## Projecting faces onto the eigenface space
# The PCA transform method is used to project the centered face data onto the eigenface space.
# X_pca will contain the coordinates of the original images in the reduced dimensionality space (eigenface space).
# This reduces the data to a lower-dimensional representation, keeping only the most significant features.
X_pca = pca.transform(X_centered)

# Reconstruct some faces from their projection in the eigenface space
# The inverse_transform method projects the data back to the original space from the eigenface space.
# X_reconstructed contains the reconstructed images, which approximate the original images using only the selected principal components.
X_reconstructed = pca.inverse_transform(X_pca)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'xticks':[], 'yticks':[]})
ax[0].imshow(X[100].reshape((h, w)), cmap='gray')
ax[0].set_title("Original Face")
ax[1].imshow(X_reconstructed[100].reshape((h, w)), cmap='gray')
ax[1].set_title("Digital Mirror: Reconstructed Face")
Recon_name="EigenFace/"+name+"/"+name+"_Recon.png"
plt.suptitle("The Digital Mirror: AI and the Evolution of Portraiture",fontsize=24)
plt.tight_layout() 
fig.savefig(Recon_name, dpi=300)
plt.show()

