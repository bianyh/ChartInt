import pandas as pd
import os
os.chdir('/home/bianyuhan/chart2code/complex')
# sheet_url = "https://docs.google.com/spreadsheets/d/1ZxN8kLq9Hhf1nTjuc5epv6bLhicfPt-UofcBbEchLS4/edit?usp=sharing"

# sh = gc.open_by_url(sheet_url)
# worksheet = sh.worksheet('ColorAttributesInVisImages')
# df_original = pd.DataFrame(worksheet.get_all_records())

# sheet_url2 = "https://docs.google.com/spreadsheets/d/1uHhign4ZHmNhWCtpy2-5dQiYu44Kv2j5idFd3dR8Lgw/edit?usp=sharing"
# sh2 = gc.open_by_url(sheet_url2)
# worksheet2 = sh2.worksheet('HeerStone_colorNaming')
# color_df = pd.DataFrame(worksheet2.get_all_records())

# sheet_url3 = "https://docs.google.com/spreadsheets/d/19f3C1LBzplAUGMYhyK1Oat4UqInrCYpJ_iQNntepnzQ/edit?usp=sharing"
# sh3 = gc.open_by_url(sheet_url3)
# worksheet3 = sh3.worksheet('HeerStone_colorSimilarity')
# color_data = pd.DataFrame(worksheet3.get_all_records())

# print(df_original['Image name'])




# 获取图像并转换为OPENCV格式
import cv2
import os
import numpy as np
import math
import ast
from scipy.spatial.distance import cdist
# from google.colab.patches import cv2_imshow
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
import io
import urllib.request


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def url_to_cv2_image(url):
    resp = urllib.request.urlopen(url)
    image = Image.open(io.BytesIO(resp.read())).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def plot_and_save_image_from_url(url,figsize=(8, 6), dpi=100):
    """
    Loads an image from a URL, displays it using matplotlib, and saves it as a figure.

    Parameters:
    - url: str, image URL
    - save_path: str, file path to save
    - figsize: tuple, size of the figure
    - dpi: int, resolution of saved figure
    """

    # Load image from URL
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot and save
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('original_image.pdf', format='pdf', bbox_inches='tight')
    plt.show()



# 核心颜色分析逻辑 (Color Mapping)
# 这是代码的核心部分之一。它将图像中的每个像素映射到数据库中最近似的颜色名称。
def analyze_image_colors_and_names_lab_bkglist(image, color_names_df, bkg_colors):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab_image.reshape(-1, lab_image.shape[-1])

    unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)

    total_pixels = pixels.shape[0]
    percentages = (counts / total_pixels) * 100

    color_names_df['LAB(OpenCV)'] = color_names_df['LAB(OpenCV)'].apply(
        lambda x: np.array([int(num) for num in x.strip('[]').split()])
    )

    distances = cdist(unique_pixels, np.stack(color_names_df['LAB(OpenCV)'].values), metric='euclidean')

    closest_color_idxs = np.argmin(distances, axis=1)
    closest_colors = [color_names_df['Color_Name'][idx] for idx in closest_color_idxs]
    closest_colors_lab_values = [color_names_df['LAB(OpenCV)'].iloc[idx] for idx in closest_color_idxs]

    color_to_lab_values = {}
    color_to_counts = {}
    for color, lab_value, count in zip(closest_colors, closest_colors_lab_values, counts):
        if color in color_to_lab_values:
            color_to_lab_values[color].append(lab_value)
            color_to_counts[color] += count
        else:
            color_to_lab_values[color] = [lab_value]
            color_to_counts[color] = count

    avg_lab_values = {color: np.mean(labs, axis=0).astype(int).tolist() for color, labs in color_to_lab_values.items()}

    color_percentages = {}
    for color, percentage in zip(closest_colors, percentages):
        if color in color_percentages:
            color_percentages[color] += percentage
        else:
            color_percentages[color] = percentage

    sorted_indices = np.argsort(list(color_percentages.values()))[::-1]
    sorted_color_names_list = [list(color_percentages.keys())[i] for i in sorted_indices]
    sorted_color_percentages_list = [list(color_percentages.values())[i] for i in sorted_indices]
    sorted_color_counts = [color_to_counts[color] for color in sorted_color_names_list]

    sorted_avg_lab_values = [avg_lab_values[color] for color in sorted_color_names_list]

    binary_mask = np.zeros(pixels.shape[0], dtype=np.uint8)
    for color in bkg_colors:
        pixel_to_color_map = {tuple(pixel): color == color_name for pixel, color_name in zip(unique_pixels, closest_colors)}
        binary_mask += np.array([pixel_to_color_map[tuple(pixel)] for pixel in pixels], dtype=np.uint8)

    binary_mask = np.where(binary_mask > 0, 1, 0)

    binary_mask = binary_mask.reshape(lab_image.shape[:2])

    return sorted_color_names_list, sorted_color_percentages_list, sorted_avg_lab_values, sorted_color_counts, binary_mask


def apply_grabcut_with_binary_mask(img, binary_mask,foreground_path,background_path):
    # Convert binary mask to GrabCut mask
    grabcut_mask = np.where(binary_mask == 1, cv2.GC_FGD, cv2.GC_BGD).astype('uint8')

    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(img, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Modify the mask such that all 0 and 2 pixels are converted to the background
    mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')

    # Create the final masks
    foreground = np.zeros_like(img)
    foreground[mask2 == 1] = img[mask2 == 1]  # Keep original color for the foreground

    background = np.copy(img)
    background[mask2 == 1] = [255, 255, 255]  # Set foreground areas to white in the background image

    # Convert BGR images to RGB for matplotlib
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    # Display images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(foreground_rgb)
    plt.title('Foreground in black')
    plt.axis('off')

    plt.savefig(foreground_path)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(background_rgb)
    plt.title('Image Remove Background')
    plt.axis('off')

    plt.savefig(background_path)
    plt.show()




def image_no_background(img, save_path):
  img=img
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  plt.figure(figsize=(10, 5))
  plt.imshow(img_rgb)
  plt.title('Image (no background), same as original image')
  plt.axis('off')

  plt.savefig(save_path)
  plt.show()



def analyze_image_colors_and_names_nobkg(image, color_names_df, bkg_colors):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    pixels = lab_image.reshape(-1, lab_image.shape[-1])

    unique_pixels, inverse_indices, counts = np.unique(pixels, axis=0, return_counts=True, return_inverse=True)

    color_names_df['LAB(OpenCV)'] = color_names_df['LAB(OpenCV)'].apply(
        lambda x: np.array([int(num) for num in x.strip('[]').split()])
    )

    distances = cdist(unique_pixels, np.stack(color_names_df['LAB(OpenCV)'].values), metric='euclidean')

    closest_color_idxs = np.argmin(distances, axis=1)
    closest_colors = [color_names_df['Color_Name'][idx] for idx in closest_color_idxs]


    binary_mask = np.zeros(pixels.shape[0], dtype=np.uint8)
    for color in bkg_colors:
        pixel_to_color_map = {tuple(pixel): color == color_name for pixel, color_name in zip(unique_pixels, closest_colors)}
        binary_mask += np.array([pixel_to_color_map[tuple(pixel)] for pixel in pixels], dtype=np.uint8)

    binary_mask = np.where(binary_mask > 0, 1, 0)
    binary_mask = binary_mask.reshape(lab_image.shape[:2])

    foreground_pixels = pixels[binary_mask.flatten() == 0]

    unique_foreground_pixels, foreground_counts = np.unique(foreground_pixels, axis=0, return_counts=True)

    total_foreground_pixels = foreground_pixels.shape[0]
    foreground_percentages = (foreground_counts / total_foreground_pixels) * 100

    foreground_distances = cdist(unique_foreground_pixels, np.stack(color_names_df['LAB(OpenCV)'].values), metric='euclidean')

    foreground_closest_color_idxs = np.argmin(foreground_distances, axis=1)
    foreground_closest_colors = [color_names_df['Color_Name'][idx] for idx in foreground_closest_color_idxs]
    foreground_closest_colors_lab_values = [color_names_df['LAB(OpenCV)'].iloc[idx] for idx in foreground_closest_color_idxs]

    color_to_lab_values = {}
    color_to_counts = {}
    for color, lab_value, count in zip(foreground_closest_colors, foreground_closest_colors_lab_values, foreground_counts):
        if color in color_to_lab_values:
            color_to_lab_values[color].append(lab_value)
            color_to_counts[color] += count
        else:
            color_to_lab_values[color] = [lab_value]
            color_to_counts[color] = count

    avg_lab_values = {color: np.mean(labs, axis=0).astype(int).tolist() for color, labs in color_to_lab_values.items()}

    color_percentages = {}
    for color, percentage in zip(foreground_closest_colors, foreground_percentages):
        if color in color_percentages:
            color_percentages[color] += percentage
        else:
            color_percentages[color] = percentage

    sorted_indices = np.argsort(list(color_percentages.values()))[::-1]
    sorted_color_names_list = [list(color_percentages.keys())[i] for i in sorted_indices]
    sorted_color_percentages_list = [list(color_percentages.values())[i] for i in sorted_indices]
    sorted_color_counts = [color_to_counts[color] for color in sorted_color_names_list]

    sorted_avg_lab_values = [avg_lab_values[color] for color in sorted_color_names_list]

    return sorted_color_names_list, sorted_color_percentages_list, sorted_avg_lab_values, sorted_color_counts, binary_mask


def analyze_image_colors_and_names_lab_array(image, color_names_df):

    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Reshape the image data into a one-dimensional array
    pixels = lab_image.reshape(-1, 3)

    # Find unique pixels values and their indices and counts
    unique_pixels, inverse_indices, counts = np.unique(pixels, axis=0, return_counts=True, return_inverse=True)

    # Calculate the percentage of each unique pixel
    total_pixels = pixels.shape[0]
    percentages = (counts / total_pixels) * 100

    # Convert LAB(OpenCV) strings in dataframe to numpy array of integers
    color_names_df['LAB(OpenCV)'] = color_names_df['LAB(OpenCV)'].apply(
        lambda x: np.array([int(num) for num in x.strip('[]').split()], dtype=np.int32)
    )

    # Compute distances from unique pixels to all colors in the dictionary
    distances = cdist(unique_pixels, np.stack(color_names_df['LAB(OpenCV)'].values), metric='euclidean')

    # Find the closest color name and LAB value for each unique pixel
    closest_color_idxs = np.argmin(distances, axis=1)
    closest_colors = color_names_df['Color_Name'].iloc[closest_color_idxs].to_numpy()
    closest_colors_lab_values = np.stack(color_names_df['LAB(OpenCV)'].iloc[closest_color_idxs].to_numpy())

    # Aggregate percentages by color name and average LAB values
    color_percentages = {}
    color_to_lab_values = {}
    for idx, percentage in enumerate(percentages):
        color = closest_colors[idx]
        lab_value = closest_colors_lab_values[idx]
        if color in color_percentages:
            color_percentages[color] += percentage
            color_to_lab_values[color].append(lab_value)
        else:
            color_percentages[color] = percentage
            color_to_lab_values[color] = [lab_value]

    # Compute average LAB values for each color
    avg_lab_values = {color: np.mean(labs, axis=0).astype(int).tolist() for color, labs in color_to_lab_values.items()}

    # Sort by percentage
    sorted_indices = np.argsort(list(color_percentages.values()))[::-1]
    sorted_color_names = [list(color_percentages.keys())[i] for i in sorted_indices]
    sorted_percentages = [list(color_percentages.values())[i] for i in sorted_indices]
    sorted_avg_lab_values = [avg_lab_values[color] for color in sorted_color_names]

    # Create an array of the same shape as the original image with closest color names
    color_map = closest_colors[inverse_indices].reshape(image.shape[:2])

    return sorted_color_names, sorted_percentages, sorted_avg_lab_values, color_map


# 聚类分析 (K-Means)
# 针对“不连续颜色”（Categorical/Discrete）的图像，代码使用加权 K-Means 算法来提取主色调。
def weighted_k_means(lab_values, counts, n_clusters=5):
    log_counts = np.log1p(counts)

    lab_values = np.array(lab_values)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=5000)

    kmeans.fit(lab_values, sample_weight=log_counts)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    cluster_weights = np.zeros(n_clusters)
    for label, weight in zip(labels, log_counts):
        cluster_weights[label] += weight

    total_weight = np.sum(cluster_weights)
    percentages = (cluster_weights / total_weight) * 100

    percentages = np.around(percentages, 2)
    percentages = [f"{p}%" for p in percentages]

    return centroids, labels, percentages

def kmean_color_name(lab_values,color_names_df):

    kmeans_lab=np.array(lab_values)

    color_names_df['LAB(OpenCV)'] = color_names_df['LAB(OpenCV)'].apply(
        lambda x: np.array([int(num) for num in x.strip('[]').split()])
    )

    distances = cdist(kmeans_lab, np.stack(color_names_df['LAB(OpenCV)'].values), metric='euclidean')

    closest_color_idxs = np.argmin(distances, axis=1)
    closest_colors = [color_names_df['Color_Name'][idx] for idx in closest_color_idxs]

    return closest_colors


from collections import defaultdict

def group_colors_by_cluster_name(color_names, kmeans_labels, kmeans_name):
    """
    Groups color names by their KMeans-assigned cluster name.

    Parameters:
    - color_names: list of original color names
    - kmeans_labels: list of cluster indices assigned to each color
    - kmeans_name: list of cluster names by cluster index

    Returns:
    - dict: {cluster_name: [color_name1, color_name2, ...]}
    """
    cluster_to_colors = defaultdict(list)
    for color_name, label in zip(color_names, kmeans_labels):
        cluster_name = kmeans_name[label]
        cluster_to_colors[cluster_name].append(color_name)

    return dict(cluster_to_colors)


def apply_kmeans_to_foreground(sorted_color_names,binary_mask, color_map, image, LAB_centroids, kmeans_labels, kmeans_percentages):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    output_array = np.empty(binary_mask.shape + (3,), dtype=np.uint8)

    color_to_centroid = {}

    for color_name, label in zip(sorted_color_names, kmeans_labels):
        if color_name not in color_to_centroid:
            color_to_centroid[color_name] = label

    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if binary_mask[y, x] == 0:
                color_name = color_map[y, x]
                centroid_idx = color_to_centroid.get(color_name)
                if centroid_idx is not None:
                    output_array[y, x] = LAB_centroids[centroid_idx]

            else:
                output_array[y, x] = lab_image[y, x]

    return output_array


def apply_kmeans_to_foreground2(sorted_color_names, binary_mask, color_map, image, LAB_centroids, kmeans_labels, kmeans_percentages):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    output_array = np.empty(binary_mask.shape + (3,), dtype=np.uint8)
    color_to_centroid = {}

    # Map each color name to its centroid label
    for color_name, label in zip(sorted_color_names, kmeans_labels):
        if color_name not in color_to_centroid:
            color_to_centroid[color_name] = label

    # Initialize dictionaries to count occurrences of each LAB value in each centroid
    centroid_lab_counts = {i: {} for i in range(len(LAB_centroids))}

    # Count occurrences of each LAB value in each centroid
    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if binary_mask[y, x] == 0:
                color_name = color_map[y, x]
                centroid_idx = color_to_centroid.get(color_name)
                if centroid_idx is not None:
                    lab_value = tuple(lab_image[y, x])
                    if lab_value in centroid_lab_counts[centroid_idx]:
                        centroid_lab_counts[centroid_idx][lab_value] += 1
                    else:
                        centroid_lab_counts[centroid_idx][lab_value] = 1

    # Find the most frequent LAB value for each centroid
    most_frequent_lab = {}
    most_frequent_lab_list = []
    for centroid_idx, lab_count in centroid_lab_counts.items():
        if lab_count:
            most_frequent_value = max(lab_count, key=lab_count.get)
            most_frequent_lab[centroid_idx] = most_frequent_value
            most_frequent_lab_list.append(most_frequent_value)
        else:
            most_frequent_lab[centroid_idx] = LAB_centroids[centroid_idx]
            most_frequent_lab_list.append(LAB_centroids[centroid_idx])

    # Replace colors in the output array
    for y in range(binary_mask.shape[0]):
        for x in range(binary_mask.shape[1]):
            if binary_mask[y, x] == 0:
                color_name = color_map[y, x]
                centroid_idx = color_to_centroid.get(color_name)
                if centroid_idx is not None:
                    output_array[y, x] = most_frequent_lab[centroid_idx]
            else:
                output_array[y, x] = lab_image[y, x]

    return output_array, most_frequent_lab_list

def LAB_3D_kmean(image, LAB_values, LAB_counts, LAB_centroids, kmeans_labels, kmeans_percentages, kmeans_array,centroids_path,kmeans_path,kmeans_name):
    # Convert LAB values and centroids to RGB for plotting
    LAB_array = np.array(LAB_values).astype(np.uint8)
    LAB_array_reshaped = LAB_array.reshape((1, len(LAB_values), 3))
    RGB_array = cv2.cvtColor(LAB_array_reshaped, cv2.COLOR_LAB2BGR)
    RGB_array = cv2.cvtColor(RGB_array, cv2.COLOR_BGR2RGB)

    LAB_centroids_array = np.array(LAB_centroids).astype(np.uint8)
    LAB_centroids_array_reshaped = LAB_centroids_array.reshape((1, len(LAB_centroids), 3))
    RGB_centroids_array = cv2.cvtColor(LAB_centroids_array_reshaped, cv2.COLOR_LAB2BGR)
    RGB_centroids_array = cv2.cvtColor(RGB_centroids_array, cv2.COLOR_BGR2RGB)

    log_counts = np.log(LAB_counts)
    scaled_log_counts = 20 * log_counts

    # Map each label to its centroid color
    point_colors = RGB_centroids_array.reshape(len(LAB_centroids), 3)[kmeans_labels] / 255.0

    # Parse percentages for centroids and calculate their sizes
    percentage_sizes = [float(p.strip('%')) for p in kmeans_percentages]
    centroid_sizes = [size * 10 for size in percentage_sizes]  # Adjust multiplier to scale the sizes visually

    # Create a colormap for centroids
    cmap = ListedColormap(plt.colormaps['tab20'].colors[:len(LAB_centroids)])
    centroid_colors = cmap.colors

    # Plot for 3D scatter plot
    fig_3d = plt.figure(figsize=(15, 10))
    ax2 = fig_3d.add_subplot(111, projection='3d')
    ax2.scatter(xs=LAB_array[:, 1], ys=LAB_array[:, 2], zs=LAB_array[:, 0], s=scaled_log_counts, c=point_colors, lw=0)

    # Scatter plot for centroids with unique colors
    if len(LAB_centroids) <= 20:
        for i, centroid in enumerate(LAB_centroids):
            face_color = centroid_colors[i]
            edge_color = point_colors[i]
            cluster_name=kmeans_name[i]
            ax2.scatter(xs=centroid[1], ys=centroid[2], zs=centroid[0], s=centroid_sizes[i],
                        facecolors=[face_color], edgecolors=[edge_color], marker='*', label=f'Centroid {i+1}:{cluster_name}')
    else:
        for i, centroid in enumerate(LAB_centroids):
            face_color = 'black'
            edge_color = point_colors[i]
            cluster_name=kmeans_name[i]
            ax2.scatter(xs=centroid[1], ys=centroid[2], zs=centroid[0], s=centroid_sizes[i],
                        facecolors=[face_color], edgecolors=[edge_color], marker='*', label=f'Centroid {i+1}:{cluster_name}')

    ax2.set_xlabel('A')
    ax2.set_ylabel('B')
    ax2.set_zlabel('L')
    # ax2.set_title('Colors and Centroids in OpenCV LAB Color Space after K-means')

    # Add legend
    ax2.legend(loc='upper right')
    # plt.savefig(centroids_path)
    plt.show()

    # Plot for K-means result with the same size as the original image
    height, width, _ = image.shape
    print(height, width)
    fig_output = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax3 = fig_output.add_subplot(111)
    output_rgb = cv2.cvtColor(kmeans_array, cv2.COLOR_Lab2RGB)
    ax3.imshow(output_rgb)
    ax3.axis('off')  # Hide axes for output array plot
    # ax3.set_title('Replot the image using K-means result')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax3.set_position([0, 0, 1, 1])
    # plt.savefig(kmeans_path, bbox_inches='tight',pad_inches=0)

    plt.show()


def LAB_3D_kmean_standard(image, LAB_values, LAB_counts, LAB_centroids,
                          kmeans_labels, kmeans_percentages, kmeans_array,
                          centroids_path, kmeans_path, kmeans_name):
    """
    Plots a 3D scatter of LAB clusters in standard LAB space.
    """

    # Convert LAB points and centroids to standard LAB
    standard_LAB = np.array([opencv_lab_to_standard_lab(lab) for lab in LAB_values])
    standard_centroids = np.array([opencv_lab_to_standard_lab(lab) for lab in LAB_centroids])

    # Convert centroids to RGB using standard LAB for labeling
    centroid_rgb = np.array([standard_lab_to_rgb(lab) for lab in standard_centroids])

    # Determine color for each point based on its assigned centroid
    point_colors = centroid_rgb[kmeans_labels]

    # Scale point size
    log_counts = np.log(LAB_counts)
    scaled_log_counts = 20 * log_counts

    # Percent size for centroids
    percentage_sizes = [float(p.strip('%')) for p in kmeans_percentages]
    centroid_sizes = [size * 10 for size in percentage_sizes]

    # Use color map for centroid markers
    cmap = ListedColormap(plt.colormaps['tab20'].colors[:len(LAB_centroids)])
    centroid_face_colors = cmap.colors

    # Plotting
    fig_3d = plt.figure(figsize=(15, 10))
    ax2 = fig_3d.add_subplot(111, projection='3d')

    # Scatter actual LAB points
    ax2.scatter(xs=standard_LAB[:, 1], ys=standard_LAB[:, 2], zs=standard_LAB[:, 0],
                s=scaled_log_counts, c=point_colors, lw=0)

    # Scatter centroids
    for i, centroid in enumerate(standard_centroids):
        cluster_name = kmeans_name[i]
        face_color = centroid_face_colors[i] if len(standard_centroids) <= 20 else 'black'
        edge_color = centroid_rgb[i]
        ax2.scatter(xs=centroid[1], ys=centroid[2], zs=centroid[0],
                    s=centroid_sizes[i],
                    facecolors=[face_color], edgecolors=[edge_color],
                    marker='*', label=f'Centroid {i+1}: {cluster_name}')

    ax2.set_xlabel('a*')
    ax2.set_ylabel('b*')
    ax2.set_zlabel('L*')
    ax2.set_zlim(0, 100)
    ax2.view_init(elev=40, azim=150)
    plt.tight_layout()
    plt.legend()
    plt.savefig('discrete_color_keman.pdf',format='pdf',bbox_inches='tight')
    plt.show()

    # Add legend
    ax2.legend(loc='upper right')
    # plt.savefig(centroids_path)
    plt.show()

    # Plot for K-means result with the same size as the original image
    height, width, _ = image.shape
    print(height, width)
    fig_output = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax3 = fig_output.add_subplot(111)
    output_rgb = cv2.cvtColor(kmeans_array, cv2.COLOR_Lab2RGB)
    ax3.imshow(output_rgb)
    ax3.axis('off')  # Hide axes for output array plot
    # ax3.set_title('Replot the image using K-means result')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax3.set_position([0, 0, 1, 1])
    # plt.savefig(kmeans_path, bbox_inches='tight',pad_inches=0)
    plt.savefig('discrete_color_replot.pdf',format='pdf',bbox_inches='tight')
    plt.show()


def check_lab_values(color_names, lab_values):
    bw = True
    color_colors = []
    color_lab_values = []
    bw_colors = []
    bw_lab_values = []
    for color_name, lab in zip(color_names, lab_values):
        L, a, b = lab
        if not (120 <= a <= 140 and 120 <= b <= 140):
            bw = False
            color_colors.append(color_name)
            color_lab_values.append(lab)
        else:
            bw_colors.append(color_name)
            bw_lab_values.append(lab)

    return bw, color_colors, color_lab_values,bw_colors,bw_lab_values

def check_lab_values_withthreshold(color_names, lab_values,threshold):
    bw = True
    color_colors = []
    color_lab_values = []
    bw_colors = []
    bw_lab_values = []
    for color_name, lab in zip(color_names, lab_values):
        L, a, b = lab
        if not (128-threshold <= a <= 128+threshold and 128-threshold <= b <= 128+threshold):
            bw = False
            color_colors.append(color_name)
            color_lab_values.append(lab)
        else:
            bw_colors.append(color_name)
            bw_lab_values.append(lab)

    return bw, color_colors, color_lab_values,bw_colors,bw_lab_values

def get_subcolor_percentages(color_names, percentages, subcolor_names):
    color_to_percentage = dict(zip(color_names, percentages))
    subcolor_percentages = [color_to_percentage[color] for color in subcolor_names if color in color_to_percentage]

    return subcolor_percentages

def get_subcolor_indices(color_names, percentages, subcolor_names):
    color_to_index = {color: index for index, color in enumerate(color_names)}
    subcolor_indices = [color_to_index[color] for color in subcolor_names if color in color_to_index]

    return subcolor_indices

def remove_indices_from_list(original_list, indices_to_remove):
    result_list = [value for index, value in enumerate(original_list) if index not in indices_to_remove]
    return result_list


def merge_similar_colors_final_wbw(colors_all, percentages_all, avg_lab_values_all, color_map, similar_colors_dict,image, binary_mask,colors_bw, percentages_bw, avg_lab_values_bw,percentages_idx_bw):
    color_to_lab = dict(zip(colors_all, avg_lab_values_all))  # Map from color name to LAB values

    colors=[color for color in colors_all if color not in colors_bw]
    percentages = remove_indices_from_list(percentages_all, percentages_idx_bw)

    avg_lab_values=[lab for lab in avg_lab_values_all if lab not in avg_lab_values_bw]
    # Start the first pointer at the last element
    i = len(colors) - 1

    while i > 0:  # Continue until the first pointer reaches the start
        has_merged = False
        j = i - 1  # Second pointer starts just before the first pointer

        while j >= 0:
            # Check if the color at the second pointer has the first pointer's color in its similar list
            # and ensure neither color is in the exclude list
            if colors[i] in similar_colors_dict.get(colors[j], []):
                # Merge the colors and percentages
                new_color = colors[j]  # New color name is the name at the second pointer
                new_percentage = percentages[i] + percentages[j]  # Sum the percentages
                new_lab_value = color_to_lab[new_color]  # Keep the LAB value of the more general color

                # Update the color_map before removing the colors from the lists
                color_map = np.where(color_map == colors[i], new_color, color_map)
                color_map = np.where(color_map == colors[j], new_color, color_map)

                # Remove the original entries
                colors.pop(i)
                percentages.pop(i)
                avg_lab_values.pop(i)
                colors.pop(j)
                percentages.pop(j)
                avg_lab_values.pop(j)

                # Insert the new merged color and percentage and update LAB values
                colors.append(new_color)
                percentages.append(new_percentage)
                avg_lab_values.append(new_lab_value)

                # Update the color to LAB mapping
                color_to_lab[new_color] = new_lab_value

                # Re-sort the lists by percentage in descending order
                combined = sorted(zip(percentages, colors, avg_lab_values), reverse=True, key=lambda x: x[0])
                percentages, colors, avg_lab_values = zip(*combined)
                percentages = list(percentages)
                colors = list(colors)
                avg_lab_values = list(avg_lab_values)

                # Since we have merged, reset the pointers based on new list length
                i = len(colors) - 1
                has_merged = True
                break
            j -= 1

        if not has_merged:
            i -= 1  # Move the first pointer one position back if no merge happened

    # Convert color_map from color names to corresponding LAB values
    lab_color_map = np.empty(color_map.shape, dtype=object)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    for index, name in np.ndenumerate(color_map):
        if binary_mask[index] == 0:  # Foreground pixel
            lab_color_map[index] = color_to_lab[name]
        else:  # Background pixel
            lab_color_map[index] = lab_image[index]

    colors+=colors_bw
    percentages+=percentages_bw
    avg_lab_values+=avg_lab_values_bw

    return colors, percentages, avg_lab_values, lab_color_map,color_map


def generate_lookup_table(color_names, lab_values, threshold=25):
    sorted_colors_labs = sorted(zip(color_names, lab_values), key=lambda x: x[1][0])
    sorted_color_names, sorted_lab_values = zip(*sorted_colors_labs)

    checked = [False] * len(sorted_color_names)
    lookup_table = {}

    for i in range(len(sorted_color_names)):
        if not checked[i]:
            lookup_table[sorted_color_names[i]] = []
            for j in range(i + 1, len(sorted_color_names)):
                if not checked[j]:
                    if abs(sorted_lab_values[i][0] - sorted_lab_values[j][0]) < threshold:
                        lookup_table[sorted_color_names[i]].append(sorted_color_names[j])
                        checked[j] = True
            checked[i] = True

    for color in sorted_color_names:
        if color not in lookup_table:
            lookup_table[color] = []

    return lookup_table


from sklearn.cluster import DBSCAN
import numpy as np

def generate_lookup_table2(color_names, lab_values, threshold=14):
    """
    Generate a lookup table grouping similar colors using DBSCAN clustering
    based on Euclidean distance in LAB space.

    Parameters:
    - color_names: List of color names.
    - lab_values: List of corresponding LAB values, each as (L, a, b).
    - threshold: Maximum Euclidean distance in LAB space to consider colors similar (DBSCAN eps).

    Returns:
    - A dictionary where each key is a color name, and the value is a list of similar color names
      in the same cluster (excluding itself).
    """

    # Convert LAB values to a numpy array
    lab_array = np.array(lab_values)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='euclidean')
    labels = clustering.fit_predict(lab_array)

    # Build the lookup table
    lookup_table = {}
    for label in set(labels):
        # Find all indices in the current cluster
        indices = [i for i, l in enumerate(labels) if l == label]
        group_names = [color_names[i] for i in indices]
        for i in indices:
            # Map each color to other colors in the same cluster (excluding itself)
            lookup_table[color_names[i]] = [name for name in group_names if name != color_names[i]]

    return lookup_table

import math

def generate_lookup_table3(color_names, lab_values, threshold=14):
    """
    Group similar colors based on adjacent LAB Euclidean distance with hierarchical merging.

    Parameters:
    - color_names: List of color names.
    - lab_values: List of corresponding LAB values, each as (L, a, b).
    - threshold: Maximum Euclidean distance in LAB space to consider two colors similar.

    Returns:
    - A dictionary {final_color_name: [list of similar color names excluding itself]}
    """

    # Sort by L value descending (bright to dark)
    sorted_colors_labs = sorted(zip(color_names, lab_values), key=lambda x: -x[1][0])
    sorted_color_names, sorted_lab_values = zip(*sorted_colors_labs)

    # Initialize parent mapping: each color points to itself
    parent = {name: name for name in sorted_color_names}

    # Helper: find final parent with path compression
    def find(name):
        if parent[name] != name:
            parent[name] = find(parent[name])
        return parent[name]

    names = list(sorted_color_names)
    labs = list(sorted_lab_values)
    merged = True

    while merged:
        merged = False
        min_dist = float('inf')
        min_index = -1

        # Find closest adjacent pair
        for i in range(len(names) - 1):
            dist = math.sqrt(
                (labs[i][0] - labs[i + 1][0]) ** 2 +
                (labs[i][1] - labs[i + 1][1]) ** 2 +
                (labs[i][2] - labs[i + 1][2]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # If closest distance ≤ threshold, merge
        if min_dist <= threshold:
            high_L_name = names[min_index]      # Larger L value
            low_L_name = names[min_index + 1]   # Smaller L value

            # Merge: lower L color points to higher L color's final parent
            parent[find(low_L_name)] = find(high_L_name)

            # Remove merged color from the active list
            names.pop(min_index + 1)
            labs.pop(min_index + 1)
            merged = True

    # Prepare output: {final_color: [merged color list excluding itself]}
    result = {}
    for name in sorted_color_names:
        final_parent = find(name)
        if final_parent not in result:
            result[final_parent] = []

    for name in sorted_color_names:
        final_parent = find(name)
        if name != final_parent:
            result[final_parent].append(name)

    return result

def merge_and_convert_colors(lookup_table, color_names, lab_values, color_matrix):
    color_to_lab = dict(zip(color_names, lab_values))

    merged_color_matrix = np.copy(color_matrix)
    unique_colors, unique_counts = np.unique(merged_color_matrix, return_counts=True)

    for key, values in lookup_table.items():
        for value in values:
            merged_color_matrix[merged_color_matrix == value] = key
    unique_colors, unique_counts = np.unique(merged_color_matrix, return_counts=True)

    lab_matrix = np.empty(merged_color_matrix.shape, dtype=object)

    for i in range(merged_color_matrix.shape[0]):
        for j in range(merged_color_matrix.shape[1]):
            color_name = merged_color_matrix[i, j]
            lab_matrix[i, j] = list(color_to_lab[color_name])

    unique_colors, unique_counts = np.unique(merged_color_matrix, return_counts=True)
    total_pixels = merged_color_matrix.size
    unique_percentages = (unique_counts / total_pixels) * 100

    return merged_color_matrix,lab_matrix,unique_colors,unique_percentages

def merge_and_convert_colors_with_mask(lookup_table, color_names, lab_values, color_matrix, binary_mask, image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    color_to_lab = dict(zip(color_names, lab_values))

    merged_color_matrix = np.copy(color_matrix)

    for key, values in lookup_table.items():
        for value in values:
            merged_color_matrix[merged_color_matrix == value] = key

    unique_colors, unique_counts = np.unique(merged_color_matrix, return_counts=True)

    lab_matrix = np.empty(merged_color_matrix.shape, dtype=object)

    for (i, j), color_name in np.ndenumerate(merged_color_matrix):
        if binary_mask[i, j] == 0:
            lab_matrix[i, j] = list(color_to_lab[color_name])
        else:
            lab_matrix[i, j] = list(lab_image[i, j])


    foreground_colors = merged_color_matrix[binary_mask == 0]
    unique_colors, unique_counts = np.unique(foreground_colors, return_counts=True)

    total_foreground_pixels = foreground_colors.size
    unique_percentages = (unique_counts / total_foreground_pixels) * 100

    return merged_color_matrix, lab_matrix,unique_colors,unique_percentages


def merge_similar_colors_nobw(colors, percentages, avg_lab_values, color_map, similar_colors_dict):
    color_to_lab = dict(zip(colors, avg_lab_values))  # Map from color name to LAB values
    i = len(colors) - 1  # Start the first pointer at the last element

    while i > 0:  # Continue until the first pointer reaches the start
        has_merged = False
        j = i - 1  # Second pointer starts just before the first pointer

        while j >= 0:
            # Check if the color at the second pointer has the first pointer's color in its similar list
            # and ensure neither color is in the exclude list
            if colors[i] in similar_colors_dict.get(colors[j], []):
                # Merge the colors and percentages
                new_color = colors[j]  # New color name is the name at the second pointer
                new_percentage = percentages[i] + percentages[j]  # Sum the percentages
                new_lab_value = color_to_lab[new_color]  # Keep the LAB value of the more general color

                # Update the color_map before removing the colors from the lists
                color_map = np.where(color_map == colors[i], new_color, color_map)
                color_map = np.where(color_map == colors[j], new_color, color_map)

                # Remove the original entries
                colors.pop(i)
                percentages.pop(i)
                avg_lab_values.pop(i)
                colors.pop(j)
                percentages.pop(j)
                avg_lab_values.pop(j)

                # Insert the new merged color and percentage and update LAB values
                colors.append(new_color)
                percentages.append(new_percentage)
                avg_lab_values.append(new_lab_value)

                # Update the color to LAB mapping
                color_to_lab[new_color] = new_lab_value

                # Re-sort the lists by percentage in descending order
                combined = sorted(zip(percentages, colors, avg_lab_values), reverse=True, key=lambda x: x[0])
                percentages, colors, avg_lab_values = zip(*combined)
                percentages = list(percentages)
                colors = list(colors)
                avg_lab_values = list(avg_lab_values)

                # Since we have merged, reset the pointers based on new list length
                i = len(colors) - 1
                has_merged = True
                break
            j -= 1

        if not has_merged:
            i -= 1  # Move the first pointer one position back if no merge happened

    # Convert color_map from color names to corresponding LAB values
    lab_color_map = np.empty(color_map.shape, dtype=object)
    for index, name in np.ndenumerate(color_map):
        lab_color_map[index] = color_to_lab[name]

    return color_map,colors, percentages, avg_lab_values, lab_color_map


def merge_similar_colors_nobackground(colors_all, percentages_all, avg_lab_values_all, color_map, similar_colors_dict,colors_bw, percentages_bw, avg_lab_values_bw,percentages_idx_bw):
    color_to_lab = dict(zip(colors_all, avg_lab_values_all))  # Map from color name to LAB values

    colors=[color for color in colors_all if color not in colors_bw]
    percentages = remove_indices_from_list(percentages_all, percentages_idx_bw)
    avg_lab_values=[lab for lab in avg_lab_values_all if lab not in avg_lab_values_bw]

    i = len(colors) - 1  # Start the first pointer at the last element
    unique_colors, unique_counts = np.unique(color_map, return_counts=True)

    while i > 0:  # Continue until the first pointer reaches the start
        has_merged = False
        j = i - 1  # Second pointer starts just before the first pointer

        while j >= 0:
            # Check if the color at the second pointer has the first pointer's color in its similar list
            if colors[i] in similar_colors_dict.get(colors[j], []):
                # Merge the colors and percentages
                new_color = colors[j]  # New color name is the name at the second pointer
                new_percentage = percentages[i] + percentages[j]  # Sum the percentages
                new_lab_value = color_to_lab[new_color]  # Keep the LAB value of the more general color

                # Update the color_map before removing the colors from the lists
                color_map = np.where(color_map == colors[i], new_color, color_map)
                color_map = np.where(color_map == colors[j], new_color, color_map)

                # Remove the original entries
                colors.pop(i)
                percentages.pop(i)
                avg_lab_values.pop(i)
                colors.pop(j)
                percentages.pop(j)
                avg_lab_values.pop(j)

                # Insert the new merged color and percentage and update LAB values
                colors.append(new_color)
                percentages.append(new_percentage)
                avg_lab_values.append(new_lab_value)

                # Update the color to LAB mapping
                color_to_lab[new_color] = new_lab_value

                # Re-sort the lists by percentage in descending order
                combined = sorted(zip(percentages, colors, avg_lab_values), reverse=True, key=lambda x: x[0])
                percentages, colors, avg_lab_values = zip(*combined)
                percentages = list(percentages)
                colors = list(colors)
                avg_lab_values = list(avg_lab_values)

                # Since we have merged, reset the pointers based on new list length
                i = len(colors) - 1
                has_merged = True
                break
            j -= 1

        if not has_merged:
            i -= 1  # Move the first pointer one position back if no merge happened
    unique_colors, unique_counts = np.unique(color_map, return_counts=True)

    # Convert color_map from color names to corresponding LAB values
    lab_color_map = np.empty(color_map.shape, dtype=object)
    for index, name in np.ndenumerate(color_map):
        lab_color_map[index] = color_to_lab[name]


    colors+=colors_bw
    percentages+=percentages_bw
    avg_lab_values+=avg_lab_values_bw

    return colors, percentages, avg_lab_values, lab_color_map,color_map


def display_lab_image(image, lab_color_map, merge_path,difference_path):
    height, width = len(lab_color_map), len(lab_color_map[0])

    lab_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            lab_image[i, j] = lab_color_map[i][j]

    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_diff_signed = lab_image.astype(int) - original_lab.astype(int)

    lab_diff_signed = np.clip(lab_diff_signed, -128, 127) + 128
    lab_diff_signed = lab_diff_signed.astype(np.uint8)

    lab_diff_bgr = cv2.cvtColor(lab_diff_signed, cv2.COLOR_LAB2BGR)
    rgb_diff = cv2.cvtColor(lab_diff_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title('Image after merging similar colors')
    plt.savefig(merge_path)

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(rgb_diff)
    plt.axis('off')
    plt.title('LAB Signed Difference Image')

    plt.savefig(difference_path)
    plt.show()


def plot_lab_matrix_3d_scatter(lab_color_map):
    height, width = len(lab_color_map), len(lab_color_map[0])

    lab_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            lab_image[i, j] = lab_color_map[i][j]

    bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Flatten the LAB matrix for scatter plotting
    L_channel = lab_image[..., 0].flatten()
    A_channel = lab_image[..., 1].flatten()
    B_channel = lab_image[..., 2].flatten()

    # Normalize LAB values for visualization
    scatter_colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize RGB for scatter plot colors

    # Plotting the 3D scatter plot in LAB space
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(A_channel, B_channel, L_channel,
                    c=scatter_colors, s=30, alpha=0.7)

    # Set labels and title
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_zlabel("L",)

    # Show the plot

    plt.show()



def plot_color_bar_chart(sorted_color_names, sorted_color_percentages, sorted_lab_values,savpath):
    """
    Plot a bar chart of color composition in the image using RGB values converted from LAB(OpenCV).

    Parameters:
    - sorted_color_names: List of color names sorted by percentage (descending).
    - sorted_color_percentages: Corresponding list of percentages for each color.
    - sorted_lab_values: Corresponding list of average LAB(OpenCV) values for each color.
    """
    # Convert LAB(OpenCV) values to RGB for plotting
    lab_array = np.array(sorted_lab_values, dtype=np.uint8).reshape(-1, 1, 3)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB).reshape(-1, 3)

    # Normalize RGB values to [0, 1] for matplotlib
    rgb_colors = rgb_array / 255.0

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(range(len(sorted_color_names)), sorted_color_percentages, color=rgb_colors)

    # Annotate each bar with the RGB value and color name
    for i, bar in enumerate(bars):
        rgb_str = f'RGB: {rgb_array[i][0]}, {rgb_array[i][1]}, {rgb_array[i][2]}'
        text = f"{sorted_color_names[i]} ({rgb_str})"
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                text, va='center', fontsize=12)

    # Axis and labels
    ax.set_yticks(range(len(sorted_color_names)))
    ax.set_yticklabels(sorted_color_names)
    ax.invert_yaxis()  # Highest percentage on top
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Color Composition in Image')

    plt.tight_layout()
    plt.savefig(savpath, format='pdf', bbox_inches='tight')
    plt.show()


def visualize_color_name_map(initial_color_array, sorted_color_names, sorted_lab_values):
    """
    Visualize the color-mapped image using color names and their RGB approximations.

    Parameters:
    - initial_color_array: 2D array of color names (same height & width as original image).
    - sorted_color_names: List of color names used in the image.
    - sorted_lab_values: Average LAB(OpenCV) values for each color name.
    """
    # Step 1: Create color name to RGB dictionary
    lab_array = np.array(sorted_lab_values, dtype=np.uint8).reshape(-1, 1, 3)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    rgb_colors = {name: tuple(rgb_array[i]) for i, name in enumerate(sorted_color_names)}

    # Step 2: Convert color name array to RGB image
    height, width = initial_color_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for name, rgb in rgb_colors.items():
        mask = initial_color_array == name
        rgb_image[mask] = rgb

    # Step 3: Show the image with a legend of color names
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title("Image with Color Name Mapping (Approximate RGB)")

    # Step 4: Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(rgb)/255.0, edgecolor='black', label=name)
                       for name, rgb in rgb_colors.items()]
    # plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
    #            ncol=min(len(legend_elements), 5), frameon=False)

    plt.legend(handles=legend_elements,
           loc='center left',
           bbox_to_anchor=(1.02, 0.5),
           ncol=4,
           frameon=False)

    plt.tight_layout()

    plt.savefig('continuous_image_ori_map.pdf', format='pdf', bbox_inches='tight')
    plt.show()

from matplotlib.patches import Patch, Circle, Polygon, Ellipse
from skimage.measure import label, regionprops

def visualize_color_regions_with_colored_circles(initial_color_array, sorted_color_names, sorted_lab_values):
    """
    Visualize color-mapped image with colored circles drawn over the largest region of each color,
    and label each circle with the color name using adaptive text color and bold font.
    """
    # Step 1: Convert LAB to RGB
    lab_array = np.array(sorted_lab_values, dtype=np.uint8).reshape(-1, 1, 3)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    rgb_colors = {name: tuple(rgb_array[i]) for i, name in enumerate(sorted_color_names)}

    # Step 2: Create RGB image
    height, width = initial_color_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for name, rgb in rgb_colors.items():
        mask = initial_color_array == name
        rgb_image[mask] = rgb

    # Step 3: Plot image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb_image)
    ax.axis('off')
    ax.set_title("Color Regions with Colored Circles")

    # Step 4: Draw circle and label for each color
    for name in sorted_color_names:
        binary_mask = (initial_color_array == name).astype(np.uint8)
        labeled = label(binary_mask, connectivity=1)
        regions = regionprops(labeled)

        if not regions:
            continue

        largest_region = max(regions, key=lambda r: r.area)
        coords = largest_region.coords
        cy, cx = largest_region.centroid
        cy, cx = int(cy), int(cx)

        # Estimate radius
        distances = np.linalg.norm(coords - np.array([cy, cx]), axis=1)
        radius = int(np.max(distances)) + 5

        # Get color in [0,1] and draw circle
        rgb = np.array(rgb_colors[name]) / 255.0
        circle = Circle((cx, cy), radius, fill=False, edgecolor=rgb, linewidth=2.0, alpha=0.9)
        ax.add_patch(circle)

        # Calculate brightness for adaptive text color
        R, G, B = rgb * 255
        brightness = 0.299 * R + 0.587 * G + 0.114 * B
        text_color = 'black' if brightness > 150 else 'white'

        # Add bold label
        ax.text(cx, cy, name, color=text_color, fontsize=9, fontweight='bold',
                ha='center', va='center')

    # Step 5: Add legend
    legend_elements = [Patch(facecolor=np.array(rgb)/255.0, edgecolor='black', label=name)
                       for name, rgb in rgb_colors.items()]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(legend_elements), 5), frameon=False)

    plt.tight_layout()
    plt.show()

def visualize_color_regions_with_colored_fixedcircles(initial_color_array, sorted_color_names, sorted_lab_values):
    """
    Visualize color-mapped image with circles of size equal to the smallest largest-region radius among all colors.
    Each circle is labeled with the color name using adaptive bold text.
    """
    # Step 1: Convert LAB to RGB
    lab_array = np.array(sorted_lab_values, dtype=np.uint8).reshape(-1, 1, 3)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    rgb_colors = {name: tuple(rgb_array[i]) for i, name in enumerate(sorted_color_names)}

    # Step 2: Create RGB image
    height, width = initial_color_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for name, rgb in rgb_colors.items():
        mask = initial_color_array == name
        rgb_image[mask] = rgb

    # Step 3: Precompute all radii and centroids
    color_centers = {}
    color_radii = {}

    for name in sorted_color_names:
        binary_mask = (initial_color_array == name).astype(np.uint8)
        labeled = label(binary_mask, connectivity=1)
        regions = regionprops(labeled)

        if not regions:
            continue

        largest_region = max(regions, key=lambda r: r.area)
        coords = largest_region.coords
        cy, cx = largest_region.centroid
        cy, cx = int(cy), int(cx)

        distances = np.linalg.norm(coords - np.array([cy, cx]), axis=1)
        radius = int(np.max(distances)) + 5

        color_centers[name] = (cx, cy)
        color_radii[name] = radius

    # Step 4: Choose minimum radius from all colors
    if color_radii:
        min_radius = min(color_radii.values())
    else:
        min_radius = min(height, width) // 10  # fallback

    # Step 5: Plot image
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.imshow(rgb_image)

    ax.axis('off')
    ax.set_title("Color Regions with Equal-Sized Circles")

    # Step 6: Draw circle and label
    for name in color_centers:
        cx, cy = color_centers[name]
        rgb = np.array(rgb_colors[name]) / 255.0

        circle = Circle((cx, cy), min_radius*3, fill=False, edgecolor='#F6057E', linewidth=2.0, alpha=0.9)
        ax.add_patch(circle)





        # Adaptive text color
        R, G, B = rgb * 255
        brightness = 0.299 * R + 0.587 * G + 0.114 * B
        text_color = 'black' if brightness > 150 else 'white'

        fontsize=15
        estimated_text_width = len(name) * (fontsize * 0.85)
        if cx + min_radius + estimated_text_width > width:
          label_x = cx - min_radius - 6
          ha = 'right'
        else:
          label_x = cx + min_radius + 4
          ha = 'left'

        # color='#e8e227'
        ax.text(label_x, cy, name, color='#F6057E', fontsize=15, fontweight='bold',
                ha=ha, va='center')

    # Step 7: Add legend
    legend_elements = [Patch(facecolor=np.array(rgb)/255.0, edgecolor='black', label=name)
                       for name, rgb in rgb_colors.items()]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(legend_elements), 5), frameon=False)

    plt.tight_layout()

    plt.savefig('continuous_image_circle.pdf', format='pdf', bbox_inches='tight')
    plt.show()


from skimage import color
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from math import ceil

def opencv_lab_to_standard_lab(lab_list):
    """
    Converts OpenCV-style LAB (uint8) to standard CIE LAB
    """
    L = float(lab_list[0]) * 100 / 255
    a = float(lab_list[1]) - 128
    b = float(lab_list[2]) - 128
    return [round(L, 1), round(a, 1), round(b, 1)]

def standard_lab_to_rgb(lab_list):
    """
    Converts standard LAB to RGB [0,1] for matplotlib
    """
    lab_arr = np.array(lab_list).reshape(1, 1, 3)
    rgb = color.lab2rgb(lab_arr)
    return rgb[0, 0]

def format_lab_label(lab):
    """
    Nicely formats standard LAB values as string.
    """
    return f"L*A*B: [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]"

def auto_bar_width_and_positions(n_bars, total_width=0.9):
    """
    Computes optimal bar width and x positions to fill total width nicely.

    Parameters:
    - n_bars: number of bars
    - total_width: portion of x-axis to fill (0.9 means 90%)

    Returns:
    - width: bar width
    - positions: adjusted x locations
    """
    width = total_width / n_bars
    positions = np.arange(n_bars)
    return width, positions


def plot_color_lab_barchart(color_names, counts, opencv_lab_values):
    """
    Plots color bar chart with LAB colors,
    with LAB values in a clean legend below and log-scaled Y axis.
    """

    standard_labs = [opencv_lab_to_standard_lab(lab) for lab in opencv_lab_values]
    bar_colors = [standard_lab_to_rgb(lab) for lab in standard_labs]

    x = np.arange(len(color_names))
    fig, ax = plt.subplots(figsize=(14, 8))

    # bar_width, x = auto_bar_width_and_positions(len(color_names))
    bars = ax.bar(x, counts, color=bar_colors, tick_label=color_names)

    ax.set_yscale('log')

    ax.set_ylabel("Pixel Count (log scale)", fontsize=12)
    ax.set_title("Color Distribution with L*a*b* Values", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(color_names, rotation=45, ha='right')

    # Legend labels
    legend_labels = [
        f"{name} — LAB: [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]"
        for name, lab in zip(color_names, standard_labs)
    ]

    ax.legend(bars, legend_labels,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.25),
              ncol=4,
              fontsize=9,
              title="Colors and L*a*b* Values")

    ax.grid(True, axis='y', linestyle='-', alpha=0.4)

    plt.tight_layout()
    plt.savefig('discrete_color_original_distribution.pdf',format='pdf',bbox_inches='tight')
    plt.show()

def plot_nearest_color_distances_splitbars(pairs, color_to_rgb):
    """
    Plots a sorted bar chart of nearest color pairs with split-color bars.

    Parameters:
    - pairs: list of tuples (color1, color2, distance)
    - color_to_rgb: dict mapping color name to RGB (values in [0, 1])
    """

    sorted_pairs = sorted(pairs, key=lambda x: x[2])
    labels = [f"{p[0]} - {p[1]}" for p in sorted_pairs]
    distances = [p[2] for p in sorted_pairs]

    x = range(len(distances))
    bar_width = 0.8

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (color1, color2, distance) in enumerate(sorted_pairs):
        # Get RGBs for both sides
        rgb1 = color_to_rgb.get(color1, (0.5, 0.5, 0.5))
        rgb2 = color_to_rgb.get(color2, (0.5, 0.5, 0.5))

        # Draw left half (color1)
        ax.bar(x[i] - bar_width/4, distance, width=bar_width/2, color=rgb1, align='center')
        # Draw right half (color2)
        ax.bar(x[i] + bar_width/4, distance, width=bar_width/2, color=rgb2, align='center')

        # Add distance label
        ax.text(x[i], distance + 0.4, f"{distance:.1f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Euclidean Distance in L*a*b* color space", fontsize=12)
    ax.set_title("Nearest Color Pairs by Distance")
    ax.grid(True, axis='y', linestyle='-', alpha=0.4)

    plt.tight_layout()
    plt.savefig('discrete_color_nearest_distance.pdf',format='pdf',bbox_inches='tight')
    plt.show()

def plot_lab_3d_scatter(color_names, counts, opencv_lab_values, label_points=False):
    """
    Plots a 3D scatter plot in LAB space.

    Parameters:
    - color_names: list of color names
    - counts: list of pixel counts
    - opencv_lab_values: list of LAB values in OpenCV format
    - label_points: whether to show text labels (color names)
    """

    standard_labs = [opencv_lab_to_standard_lab(lab) for lab in opencv_lab_values]
    rgb_colors = [standard_lab_to_rgb(lab) for lab in standard_labs]

    # Unpack into separate coordinates
    L_vals = [lab[0] for lab in standard_labs]
    a_vals = [lab[1] for lab in standard_labs]
    b_vals = [lab[2] for lab in standard_labs]

    # Normalize sizes for scatter points
    counts_array = np.array(counts)
    sizes = 10 + 200 * (np.log1p(counts_array) / np.log1p(counts_array.max()))

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(a_vals, b_vals, L_vals, c=rgb_colors, s=sizes, edgecolors='k', alpha=0.8)

    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_zlabel("L*")
    ax.set_zlim(0, 100)
    ax.view_init(elev=40, azim=150)
    ax.set_title("Unique Color names and values in L*a*b* Color Space")

    # Optional label
    if label_points:
        for name, x, y, z in zip(color_names, a_vals, b_vals, L_vals):
            ax.text(x, y, z, name, fontsize=8)

    plt.tight_layout()
    plt.savefig('discrete_color_scatter.pdf',format='pdf',bbox_inches='tight')
    plt.show()

def find_nearest_color_pairs(color_names, opencv_lab_values):
    """
    For each color, finds the nearest other color (based on LAB distance).

    Returns:
    - nearest_pairs: list of (color_name, closest_color_name, distance)
    - min_pair: (color1, lab1, color2, lab2, distance)
    """

    standard_labs = np.array([
        opencv_lab_to_standard_lab(lab) for lab in opencv_lab_values
    ])

    dist_matrix = cdist(standard_labs, standard_labs, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    nearest_pairs = []
    min_distance = np.inf
    min_pair = None

    for i, name in enumerate(color_names):
        min_idx = np.argmin(dist_matrix[i])
        closest_name = color_names[min_idx]
        distance = float(dist_matrix[i, min_idx])

        nearest_pairs.append((name, closest_name, round(distance, 2)))

        if distance < min_distance:
            min_distance = distance
            min_pair = (
                name,
                [round(float(x), 2) for x in standard_labs[i]],
                closest_name,
                [round(float(x), 2) for x in standard_labs[min_idx]],
                round(distance, 2)
            )

    return nearest_pairs, min_pair

def plot_nearest_color_distances(pairs):
    """
    Plots a sorted bar chart of nearest color pair distances.

    Parameters:
    - pairs: list of tuples (color1, color2, distance)
    """

    # Sort pairs by distance (ascending)
    sorted_pairs = sorted(pairs, key=lambda x: x[2])

    # Create labels like "color1 → color2"
    labels = [f"{p[0]} → {p[1]}" for p in sorted_pairs]
    distances = [p[2] for p in sorted_pairs]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(distances)), distances, tick_label=labels, color='steelblue')

    # Add distance labels above bars
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 f"{distance:.2f}", ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Euclidean Distance in L*a*b* color space")
    plt.title("Nearest Color Pairs by Distance")
    plt.tight_layout()

    plt.show()

def plot_color_lab_barchart_grouped(color_names, counts, opencv_lab_values, cluster_dict):
    """
    Plots a grouped bar chart where bars are grouped by cluster.

    Parameters:
    - color_names: list of all color names (same order as counts and lab_values)
    - counts: list of pixel counts
    - opencv_lab_values: list of LAB(OpenCV) values
    - cluster_dict: dict {cluster_name: [color_name1, color_name2, ...]}
    """

    # Step 1: create lookup dicts
    name_to_count = dict(zip(color_names, counts))
    name_to_lab = dict(zip(color_names, opencv_lab_values))

    # Step 2: create grouped & sorted list
    grouped_color_names = []
    for cluster, color_list in cluster_dict.items():
        grouped_color_names.extend(color_list)

    # Step 3: extract reordered data
    grouped_counts = [name_to_count[name] for name in grouped_color_names]
    grouped_labs = [name_to_lab[name] for name in grouped_color_names]
    standard_labs = [opencv_lab_to_standard_lab(lab) for lab in grouped_labs]
    bar_colors = [standard_lab_to_rgb(lab) for lab in standard_labs]

    # Step 4: plot
    x = np.arange(len(grouped_color_names))
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(x, grouped_counts, tick_label=grouped_color_names, color=bar_colors)
    ax.set_yscale('log')

    ax.set_ylabel("Pixel Count (log scale)", fontsize=12)
    ax.set_title("Grouped Color Distribution by Cluster (L*a*b*)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_color_names, rotation=45, ha='right')

    # Legend labels
    legend_labels = [
        f"{name} — LAB: [{lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}]"
        for name, lab in zip(grouped_color_names, standard_labs)
    ]

    ax.legend(bars, legend_labels,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.25),
              ncol=4,
              fontsize=9,
              title="Colors and L*a*b* Values")

    ax.grid(True, axis='y', linestyle='-', alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_color_lab_barchart_grouped_with_clusters(
    color_names, counts, opencv_lab_values, cluster_dict, cluster_name_to_lab):
    """
    Plots a grouped color bar chart by cluster with:
    - cluster labels above each group
    - color patches for each cluster
    - vertical dashed separators between clusters
    """

    # Create lookup tables
    name_to_count = dict(zip(color_names, counts))
    name_to_lab = dict(zip(color_names, opencv_lab_values))

    # Create reordered lists
    grouped_color_names = []
    group_boundaries = []
    cluster_labels = []
    cluster_patches = []

    for cluster_name, group_colors in cluster_dict.items():
        group_boundaries.append(len(grouped_color_names))
        grouped_color_names.extend(group_colors)
        cluster_labels.append(cluster_name)
        cluster_patches.append(cluster_name_to_lab[cluster_name])  # OpenCV format

    grouped_counts = [name_to_count[name] for name in grouped_color_names]
    grouped_labs = [name_to_lab[name] for name in grouped_color_names]
    standard_labs = [opencv_lab_to_standard_lab(lab) for lab in grouped_labs]
    bar_colors = [standard_lab_to_rgb(lab) for lab in standard_labs]

    # Setup plot
    x = np.arange(len(grouped_color_names))
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(x, grouped_counts, tick_label=grouped_color_names, color=bar_colors)
    ax.set_yscale('log')
    ax.set_ylabel("Pixel Count (log scale)", fontsize=12)
    # ax.set_title("Grouped Color Distribution by Cluster (L*a*b*)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_color_names, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='-', alpha=0.4)

    # Plot vertical separators + cluster labels and color patches
    for i, start_idx in enumerate(group_boundaries):
        # Determine end of this cluster group
        end_idx = group_boundaries[i + 1] if i + 1 < len(group_boundaries) else len(grouped_color_names)
        mid_x = (start_idx + end_idx - 1) / 2

        # Draw cluster label
        ax.text(mid_x, ax.get_ylim()[1] * 1.05, cluster_labels[i],
                ha='center', va='bottom', fontsize=10, weight='bold')

        # Draw cluster color patch (converted to standard LAB RGB)
        standard_lab = opencv_lab_to_standard_lab(cluster_patches[i])
        rgb = standard_lab_to_rgb(standard_lab)
        ax.add_patch(plt.Rectangle((mid_x - 0.5, ax.get_ylim()[1] * 1.02),
                                   1.0, ax.get_ylim()[1] * 0.12,
                                   color=rgb, transform=ax.transData, clip_on=False))

        # Draw vertical separator between groups (skip last one)
        if i > 0:
            ax.axvline(x=start_idx - 0.5, linestyle='-', color='gray', alpha=0.5)

    plt.tight_layout()

    plt.savefig('discrete_color_cluster.pdf',format='pdf',bbox_inches='tight')
    plt.show()

def plot_color_palette(sorted_color_names, sorted_lab_values, savepath, items_per_row=5,
                                             box_px=80, label_height_px=20, dpi=100, font_size=9.2):
    """
    Plots a fixed-size color palette with dynamic horizontal spacing to avoid label overlap.
    """

    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    # Convert LAB(OpenCV) to RGB
    rgb_colors = [
        standard_lab_to_rgb(opencv_lab_to_standard_lab(lab))
        for lab in sorted_lab_values
    ]

    total = len(sorted_color_names)
    n_rows = ceil(total / items_per_row)

    # Estimate label width in pixels based on max name length
    longest_label_len = max(len(name) for name in sorted_color_names)
    label_px_per_char = font_size * 0.6  # estimated width in pixels per character
    spacing_x = int(label_px_per_char * longest_label_len * 0.1)
    spacing_x = max(spacing_x, 10)  # ensure some space

    spacing_y = 10 + label_height_px

    fig_width_px = items_per_row * (box_px + spacing_x)
    fig_height_px = n_rows * (box_px + spacing_y)

    fig_width_in = fig_width_px / dpi
    fig_height_in = fig_height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_xlim(0, fig_width_px)
    ax.set_ylim(0, fig_height_px)
    ax.axis('off')

    for idx, (name, rgb) in enumerate(zip(sorted_color_names, rgb_colors)):
        row = idx // items_per_row
        col = idx % items_per_row

        x = col * (box_px + spacing_x)
        y = fig_height_px - (row + 1) * (box_px + spacing_y)

        rect = Rectangle((x, y), box_px, box_px, facecolor=rgb)
        ax.add_patch(rect)

        ax.text(x + box_px / 2, y - 5, name,
                ha='center', va='top', fontsize=font_size)

    plt.savefig(savepath, bbox_inches='tight')

    plt.show()


def visualize_color_regions_with_colored_fixedcircles(initial_color_array, sorted_color_names, sorted_lab_values, savepath, bkg_color=None):
    """
    Visualize color-labeled image with fixed-size circles at representative positions for each color.
    Each circle is labeled with the color name and a color legend is shown below.
    If bkg_color is provided, that color will be excluded from circles, labels, and legend.
    """

    # Step 1: Convert LAB (OpenCV format) to RGB
    lab_array = np.array(sorted_lab_values, dtype=np.uint8).reshape(-1, 1, 3)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB).reshape(-1, 3)
    rgb_colors = {name: tuple(rgb_array[i]) for i, name in enumerate(sorted_color_names)}

    # Step 2: Create RGB image for display
    height, width = initial_color_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for name, rgb in rgb_colors.items():
        mask = initial_color_array == name
        rgb_image[mask] = rgb

    # Step 3: Choose fixed-radius circle and pick a representative center for each color
    color_centers = {}
    color_radii = {}
    for name in sorted_color_names:
        if bkg_color is not None and name == bkg_color:
            continue  # skip background color

        coords = np.argwhere(initial_color_array == name)
        if coords.size == 0:
            continue
        cy, cx = coords[len(coords) // 2]  # use the middle point
        color_centers[name] = (cx, cy)
        color_radii[name] = 30  # fixed radius

    # Step 4: Plot image
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.imshow(rgb_image)
    ax.axis('off')
    ax.set_title("Color Regions with Fixed-Sized Circles")

    # Step 5: Draw circles and labels
    for name in color_centers:
        cx, cy = color_centers[name]
        rgb = np.array(rgb_colors[name]) / 255.0

        circle = Circle((cx, cy), 15, fill=False, edgecolor='white', linewidth=2.0, alpha=0.9)
        ax.add_patch(circle)

        # Adaptive text color for readability
        R, G, B = rgb * 255
        brightness = 0.299 * R + 0.587 * G + 0.114 * B
        text_color = 'black' if brightness > 150 else 'white'

        fontsize = 15
        estimated_text_width = len(name) * (fontsize * 0.85)
        if cx + 30 + estimated_text_width > width:
            label_x = cx - 30 - 6
            ha = 'right'
        else:
            label_x = cx + 30 + 4
            ha = 'left'

        ax.text(label_x, cy, name, color='#FFF608', fontsize=15, fontweight='bold',
                ha=ha, va='center')

    # Step 6: Add legend (exclude bkg_color if specified)
    legend_elements = [
        Patch(facecolor=np.array(rgb_colors[name]) / 255.0, edgecolor='black', label=name)
        for name in sorted_color_names if bkg_color is None or name != bkg_color
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(legend_elements), 5), frameon=False)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def extract_representative_labs(unique_colors, color_name_matrix, color_map_final):
    """
    Extracts a representative LAB value (first occurrence) for each unique color name.

    Returns:
    - sorted_lab_mid: list of LAB values (same order as unique_colors)
    """
    representative_labs = []

    for color in unique_colors:
        positions = np.argwhere(color_name_matrix == color)
        if positions.size > 0:
            y, x = positions[0]
            lab_value = color_map_final[y, x]
            representative_labs.append(lab_value)
        else:
            representative_labs.append([0, 0, 0])  # fallback in case it's missing (shouldn't happen)

    return representative_labs




def calculate_local_image_omec(image_path, color_df, similar_colors_dict, threshold=14):
    """
    计算本地单张图片的 O.MeC 值
    参数:
        image_path: 本地图片文件的路径 (例如 'C:/Images/test.jpg' 或 './test.jpg')
    """
    print(f"正在读取本地图片: {image_path}")
    
    # [修改点] 使用 cv2.imread 直接读取本地文件
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"错误: 无法读取图片，请检查路径是否正确: {image_path}")
        return None

    # 以下逻辑与原逻辑完全一致 ------------------------
    
    # 1. 初始分析
    # 注意：假设无背景去除 (即 assume background=['no'])
    sorted_color_names, sorted_percentages, sorted_lab, initial_color_map = \
        analyze_image_colors_and_names_lab_array(image, color_df)

    # 2. 准备黑白分离
    bw, _, _, bw_colors, bw_lab_values = check_lab_values_withthreshold(sorted_color_names, sorted_lab, threshold)
    bw_perc = get_subcolor_percentages(sorted_color_names, sorted_percentages, bw_colors)
    bw_perc_idx = get_subcolor_indices(sorted_color_names, sorted_percentages, bw_colors)

    # 3. 语义合并
    colors_mid, percentages_mid, lab_mid, _, color_map_mid = \
        merge_similar_colors_nobackground(
            sorted_color_names, sorted_percentages, sorted_lab, initial_color_map,
            similar_colors_dict, bw_colors, bw_perc, bw_lab_values, bw_perc_idx
        )

    # 4. 距离合并 (聚类)
    if len(bw_colors) > 0:
        lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=threshold)
        _, _, unique_colors, _ = merge_and_convert_colors(lookup_table, colors_mid, lab_mid, color_map_mid)
    else:
        lookup_table = generate_lookup_table3(colors_mid, lab_mid, threshold=threshold)
        _, _, unique_colors, _ = merge_and_convert_colors(lookup_table, colors_mid, lab_mid, color_map_mid)

    # 5. 输出结果
    omec_value = len(unique_colors)
    # print(f"O.MeC 计算完成。")
    # print(f"最终颜色簇数量 (O.MeC): {omec_value}")
    # print(f"包含颜色: {unique_colors}")
    
    return omec_value, unique_colors

if __name__ == "__main__":
    color_df = pd.read_excel('./HeerStone_colorNaming.xlsx')
    color_data = pd.read_excel('./HeerStone_colorSimilarity.xlsx')
    # Converting string representation of list to actual list
    color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
    # Create a dictionary to map each color to its similar colors
    similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))


    num, ans = calculate_local_image_omec('./output/2/chart.png', color_df, similar_colors_dict, threshold=14)
    print('最终 O.MeC 值:', num)
    print('颜色有：', ans)


# color_df = pd.DataFrame(worksheet2.get_all_records())
# color_data = pd.DataFrame(worksheet3.get_all_records())

# Images=df_original['ImageLink'].to_list()
# Names=df_original['Image name'].to_list()
# background=df_original['Background Color'].to_list()
# k_cluster=df_original['O.MeC'].to_list()
# isbw=df_original['Isbw'].to_list()
# continuous=df_original['IsContinuous'].to_list()
# mec=df_original['O.MeC'].to_list()



# color_data = pd.DataFrame(worksheet3.get_all_records())
# # Converting string representation of list to actual list
# color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
# # Create a dictionary to map each color to its similar colors
# similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))


# for i in range(len(Images)):

#   if continuous[i]=='Y':
#       if str(isbw[i])=='0':
#           idx=i
#           print(idx,Names[idx])
#           save_path='/content/drive/MyDrive/1800color/output_image/original_color_palette/'+Names[idx]
#           os.makedirs(os.path.dirname(save_path), exist_ok=True)
#           if ',' in background[idx]:
#               bkg_color=background[idx].split(',')
#           else:
#               bkg_color=background[idx].split(' ')

#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#               plot_color_palette(sorted_color_names, sorted_lab,save_path)

#           else:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#               plot_color_palette(sorted_color_names, sorted_lab,save_path)

#       elif str(isbw[i])=='1':
#             idx=i
#             print(idx,Names[idx])
#             save_path='/content/drive/MyDrive/1800color/output_image/original_color_palette/'+Names[idx]
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             if ',' in background[idx]:
#               bkg_color=background[idx].split(',')
#             else:
#               bkg_color=background[idx].split(' ')

#             if bkg_color!=['no']:
#                 color_df = pd.DataFrame(worksheet2.get_all_records())
#                 img=Images[idx]
#                 image = url_to_cv2_image(img)
#                 sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#                 color_df = pd.DataFrame(worksheet2.get_all_records())
#                 plot_color_palette(sorted_color_names, sorted_lab,save_path)
#             else:
#                 img=Images[idx]
#                 image = url_to_cv2_image(img)
#                 color_df = pd.DataFrame(worksheet2.get_all_records())
#                 sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#                 plot_color_palette(sorted_color_names, sorted_lab,save_path)

#   else:
#     idx=i
#     if ',' in background[idx]:
#       bkg_color=background[idx].split(',')
#     else:
#       bkg_color=background[idx].split(' ')

#     print(idx,Names[idx])
#     save_path='/content/drive/MyDrive/1800color/output_image/original_color_palette/'+Names[idx]
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     img=Images[idx]
#     image = url_to_cv2_image(img)
#     color_df = pd.DataFrame(worksheet2.get_all_records())
#     sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#     plot_color_palette(sorted_color_names, sorted_lab,save_path)



# for idx in range(len(Images)):

#   img=Images[idx]
#   image = url_to_image(img)

#   if ',' in background[idx]:
#       bkg_color=background[idx].split(',')
#   else:
#       bkg_color=background[idx].split(' ')
#   if bkg_color==['no']:
#       bkg_color=[]



#   if continuous[idx]=='N':
#     print(idx,Names[idx])
#     color_df = pd.DataFrame(worksheet2.get_all_records())
#     sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#     color_df = pd.DataFrame(worksheet2.get_all_records())
#     sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)
#     kmeans_centroids, kmeans_labels, kmeans_percentages=weighted_k_means(sorted_lab, sorted_counts, n_clusters=int(k_cluster[idx]))
#     color_df = pd.DataFrame(worksheet2.get_all_records())
#     kmeans_name=kmean_color_name(kmeans_centroids,color_df)
#     print('Centroid name:',kmeans_name,'perc:',kmeans_percentages)
#     kmean_image,most_lab=apply_kmeans_to_foreground2(sorted_color_names,binary_mask, initial_color_array_wb, image, kmeans_centroids, kmeans_labels, kmeans_percentages)
#     color_df = pd.DataFrame(worksheet2.get_all_records())
#     kmeans_name2=kmean_color_name(most_lab,color_df)
#     print('Most lab vale name within cluster:',kmeans_name2)
#     centroids_path='1800color/output_image/centroids/'+Names[idx]
#     kmeans_path='output_image/Kmean_replot/'+Names[idx]
#     LAB_3D_kmean(image,sorted_lab,sorted_counts,kmeans_centroids,kmeans_labels,kmeans_percentages,kmean_image,centroids_path,kmeans_path,kmeans_name2)


# idx=0

# img=Images[idx]
# image = url_to_image(img)

# plot_and_save_image_from_url(img)

# if ',' in background[idx]:
#     bkg_color=background[idx].split(',')
# else:
#     bkg_color=background[idx].split(' ')
# if bkg_color==['no']:
#     bkg_color=[]


# categorical=not math.isnan(k_cluster[idx])

# if categorical:
#   print(idx,Names[idx])
#   color_df = pd.DataFrame(worksheet2.get_all_records())

#   sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#   plot_color_palette(sorted_color_names, sorted_lab,'discrete_color_ori_palette.pdf')
#   print(len(sorted_color_names),sorted_color_names)
#   plot_color_lab_barchart(sorted_color_names, sorted_counts, sorted_lab)
#   plot_lab_3d_scatter(sorted_color_names, sorted_counts, sorted_lab, label_points=True)
#   pairs, closest_pair = find_nearest_color_pairs(sorted_color_names, sorted_lab)
#   # for p in pairs:
#   #   print(p)
#   print("Closest overall pair:", closest_pair)

#   color_to_rgb = {name: standard_lab_to_rgb(opencv_lab_to_standard_lab(lab))
#                 for name, lab in zip(sorted_color_names, sorted_lab)}

#   plot_nearest_color_distances_splitbars(pairs, color_to_rgb)



#   color_df = pd.DataFrame(worksheet2.get_all_records())
#   sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)


#   n_clusters=int(k_cluster[idx])
#   kmeans_centroids, kmeans_labels, kmeans_percentages=weighted_k_means(sorted_lab, sorted_counts, n_clusters=n_clusters)

#   color_df = pd.DataFrame(worksheet2.get_all_records())
#   kmeans_name=kmean_color_name(kmeans_centroids,color_df)
#   print('Centroid name:',kmeans_name,'perc:',kmeans_percentages)

#   kmean_image,most_lab=apply_kmeans_to_foreground2(sorted_color_names,binary_mask, initial_color_array_wb, image, kmeans_centroids, kmeans_labels, kmeans_percentages)
#   color_df = pd.DataFrame(worksheet2.get_all_records())
#   kmeans_name2=kmean_color_name(most_lab,color_df)
#   print('Most lab value name within cluster:',kmeans_name2)

#   color_cluster_dict = group_colors_by_cluster_name(sorted_color_names, kmeans_labels, kmeans_name2)
#   # for cluster, colors in color_cluster_dict.items():
#   #   print(f"{cluster}: {colors}")


#   centroids_path='1800color/output_image/centroids/'+Names[idx]
#   kmeans_path='1800color/output_image/Kmean_replot/'+Names[idx]


#   # LAB_3D_kmean(image,sorted_lab,sorted_counts,kmeans_centroids,kmeans_labels,kmeans_percentages,kmean_image,centroids_path,kmeans_path,kmeans_name2)
#   LAB_3D_kmean_standard(image,sorted_lab,sorted_counts,kmeans_centroids,kmeans_labels,kmeans_percentages,kmean_image,centroids_path,kmeans_path,kmeans_name2)

#   # plot_color_lab_barchart_grouped(sorted_color_names,sorted_counts,sorted_lab,color_cluster_dict)

#   cluster_name_to_lab = {
#       name: most_lab[i]
#       for i, name in enumerate(kmeans_name2)
#   }
#   plot_color_lab_barchart_grouped_with_clusters(
#       sorted_color_names,
#       sorted_counts,
#       sorted_lab,
#       color_cluster_dict,
#       cluster_name_to_lab
#   )

#   plot_color_palette(kmeans_name2, most_lab,'discrete_color_cluster_palette.pdf')


# color_df = pd.DataFrame(worksheet2.get_all_records())
# color_data = pd.DataFrame(worksheet3.get_all_records())

# Images=df_original['ImageLink'].to_list()
# Names=df_original['Image name'].to_list()
# background=df_original['Background Color'].to_list()
# k_cluster=df_original['O.MeC'].to_list()
# isbw=df_original['Isbw'].to_list()
# continuous=df_original['IsContinuous'].to_list()
# mec=df_original['O.MeC'].to_list()


# # Converting string representation of list to actual list
# color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
# # Create a dictionary to map each color to its similar colors
# similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))

# color_nogroup=[]
# color_group=[]
# group_table=[]


# for i in range(len(Images)):
#   if continuous[i]=='Y':

#     if not math.isnan(mec[i]):
#       if str(isbw[i])=='0':
#           idx=i
#           print(idx,Names[idx])
#           print('Background color:',background[idx])


#           if ',' in background[idx]:
#               bkg_color=background[idx].split(',')
#           else:
#               bkg_color=background[idx].split(' ')

#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)

#               sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#               # print(sorted_color_names)
#               # plot_color_bar_chart(sorted_color_names, sorted_color_percentages, sorted_lab)

#           else:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)

#               sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#               # print(sorted_color_names)
#               # plot_color_bar_chart(sorted_color_names, sorted_color_percentages, sorted_lab)
#               # visualize_color_name_map(initial_color_array, sorted_color_names, sorted_lab)

#               bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names, sorted_lab,threshold=14)
#               bw_perc = get_subcolor_percentages(sorted_color_names, sorted_color_percentages, bw_colors)
#               bw_perc_idx = get_subcolor_indices(sorted_color_names, sorted_color_percentages, bw_colors)


#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)


#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)

#               bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#               bw_perc = get_subcolor_percentages(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)
#               bw_perc_idx = get_subcolor_indices(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)

#               sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_final_wbw(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,initial_color_array_wb,similar_colors_dict,image,binary_mask,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#               # print(sorted_color_names_mid)
#               # print(bw_colors)
#               # plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid)
#               # visualize_color_name_map(color_map_mid, sorted_color_names_mid, sorted_lab_mid)
#               # visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid)

#               lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#               # print(lookup_table)
#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors_with_mask(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid,binary_mask,image)
#               print('After grouping:',unique_colors,len(unique_colors))

#               lab_tuples = np.array([[tuple(lab) for lab in row] for row in color_map_final])
#               unique_lab_values = np.unique(lab_tuples.reshape(-1, lab_tuples.shape[-1]), axis=0)

#               color_nogroup.append(len(sorted_color_names_mid))
#               color_group.append(len(unique_colors))
#               group_table.append(lookup_table)
#               # image = url_to_image(img)
#               # merge_path='/content/drive/MyDrive/1800color/output_image/Merge similar/'+Names[idx]
#               # difference_path='/content/drive/MyDrive/1800color/output_image/Lab difference/'+Names[idx]

#               # display_lab_image(image,color_map_final,merge_path,difference_path)

#           else:
#                 color_df = pd.DataFrame(worksheet2.get_all_records())

#                 img=Images[idx]
#                 image = url_to_cv2_image(img)
#                 sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_nobackground(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#                 # print(sorted_color_names_mid)
#                 # print(bw_colors)
#                 # plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid)
#                 # visualize_color_name_map(color_map_mid, sorted_color_names_mid, sorted_lab_mid)
#                 # visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid)
#                 if len(bw_colors)>0:
#                     lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#                     # print(lookup_table)
#                     color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid)
#                     print('After grouping:',unique_colors,len(unique_colors))

#                     group_table.append(lookup_table)
#                     color_group.append(len(unique_colors))

#                 else:
#                     color_map,sorted_color_names_final,sorted_color_percentages_final,sorted_lab_final,color_map_final=merge_similar_colors_nobw(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict)
#                     print('After grouping:',sorted_color_names_final,len(sorted_color_names_final))

#                     color_group.append(len(sorted_color_names_final))
#                     group_table.append('')

#                 color_nogroup.append(len(sorted_color_names_mid))



#                 lab_tuples = np.array([[tuple(lab) for lab in row] for row in color_map_final])
#                 unique_lab_values = np.unique(lab_tuples.reshape(-1, lab_tuples.shape[-1]), axis=0)

#       elif str(isbw[i])=='1':
#           idx=i

#           print(idx,Names[idx])
#           print('Background color:',background[idx])

#           if ',' in background[idx]:
#             bkg_color=background[idx].split(',')
#           else:
#             bkg_color=background[idx].split(' ')

#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)

#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)
#               # print('Before grouping:',sorted_color_names,sorted_lab,len(sorted_color_names))
#               lookup_table = generate_lookup_table3(sorted_color_names, sorted_lab, threshold=14)
#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors_with_mask(lookup_table, sorted_color_names, sorted_lab, initial_color_array_wb,binary_mask,image)
#               print('After grouping:',unique_colors,len(unique_colors),unique_percentages)

#               lab_tuples = np.array([[tuple(lab) for lab in row] for row in color_map_final])
#               unique_lab_values = np.unique(lab_tuples.reshape(-1, lab_tuples.shape[-1]), axis=0)
#               image = url_to_cv2_image(img)

#               color_nogroup.append(len(sorted_color_names))
#               color_group.append(len(unique_colors))
#               group_table.append(lookup_table)


#           else:
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#               # print('Before grouping:',sorted_color_names,sorted_lab)
#               lookup_table = generate_lookup_table3(sorted_color_names, sorted_lab, threshold=14)
#               print(lookup_table)
#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors(lookup_table, sorted_color_names, sorted_lab, initial_color_array)

#               print('After grouping:',unique_colors,len(unique_colors),unique_percentages)
#               lab_tuples = np.array([[tuple(lab) for lab in row] for row in color_map_final])
#               unique_lab_values = np.unique(lab_tuples.reshape(-1, lab_tuples.shape[-1]), axis=0)

#               color_nogroup.append(len(sorted_color_names))
#               color_group.append(len(unique_colors))
#               group_table.append(lookup_table)
#     else:
#       color_nogroup.append('')
#       color_group.append('')
#       group_table.append('')
#   else:
#     color_nogroup.append('')
#     color_group.append('')
#     group_table.append('')

# df = pd.DataFrame({'Image name':Names,'O.MeC for continuous':color_group,})
# df.index.name = 'Index'
# df.to_csv('MeC for continuous color image.csv',encoding='utf-8-sig')


# color_df = pd.DataFrame(worksheet2.get_all_records())
# color_data = pd.DataFrame(worksheet3.get_all_records())

# Images=df_original['ImageLink'].to_list()
# Names=df_original['Image name'].to_list()
# background=df_original['Background Color'].to_list()
# k_cluster=df_original['O.MeC'].to_list()
# isbw=df_original['Isbw'].to_list()
# continuous=df_original['IsContinuous'].to_list()

# # Converting string representation of list to actual list
# color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
# # Create a dictionary to map each color to its similar colors
# similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))

# save_path1='/content/drive/MyDrive/1800color/output_image/merge_color_circles/'
# save_path2='/content/drive/MyDrive/1800color/output_image/merge_color_list/'
# save_path3='/content/drive/MyDrive/1800color/output_image/final_color_circles/'
# save_path4='/content/drive/MyDrive/1800color/output_image/final_color_list/'
# os.makedirs(os.path.dirname(save_path1), exist_ok=True)
# os.makedirs(os.path.dirname(save_path2), exist_ok=True)
# os.makedirs(os.path.dirname(save_path3), exist_ok=True)
# os.makedirs(os.path.dirname(save_path4), exist_ok=True)


# for i in range(len(Images)):
#   if continuous[i]=='Y':
#       # plot_and_save_image_from_url(img)
#       if str(isbw[i])=='0':
#           idx=i
#           print(idx,Names[idx])
#           print('Background color:',background[idx])
#           if ',' in background[idx]:
#               bkg_color=background[idx].split(',')
#           else:
#               bkg_color=background[idx].split(' ')

#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#               # plot_color_palette(sorted_color_names, sorted_lab,'continouous_color_ori_palette.pdf')

#           else:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#               # print(sorted_color_names)
#               # plot_color_bar_chart(sorted_color_names, sorted_color_percentages, sorted_lab,'continuous_color_ori_distribution.pdf')
#               # plot_color_palette(sorted_color_names, sorted_lab,'continouous_color_ori_palette.pdf')
#               # visualize_color_name_map(initial_color_array, sorted_color_names, sorted_lab)

#               bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#               bw_perc = get_subcolor_percentages(sorted_color_names, sorted_color_percentages, bw_colors)
#               bw_perc_idx = get_subcolor_indices(sorted_color_names, sorted_color_percentages, bw_colors)


#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_image(img)

#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)
#               # plot_color_bar_chart(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,'continuous_color_ori_distribution.pdf')
#               # visualize_color_name_map(initial_color_array_wb, sorted_color_names_wb, sorted_lab_wb)

#               bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#               # print(sorted_color_names_wb)
#               # print(bw_colors)
#               bw_perc = get_subcolor_percentages(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)
#               bw_perc_idx = get_subcolor_indices(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)

#               sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_final_wbw(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,initial_color_array_wb,similar_colors_dict,image,binary_mask,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#               # plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid,'continuous_color_merge_distribution.pdf')
#               lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#               # print(lookup_table)
#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors_with_mask(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid,binary_mask,image)
#               print('After grouping:',unique_colors,len(unique_colors))

#               save_path=save_path1+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid,save_path)

#               save_path=save_path2+Names[idx]
#               plot_color_palette(sorted_color_names_mid, sorted_lab_mid,save_path)

#               background_colors = color_name_matrix[binary_mask == 1]
#               unique_bg_colors, bg_counts = np.unique(background_colors, return_counts=True)
#               background_lab_values = color_map_final[binary_mask == 1]
#               background_lab_values = np.array(background_lab_values.tolist())[0]

#               lab_final = extract_representative_labs(unique_colors, color_name_matrix, color_map_final)

#               unique_colors2=np.concatenate((unique_colors, unique_bg_colors))
#               lab_final2 = lab_final + [background_lab_values.tolist()]

#               save_path=save_path3+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors2, lab_final2,save_path,unique_bg_colors.item())

#               save_path=save_path4+Names[idx]
#               plot_color_palette(unique_colors, lab_final,save_path)



#           else:
#               color_df = pd.DataFrame(worksheet2.get_all_records())

#               img=Images[idx]
#               image = url_to_image(img)
#               sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_nobackground(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#               # print(sorted_color_names_mid)
#               # print(bw_colors)
#               # plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid,'continuous_color_merge_distribution.pdf')
#               # visualize_color_name_map(color_map_mid, sorted_color_names_mid, sorted_lab_mid)

#               # visualize_color_regions_with_colored_circles(color_map_mid, sorted_color_names_mid, sorted_lab_mid)
#               save_path=save_path1+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid,save_path)
#               save_path=save_path2+Names[idx]
#               plot_color_palette(sorted_color_names_mid, sorted_lab_mid,save_path)

#               if len(bw_colors)>0:
#                   lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#                   color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid)
#                   print('After grouping:',unique_colors,len(unique_colors))
#                   lab_final = extract_representative_labs(unique_colors, color_name_matrix, color_map_final)

#                   save_path=save_path3+Names[idx]
#                   visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors, lab_final,save_path)
#                   save_path=save_path4+Names[idx]
#                   plot_color_palette(unique_colors, lab_final,save_path)

#               else:
#                   color_name_matrix,sorted_color_names_final,sorted_color_percentages_final,sorted_lab_final,color_map_final=merge_similar_colors_nobw(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict)
#                   print('After grouping:',sorted_color_names_final,len(sorted_color_names_final))
#                   save_path=save_path3+Names[idx]
#                   visualize_color_regions_with_colored_fixedcircles(color_name_matrix, sorted_color_names_final, sorted_lab_final,save_path)
#                   save_path=save_path4+Names[idx]
#                   plot_color_palette(sorted_color_names_final, sorted_lab_final,save_path)

#       elif str(isbw[i])=='1':
#           idx=i
#           print(idx,Names[idx])
#           print('Background color:',background[idx])

#           if ',' in background[idx]:
#             bkg_color=background[idx].split(',')
#           else:
#             bkg_color=background[idx].split(' ')

#           if bkg_color!=['no']:
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)

#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)

#               bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#               bw_perc = get_subcolor_percentages(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)
#               bw_perc_idx = get_subcolor_indices(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)

#               sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_final_wbw(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,initial_color_array_wb,similar_colors_dict,image,binary_mask,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#               # print('Before grouping:',sorted_color_names,sorted_lab,len(sorted_color_names))

#               save_path=save_path1+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid,save_path)
#               save_path=save_path2+Names[idx]
#               plot_color_palette(sorted_color_names_mid, sorted_lab_mid,save_path)


#               lookup_table = generate_lookup_table3(sorted_color_names, sorted_lab, threshold=14)
#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors_with_mask(lookup_table, sorted_color_names, sorted_lab, initial_color_array_wb,binary_mask,image)
#               print('After grouping:',unique_colors,len(unique_colors),unique_percentages)

#               background_colors = color_name_matrix[binary_mask == 1]
#               unique_bg_colors, bg_counts = np.unique(background_colors, return_counts=True)
#               background_lab_values = color_map_final[binary_mask == 1]
#               background_lab_values = np.array(background_lab_values.tolist())[0]

#               lab_final = extract_representative_labs(unique_colors, color_name_matrix, color_map_final)

#               unique_colors2=np.concatenate((unique_colors, unique_bg_colors))
#               lab_final2 = lab_final + [background_lab_values.tolist()]

#               save_path=save_path3+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors2, lab_final2,save_path,unique_bg_colors.item())
#               save_path=save_path4+Names[idx]
#               plot_color_palette(unique_colors, lab_final,save_path)



#           else:
#               img=Images[idx]
#               image = url_to_cv2_image(img)
#               color_df = pd.DataFrame(worksheet2.get_all_records())
#               sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#               # print('Before grouping:',sorted_color_names,sorted_lab)
#               lookup_table = generate_lookup_table3(sorted_color_names, sorted_lab, threshold=14)

#               color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors(lookup_table, sorted_color_names, sorted_lab, initial_color_array)
#               save_path=save_path1+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors, lab_final,save_path)
#               save_path=save_path2+Names[idx]
#               plot_color_palette(unique_colors, lab_final,save_path)
#               print('After grouping:',unique_colors,len(unique_colors),unique_percentages)

#               save_path=save_path3+Names[idx]
#               visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors, lab_final,save_path)
#               save_path=save_path4+Names[idx]
#               plot_color_palette(unique_colors, lab_final,save_path)



# color_data = pd.DataFrame(worksheet3.get_all_records())
# # Converting string representation of list to actual list
# color_data['Similar_name'] = color_data['Similar_name'].apply(ast.literal_eval)
# # Create a dictionary to map each color to its similar colors
# similar_colors_dict = dict(zip(color_data['Color_name'], color_data['Similar_name']))


# i=435
# img=Images[i]

# if continuous[i]=='Y':
#     plot_and_save_image_from_url(img)
#     if str(isbw[i])=='0':
#         idx=i
#         print(idx,Names[idx])
#         print('Background color:',background[idx])
#         if ',' in background[idx]:
#             bkg_color=background[idx].split(',')
#         else:
#             bkg_color=background[idx].split(' ')

#         if bkg_color!=['no']:
#             color_df = pd.DataFrame(worksheet2.get_all_records())
#             img=Images[idx]
#             image = url_to_image(img)
#             sorted_color_names, sorted_color_percentages,sorted_lab,sorted_counts,binary_mask = analyze_image_colors_and_names_nobkg(image, color_df,bkg_color)
#             plot_color_palette(sorted_color_names, sorted_lab,'continouous_color_ori_palette.pdf')

#         else:
#             color_df = pd.DataFrame(worksheet2.get_all_records())
#             img=Images[idx]
#             image = url_to_image(img)
#             sorted_color_names, sorted_color_percentages,sorted_lab,initial_color_array = analyze_image_colors_and_names_lab_array(image, color_df)
#             print(sorted_color_names)
#             plot_color_bar_chart(sorted_color_names, sorted_color_percentages, sorted_lab,'continuous_color_ori_distribution.pdf')
#             plot_color_palette(sorted_color_names, sorted_lab,'continouous_color_ori_palette.pdf')
#             visualize_color_name_map(initial_color_array, sorted_color_names, sorted_lab)

#             bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#             bw_perc = get_subcolor_percentages(sorted_color_names, sorted_color_percentages, bw_colors)
#             bw_perc_idx = get_subcolor_indices(sorted_color_names, sorted_color_percentages, bw_colors)


#         if bkg_color!=['no']:
#             color_df = pd.DataFrame(worksheet2.get_all_records())
#             img=Images[idx]
#             image = url_to_image(img)

#             color_df = pd.DataFrame(worksheet2.get_all_records())
#             sorted_color_names_wb, sorted_color_percentages_wb,sorted_lab_wb,initial_color_array_wb = analyze_image_colors_and_names_lab_array(image, color_df)
#             plot_color_bar_chart(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,'continuous_color_ori_distribution.pdf')
#             visualize_color_name_map(initial_color_array_wb, sorted_color_names_wb, sorted_lab_wb)

#             bw, color_colors, color_lab_values,bw_colors,bw_lab_values = check_lab_values_withthreshold(sorted_color_names_wb, sorted_lab_wb,threshold=14)
#             print(sorted_color_names_wb)
#             print(bw_colors)
#             bw_perc = get_subcolor_percentages(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)
#             bw_perc_idx = get_subcolor_indices(sorted_color_names_wb, sorted_color_percentages_wb, bw_colors)

#             sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_final_wbw(sorted_color_names_wb, sorted_color_percentages_wb, sorted_lab_wb,initial_color_array_wb,similar_colors_dict,image,binary_mask,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#             plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid,'continuous_color_merge_distribution.pdf')
#             lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#             print(lookup_table)
#             color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors_with_mask(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid,binary_mask,image)
#             print('After grouping:',unique_colors,len(unique_colors))

#             visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid,'continuous_image_merge_circle.pdf')
#             plot_color_palette(sorted_color_names_mid, sorted_lab_mid,'continouous_color_merge_palette.pdf')

#             lab_final = extract_representative_labs(unique_colors, color_name_matrix, color_map_final)

#             plot_color_palette(unique_colors, lab_final,'continouous_color_final_palette.pdf')
#             visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors, lab_final,'continuous_image_final_circle.pdf')



#         else:
#             color_df = pd.DataFrame(worksheet2.get_all_records())

#             img=Images[idx]
#             image = url_to_image(img)
#             sorted_color_names_mid,sorted_color_percentages_mid,sorted_lab_mid,lab_map_mid,color_map_mid=merge_similar_colors_nobackground(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict,bw_colors,bw_perc,bw_lab_values,bw_perc_idx)
#             print(sorted_color_names_mid)
#             print(bw_colors)
#             plot_color_bar_chart(sorted_color_names_mid, sorted_color_percentages_mid, sorted_lab_mid,'continuous_color_merge_distribution.pdf')
#             # visualize_color_name_map(color_map_mid, sorted_color_names_mid, sorted_lab_mid)

#             # visualize_color_regions_with_colored_circles(color_map_mid, sorted_color_names_mid, sorted_lab_mid)
#             visualize_color_regions_with_colored_fixedcircles(color_map_mid, sorted_color_names_mid, sorted_lab_mid,'continuous_image_merge_circle.pdf')

#             plot_color_palette(sorted_color_names_mid, sorted_lab_mid,'continouous_color_merge_palette.pdf')

#             if len(bw_colors)>0:
#                 lookup_table = generate_lookup_table3(bw_colors, bw_lab_values, threshold=14)
#                 color_name_matrix,color_map_final,unique_colors,unique_percentages = merge_and_convert_colors(lookup_table, sorted_color_names_mid, sorted_lab_mid, color_map_mid)
#                 print('After grouping:',unique_colors,len(unique_colors))
#                 lab_final = extract_representative_labs(unique_colors, color_name_matrix, color_map_final)
#                 plot_color_palette(unique_colors, lab_final,'continouous_color_final_palette.pdf')

#                 visualize_color_regions_with_colored_fixedcircles(color_name_matrix, unique_colors, lab_final,'continuous_image_final_circle.pdf')

#             else:
#                 color_map,sorted_color_names_final,sorted_color_percentages_final,sorted_lab_final,color_map_final=merge_similar_colors_nobw(sorted_color_names, sorted_color_percentages, sorted_lab,initial_color_array,similar_colors_dict)
#                 print('After grouping:',sorted_color_names_final,len(sorted_color_names_final))


#             lab_tuples = np.array([[tuple(lab) for lab in row] for row in color_map_final])
#             unique_lab_values = np.unique(lab_tuples.reshape(-1, lab_tuples.shape[-1]), axis=0)
#             # merge_path='/content/drive/MyDrive/1800color/output_image/Merge similar/'+Names[idx]
#             # difference_path='/content/drive/MyDrive/1800color/output_image/Lab difference/'+Names[idx]
#             # display_lab_image(image,color_map_final,merge_path,difference_path)

