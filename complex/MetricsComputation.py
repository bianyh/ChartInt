import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
import zstandard as zstd

class ImageFeatureExtractor:
    def __init__(self):
        self.feature_methods = {
            'O.H': self.calculate_glcm_heterogeneity,
            'O.CF': self.calculate_colorfulness,
            'O.ED': self.calculate_edge_density,
            'O.FP': self.interest_point,
            'O.IE': self.calculate_image_entropy,
            'O.ERGB': self.calculate_rgb_entropy,
            'O.KC': self.compression_score,
            'O.IG': self.calculate_information_gain_complexity,
        }

    def calculate_information_gain_complexity(self, image_path):
        # Read and resize image
        n = 4
        image = Image.open(image_path)
        width, height = image.size
        new_width, new_height = width // n, height // n
        image = image.resize((new_width, new_height), Image.LANCZOS)
        img_array = np.array(image)

        # Map RGB values to color levels
        num_colors = 32
        color_range = 256 / num_colors
        img_mapped = np.floor_divide(img_array, color_range).astype(int)

        # Convert RGB color levels to single integer values
        img_mapped_flattened = (img_mapped[:, :, 0] * (num_colors ** 2) +
                                img_mapped[:, :, 1] * num_colors +
                                img_mapped[:, :, 2])

        # Calculate color probabilities
        unique_colors, color_counts = np.unique(img_mapped_flattened, return_counts=True)
        total_pixels = new_height * new_width
        color_probs = color_counts / total_pixels

        # Calculate joint probabilities
        joint_counts = np.zeros((len(unique_colors), len(unique_colors)))
        for i in range(new_height - 1):
            for j in range(new_width):
                idx1 = np.where(unique_colors == img_mapped_flattened[i, j])[0][0]
                idx2 = np.where(unique_colors == img_mapped_flattened[i + 1, j])[0][0]
                joint_counts[idx1, idx2] += 1
        joint_probs = joint_counts / total_pixels

        # Calculate conditional probabilities
        cond_probs = joint_probs / color_probs.reshape(-1, 1)

        # Calculate complexity G
        G = -np.sum(joint_probs[joint_probs > 0] * np.log2(cond_probs[joint_probs > 0]))

        return G

    def calculate_colorfulness(self, image_path):
        # Read and resize image
        n = 4
        image = Image.open(image_path)
        width, height = image.size
        new_width, new_height = width // n, height // n
        image = image.resize((new_width, new_height), Image.LANCZOS)
        image = np.array(image)

        # Calculate colorfulness
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        return colorfulness

    def calculate_edge_density(self, image_path):
        # Read image and calculate edge density
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 5, 125)
        edge_pixels = np.sum(edges == 255)
        total_pixels = image.size
        edge_density = edge_pixels / total_pixels
        return edge_density

    def interest_point(self, image_path):
        # ORB feature detection with non-maximum suppression
        nfeatures, scaleFactor, nlevels, edgeThreshold = 2000, 1.5, 10, 31
        nms_radius = 10

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, edgeThreshold=edgeThreshold)
        keypoints = orb.detect(gray, None)

        # Non-maximum suppression
        corners = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        nms_corners = []
        while corners:
            max_corner = max(corners, key=lambda c: keypoints[corners.index(c)].response)
            nms_corners.append(max_corner)
            corners = [(x, y) for x, y in corners
                       if abs(x - max_corner[0]) > nms_radius or abs(y - max_corner[1]) > nms_radius]

        return len(nms_corners)

    def calculate_image_entropy(self, image_path):
        # Calculate image entropy
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (11, 11), 0)
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        histogram_normalized = histogram / histogram.sum()
        histogram_normalized = histogram_normalized[histogram_normalized > 0]
        entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized))
        return entropy

    def calculate_rgb_entropy(self, image_path):
        # Calculate RGB entropy
        img = cv2.imread(image_path)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        binnum = 256
        hist, _ = np.histogramdd(img.reshape(-1, 3), bins=(binnum, binnum, binnum), range=((0, 255), (0, 255), (0, 255)), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def compression_score(self, image_path):
        # Calculate compression score
        img = cv2.imread(image_path)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        _, encoded_img = cv2.imencode('.png', img)
        png_data = np.array(encoded_img).tobytes()
        compressor = zstd.ZstdCompressor(level=11)
        compressed_data = compressor.compress(png_data)
        return len(compressed_data)

    def calculate_glcm_properties(self, image_path):
        # Calculate GLCM properties
        img = Image.open(image_path)
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        new_width, new_height = width // 4, height // 4
        img_np = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        if img_np.ndim == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        glcm = graycomatrix(img_gray, [1], [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        heterogeneity = -graycoprops(glcm, 'homogeneity').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        return contrast, correlation, energy, heterogeneity, dissimilarity

    def calculate_glcm_heterogeneity(self, image_path):
        _, _, _, heterogeneity, _ = self.calculate_glcm_properties(image_path)
        return heterogeneity

    def extract_features(self, image_path):
        features = {}
        for feature_name, method in self.feature_methods.items():
            features[feature_name] = method(image_path)
        return features

def process_images(excel_path, folder_path, output_path):
    df = pd.read_excel(excel_path)
    extractor = ImageFeatureExtractor()

    results = {feature: [] for feature in extractor.feature_methods.keys()}

    for image_name in df['ImageName']:
        image_path = os.path.join(folder_path, image_name)
        features = extractor.extract_features(image_path)
        for feature_name, value in features.items():
            results[feature_name].append(value)
        print(f"Processed {image_name}")

    for feature_name, values in results.items():
        df[feature_name] = np.array(values)

    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # excel_path = 'ImageList.xlsx'
    # folder_path = '1800Image'
    # output_path = 'ImageMetrics.xlsx'
    # process_images(excel_path, folder_path, output_path)
    extractor = ImageFeatureExtractor()
    # Example usage
    features = extractor.extract_features('C:\\Users\\11494\\Desktop\\data\\output\\output\\103\\chart.png')
    print(features)
