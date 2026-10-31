import cv2
import numpy as np
import os

class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def check_package_versions(self):

        import numpy as np
        import matplotlib
        import cv2

        print(np.__version__)
        print(matplotlib.__version__)
        print(cv2.__version__)

    def load_and_analyze_image(self):
        image1 = self.image
        pixel_dtype = image1[0,0].dtype
        print(f"Image data type: {image1.dtype}")
        print(f"Pixel data type: {pixel_dtype}")
        print(f"Image dimensions: {image1.shape}")

        return image1.dtype, pixel_dtype, image1.shape

    def create_red_image(self):
        red_image = self.image.copy()
        red_image[:,:,0] = 0
        red_image[:,:,1] = 0
        return red_image

    def create_photographic_negative(self):
        negative_image = 255 - self.image
        return negative_image

    def swap_color_channels(self):
        swapped_image = self.image.copy()[:,:,::-1]
        return swapped_image

    def foliage_detection(self):
        blue = self.image[:,:,0]
        green = self.image[:,:,1]
        red = self.image[:,:,2]
        mask = (green >= 50) & (red < 50) & (blue < 50)
        return mask.astype(np.uint8) * 255

    def shift_image(self):
        shifted_image = np.zeros_like(self.image)
        shifted_image[100:, 200:] = self.image[:-100,:-200]
        return shifted_image

    def rotate_image(self):
        rotated_image = self.image.copy()
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        h, w = self.image.shape[:2]
        matrix = cv2.getRotationMatrix2D((0,0),-theta, scale)
        matrix[0,2] += shift[0]
        matrix[1,2] += shift[1]
        transformed_image = cv2.warpAffine(self.image, matrix,(w, h), flags = cv2.INTER_NEAREST)

        return transformed_image

    def convert_to_grayscale(self):
        b = self.image[:,:,0].astype(np.float64)
        g = self.image[:,:,1].astype(np.float64)
        r = self.image[:,:,2].astype(np.float64)
        return np.round((3 * r + 6 * g + b) / 10).astype(np.uint8)
        

    def compute_moments(self):

        m00 = np.sum(self.binary_image)
        y,x = np.indices(self.binary_image.shape)
        m10 = np.sum(x * self.binary_image)
        m01 = np.sum(y* self.binary_image)
        m20 = np.sum(x**2 * self.binary_image)
        m02 = np.sum(y**2 * self.binary_image)
        m11 = np.sum(x * y * self.binary_image)

        x_bar = m10 / m00
        y_bar = m01 / m00

        dx = x - x_bar
        dy = y - y_bar

        mu11 = np.sum(dx * dy * self.binary_image)
        mu20 = np.sum(dx**2 * self.binary_image)
        mu02 = np.sum(dy**2 * self.binary_image)
        
        print("First-Order Moments:")
        print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        print("Centralized Moments:")
        print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        print("Second-Order Centralized Moments:")
        print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):

        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()
        orientation = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
        a = mu20 / m00
        b = mu11 / m00
        c = mu02 / m00

        matrix = np.array([[a,b],[b,c]])
        lambda2, lambda1 = np.linalg.eigvalsh(matrix)
        eccentricity = np.sqrt((lambda1 - lambda2) / lambda1)
        full_axes1 = 4 * np.sqrt(lambda1)
        full_axes2 = 4 * np.sqrt(lambda2)

        half_axes = (int(full_axes1 / 2), int(full_axes2 / 2))
        center = (int(x_bar), int(y_bar))

        glasses_with_ellipse = self.image.copy()
        cv2.ellipse(glasses_with_ellipse,center, half_axes, float(orientation), 0,360,(0,0,255),1)

        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":

    assignment = ComputerVisionAssignment("original_image.png", "binary_image.png")

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()
    #cv2.imshow("Red Image", red_image)
    #cv2.waitKey(0)


    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()
    #cv2.imshow("Negative Image", negative_image)
    #cv2.waitKey(0)

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()
    #cv2.imshow("Swapped Image", swapped_image)
    #cv2.waitKey(0)

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()
    #cv2.imshow("Foliage Image", foliage_image)
    #cv2.waitKey(0)

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()
    #cv2.imshow("Shifted Image", shifted_image)
    #cv2.waitKey(0)

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()
    #cv2.imshow("Rotated Image", rotated_image)
    #cv2.waitKey(0)

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )
    #cv2.imshow("Transformed Image", transformed_image)
    #cv2.waitKey(0)

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()
    #cv2.imshow("Grayscale Image", gray_image)
    #cv2.waitKey(0)

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )

    #cv2.imshow("Glasses with Ellipse", glasses_with_ellipse)
    #cv2.waitKey(0)
