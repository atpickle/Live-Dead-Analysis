import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def Mean_Shift_Array(folder_path, show_test, save):
    """
    Applies mean shift filtering to images in a folder and saves the resulting images to a new folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the images.
    show_test : bool, optional
        Flag to display the processed images. Default is True.
    save : bool, optional
        Flag to save the processed images. Default is True.

    Returns
    -------
    None
    """
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff') or file.endswith('.tif')]

    # Iterate over each image file
    for image_file in image_files:
        # Check if the image file ends with C_0 or C_1
        if image_file.endswith('C_0.tiff') or image_file.endswith('C_1.tiff'):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, image_file)

            # Read the image
            image = cv2.imread(image_path)

            # Apply mean shift filtering with different parameters
            shifted_images = [
                cv2.pyrMeanShiftFiltering(image, 30, 30),
                cv2.pyrMeanShiftFiltering(image, 30, 20),
                cv2.pyrMeanShiftFiltering(image, 30, 10),
                cv2.pyrMeanShiftFiltering(image, 20, 30),
                cv2.pyrMeanShiftFiltering(image, 20, 20),
                cv2.pyrMeanShiftFiltering(image, 20, 10),
                cv2.pyrMeanShiftFiltering(image, 10, 30),
                cv2.pyrMeanShiftFiltering(image, 10, 20),
                cv2.pyrMeanShiftFiltering(image, 10, 10)
            ]

            # Create a figure with a 3x3 grid of subplots
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            # Iterate over each shifted image and its respective sp and sr values
            for i, (shifted, sp, sr) in enumerate(zip(shifted_images, [30, 30, 30, 20, 20, 20, 10, 10, 10], [30, 20, 10, 30, 20, 10, 30, 20, 10])):
                # Calculate the row and column indices for the subplot
                row = i // 3
                col = i % 3

                # Display the shifted image in the corresponding subplot
                axes[row, col].imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f"sp={sp}, sr={sr}")  # Set the figure title as sp and sr
                axes[row, col].axis('off')

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

          

            if save:
                  # Create the "Mean_Shift_Arrays" folder in the specified folder path
                output_folder = os.path.join(folder_path, "Mean Shift Arrays")
                os.makedirs(output_folder, exist_ok=True)
                # Save the plot to the "Mean_Shift_Arrays" folder
                plot_name = os.path.splitext(image_file)[0] + "_mean_shift.png"
                plot_path = os.path.join(output_folder, plot_name)
                plt.savefig(plot_path, bbox_inches='tight', dpi=300, pad_inches=0.1, transparent=False)

            if show_test:
                # Show the plot
                plt.show()
            plt.close()


def Apply_Otsu(folder_path, SR, CR, save, display):
    """
    Apply mean shift filtering and Otsu's thresholding to all .tif and .tiff images in the specified folder.
    Save and/or display the results based on the provided flags.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    SR : int
        Spatial radius for mean shift filtering.
    CR : int
        Color radius for mean shift filtering.
    save : bool, optional
        Flag to save the thresholded images. Default is True.
    display : bool, optional
        Flag to display the original and thresholded images. Default is True.

    Returns
    -------
    None
    """
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff') or file.endswith('.tif')]

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        # Read the image
        image = cv2.imread(image_path)

        # Apply mean shift filtering
        shifted = cv2.pyrMeanShiftFiltering(image, SR, CR)

        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if save:
            # Create the "Thresholded" folder in the specified folder path
            output_folder = os.path.join(folder_path, "Thresholded")
            os.makedirs(output_folder, exist_ok=True)
            # Save the thresholded image to the "Thresholded" folder
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, thresh)

        if display:
            # Create a figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 10))

            # Display the original image
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title(image_file)  # Set the image title as the file name
            axes[0].axis('off')

            # Display the mean shift filtered image
            axes[1].imshow(gray, cmap='gray')
            axes[1].set_title('Mean Shift Filtered')
            axes[1].axis('off')

            # Display the thresholded image
            axes[2].imshow(thresh, cmap='gray')
            axes[2].set_title('Otsu Thresholded')
            axes[2].axis('off')

            plt.show()
            plt.close()


def Erosion_Test(folder_path, S1, S2, S3, display, save):
    """
    Apply erosion with different structuring element sizes to all .tiff images in the specified folder.
    Display and/or save the results based on the provided flags.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    S1, S2, S3 : int
        Sizes of the structuring elements for erosion.
    display : bool, optional
        Flag to display the original and processed images. Default is True.
    save : bool, optional
        Flag to save the processed images. Default is True.

    Returns
    -------
    None
    """
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]

    # Iterate over each image file
    for image_file in image_files:
        # Skip DAPI Images
        if  image_file.endswith('C_1.tiff'):
            continue

        image_path = os.path.join(folder_path, image_file)

        # Read the image
        image = cv2.imread(image_path)

        # Define the size of the structuring elements
        structuring_element_sizes = [S1, S2, S3]

        # Create the figure and subplots outside of the loop
        fig, axes = plt.subplots(1, len(structuring_element_sizes) + 1, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Display the original image in the first column
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(image_file)  # Set the image title as the file name
        axes[0].axis('off')

        # Iterate over each structuring element size
        for i, size in enumerate(structuring_element_sizes):
            # Create the structuring element
            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

            # Perform the erosion
            eroded_image = cv2.erode(image, structuring_element)

            if display:
                # Display the eroded images in the columns next to the original image
                axes[i + 1].imshow(eroded_image, cmap='gray')
                axes[i + 1].set_title(f"Structuring Element Size: {size}")  # Set the image title with the structuring element size
                axes[i + 1].axis('off')

        if save:
            # Create the folder in the specified folder path
            output_folder = os.path.join(folder_path, "Eroded_Arrays")
            os.makedirs(output_folder, exist_ok=True)

            # Save the plot to the folder
            plot_name = os.path.splitext(image_file)[0] + "_Live_Eroded.png"
            plot_path = os.path.join(output_folder, plot_name)
            plt.savefig(plot_path, bbox_inches='tight', dpi=300, pad_inches=0.1, transparent=False)

        if display:
            plt.show()
            plt.close()


def Calculate_Live_Dead(folder_path, SR, CR, Erosion_Size, Minimum_distance, display=True, save=True):
    import os
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage
    import imutils
    import pandas as pd
    from tifffile import imsave
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]

    # Create an empty DataFrame
    contour_table = pd.DataFrame(columns=['Image Name', 'Live', 'Dead', 'Total', 'True Total', 'Live %', 'Dead %'])

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        
        print(f"Processing image: {image_file}")

        image_path = os.path.join(folder_path, image_file)

        image = cv2.imread(image_path)

        # Apply Gaussian Filtering
        image = cv2.GaussianBlur(image, (1, 1), 0)

        # Apply mean shift filtering
        shifted = cv2.pyrMeanShiftFiltering(image, SR, CR)  # INPUT SPATIAL RADIUS AND COLOR RADIUS

        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Erode only live images: C_0 = live, C_1 = dead, C_2 = DAPI
        if image_file.endswith('C_0.tiff'):
            structuring_element_size = Erosion_Size
            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))
            eroded_image = cv2.erode(thresh, structuring_element)
            thresh = eroded_image

        D = ndimage.distance_transform_edt(thresh)
        coords = peak_local_max(D, min_distance=Minimum_distance, labels=thresh)  # INPUT MIN DISTANCE

        # Create a new array of the same shape as `thresh`, filled with False
        localMax = np.zeros_like(thresh, dtype=bool)

        # Set the peaks to True
        for coord in coords:
            localMax[tuple(coord)] = True

        # Perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        num_contours = len(np.unique(labels)) - 1

        # Remove everything after the last underscore in the image name
        # Find index of last underscore
        underscore_index = image_file.rfind('_')
        image_name = image_file[:underscore_index]
        cntid = image_file.split('_')[-1][0]

        column_names = {'0': 'Live', '1': 'Dead', '2': 'Total'}
        if image_name in contour_table['Image Name'].values:
            contour_table.loc[contour_table['Image Name'] == image_name, column_names[cntid]] = num_contours
        else:
            new_row = pd.DataFrame({'Image Name': [image_name], column_names[cntid]: [num_contours]})
            contour_table = pd.concat([contour_table, new_row], ignore_index=True)

    # Calculate the True Total, Live %, and Dead % columns
    contour_table['Live'] = contour_table['Live'].astype(float).fillna(0)
    contour_table['Dead'] = contour_table['Dead'].astype(float).fillna(0)
    contour_table['Total'] = contour_table['Total'].astype(float).fillna(0)
    contour_table['True Total'] = contour_table['Live'] + contour_table['Dead']
    contour_table['Live %'] = (contour_table['Live'] / contour_table['True Total']) * 100
    contour_table['Dead %'] = (contour_table['Dead'] / contour_table['True Total']) * 100

    # Save the contour_table DataFrame to an Excel file in the folder directory
    excel_file_path = os.path.join(folder_path, "Contour Count.xlsx")
    contour_table.to_excel(excel_file_path, index=False)

    # Iterate over each image file again to create and save the plots
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        image = cv2.imread(image_path)

        # Apply Gaussian Filtering
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Apply mean shift filtering
        shifted = cv2.pyrMeanShiftFiltering(image, SR, CR)  # INPUT SPATIAL RADIUS AND COLOR RADIUS

        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Erode only live images: C_0 = live, C_1 = dead, C_2 = DAPI
        if image_file.endswith('C_0.tiff'):
            structuring_element_size = Erosion_Size
            structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))
            eroded_image = cv2.erode(thresh, structuring_element)
            thresh = eroded_image

        D = ndimage.distance_transform_edt(thresh)
        coords = peak_local_max(D, min_distance=Minimum_distance, labels=thresh)  # INPUT MIN DISTANCE

        # Create a new array of the same shape as `thresh`, filled with False
        localMax = np.zeros_like(thresh, dtype=bool)

        # Set the peaks to True
        for coord in coords:
            localMax[tuple(coord)] = True

        # Perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        if display or save:
            # Create a 3 by 1 subplot with increased size and resolution
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Display the shifted image using Matplotlib
            axes[0].imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
            axes[0].axis('off')

            # Display the grayscale image using Matplotlib
            axes[1].imshow(thresh, cmap='gray')
            axes[1].axis('off')

            # Show the output image using Matplotlib
            axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[2].axis('off')

            fig_name = os.path.splitext(image_file)[0] + "_Count"
            plt.subplots_adjust(wspace=0.1)

        if save:
            # Create a new folder called "Count" in the specified folder path
            count_folder_path = os.path.join(folder_path, "Count")
            os.makedirs(count_folder_path, exist_ok=True)

            # Save the figure to the "Count" folder
            fig_path = os.path.join(count_folder_path, fig_name + ".png")
            plt.savefig(fig_path, bbox_inches='tight', dpi=300, pad_inches=0.1, transparent=True)

        if display:
            plt.show()
        plt.close()
