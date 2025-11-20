def rename_and_copy_czi_files(main_folder_path):
    import os
    import shutil
    """
    Rename all .czi files in each subfolder of a directory to the name of the folder with an incremental number at the end,
    and copy them to a new "All" folder.

    Parameters
    ----------
    main_folder_path : str
        Path to the main folder containing subfolders with .czi files.
    Returns
    -------
    None
    """
    # First, rename all .czi files
    for root, dirs, files in os.walk(main_folder_path):
        for folder_name in dirs:
            if folder_name == "All":
                continue  # Skip the "All" folder
            folder_path = os.path.join(root, folder_name)
            
            # Get a list of all CZI files in the folder
            czi_files = [file for file in os.listdir(folder_path) if file.endswith(".czi")]
            
            # Sort the files to ensure they are renamed in the order they appear in the folder
            czi_files.sort()

            # Rename the CZI files
            for j, czi_file in enumerate(czi_files):
                new_name = f"{folder_name}_{j+1:02d}.czi"
                os.rename(os.path.join(folder_path, czi_file), os.path.join(folder_path, new_name))

    # Create the "All" folder if it doesn't exist
    all_folder_path = os.path.join(main_folder_path, "All")
    if not os.path.exists(all_folder_path):
        os.makedirs(all_folder_path)

    # Then, copy all renamed .czi files into the "All" folder
    for root, dirs, files in os.walk(main_folder_path):
        for folder_name in dirs:
            if folder_name == "All":
                continue  # Skip the "All" folder
            folder_path = os.path.join(root, folder_name)
            
            # Get a list of all renamed CZI files in the folder
            czi_files = [file for file in os.listdir(folder_path) if file.endswith(".czi")]
            
            # Copy the files to the "All" folder, skipping repeat files
            for file_name in czi_files:
                src_file_path = os.path.join(folder_path, file_name)
                dst_file_path = os.path.join(all_folder_path, file_name)
                
                if not os.path.exists(dst_file_path):
                    shutil.copy2(src_file_path, dst_file_path)

def Convert_czi_to_tiff(folder_path):
    import os
    import czifile
    from tifffile import imsave
    import shutil
    import cv2
    import numpy as np
    
    # Create the "CZI" folder in each folder path
    czi_folder_path = os.path.join(folder_path, "CZI")
    os.makedirs(czi_folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".czi"):
            czi_path = os.path.join(folder_path, filename)
            with czifile.CziFile(czi_path) as czi:
                image_arrays = czi.asarray()
                for channel_idx, channel_image in enumerate(image_arrays):
                    # Convert the image to 16-bit
                    channel_image_16bit = cv2.normalize(channel_image, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
                    
                    tiff_path = os.path.splitext(czi_path)[0] + f"_C_{channel_idx}.tiff"
                    imsave(tiff_path, channel_image_16bit)

            # Move the CZI files to the "CZI" folder
            new_czi_path = os.path.join(czi_folder_path, filename)
            shutil.move(czi_path, new_czi_path)

def Background_Subtraction(folder_path, radius, display, save):
    import os
    import warnings
    from skimage import io, img_as_float32
    from skimage.restoration import rolling_ball
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    """
    Apply a 50-pixel radius rolling ball background subtraction to TIFF files in a folder, convert to 32-bit, and save the results.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    radius : int, optional
        Radius of the rolling ball. Default is 50.

    Returns
    -------
    None
    """
    # Create a subfolder for the processed images
    processed_folder = os.path.join(folder_path, "0_Background Subtraction")
    os.makedirs(processed_folder, exist_ok=True)

    # Get a list of all TIFF files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]

    if not image_files:
        return

    # Suppress low contrast image warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image.*")

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        # Read the image
        image = io.imread(image_path)

        # Apply the rolling ball algorithm for background subtraction
        background = rolling_ball(image, radius=radius)
        subtracted_image = image - background

        # Ensure the pixel values are within the valid range
        subtracted_image = np.clip(subtracted_image, 0, np.max(subtracted_image))

        # Convert the image to 32-bit
        subtracted_image_32bit = img_as_float32(subtracted_image)

        if save:
            # Save the processed image to the processed folder
            processed_image_path = os.path.join(processed_folder, image_file)
            io.imsave(processed_image_path, subtracted_image_32bit)
        
        if display:
            # Display the original and background subtracted images side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=DEFAULT_DPI)
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(subtracted_image_32bit, cmap='gray')
            axes[1].set_title('Background Subtracted Image')
            axes[1].axis('off')
            plt.show()


def optimize_contrast(folder_path, display, save):
    import skimage
    from skimage import io, exposure
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    """Optimize the contrast of images in a folder and display the results

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the images

    Returns
    -------
    None
    """
    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        img = skimage.io.imread(image_path)

        # Normalize the image to be between -1 and 1
        img = img / np.max(np.abs(img))

        # Optimize the contrast of the image
        img_optimized = exposure.equalize_adapthist(img)

        # Specify the file path of the optimized image
        optimized_image_path = os.path.join(folder_path, image_file)

        if save:
            # Save the optimized image
            skimage.io.imsave(optimized_image_path, img_optimized)

        if display:
            # Display the original image and the optimized image side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(img_optimized, cmap='gray')
            axes[1].set_title('Optimized Image')
            axes[1].axis('off')
            plt.show()


def Colorize_Composite(folder_path, display_results=False, save_results=True):
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from skimage.io import imread as load_image
    from skimage.color import gray2rgb
    """
    Colorize specified TIFF images and create composite images.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the TIFF images.
    display_results : bool, optional
        Whether to display the images. Default is False.
    save_results : bool, optional
        Whether to save the images. Default is True.

    Returns
    -------
    None
    """
    # Create a new output path that includes the "Colorized" folder
    output_path = os.path.join(folder_path, "Colorized")

    # Create the "Colorized" folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    def create_colormap(color):
        colors = ["black", color]
        return mcolors.LinearSegmentedColormap.from_list("", colors)
    
    # Get a list of base file names for all TIFF images ending in _C_0
    base_files = [os.path.splitext(os.path.basename(f))[0].rsplit('_C_0', 1)[0] for f in glob.glob(folder_path + "/*_C_0.tiff")]

    # Define the colors for each channel
    channel_colors = ["green", "red", "blue"]
    rgb_colors = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]  # green, red, blue

    # Iterate through each base file name
    for base_name in base_files:
        channel_images = []
        for i in range(3):
            channel_path = os.path.join(folder_path, f"{base_name}_C_{i}.tiff")
            if os.path.exists(channel_path):
                channel_image = load_image(channel_path).astype(np.float32)  # Use float32 for compatibility
                channel_images.append(channel_image)
            else:
                print(f"Channel {i} not found for {base_name}")
                break

        if len(channel_images) != 3:
            print(f"Skipping {base_name} due to missing channels")
            continue

        cmap_colors = [create_colormap(color) for color in channel_colors]

        # Create individual colorized images
        colorized_images = []
        for i, channel_image in enumerate(channel_images):
            vmax = np.percentile(channel_image, 99.9)
            vmin = np.percentile(channel_image, 0.1)

            # Normalize the channel image and apply the RGB color
            normalized_channel = (channel_image - vmin) / (vmax - vmin)
            normalized_channel = np.clip(normalized_channel, 0, 1)
            colorized_image = gray2rgb(normalized_channel) * np.array(rgb_colors[i], dtype=np.float32)
            colorized_images.append(colorized_image)

            # Save the individual colorized image
            if save_results:
                channel_output_path = os.path.join(output_path, f'{base_name}_C_{i}.png')
                plt.imsave(channel_output_path, colorized_image)
                print(f"Saved colorized image: {channel_output_path}")

        # Create Composite Image with a black backdrop
        merged = np.zeros_like(colorized_images[0], dtype=np.float32)
        for colorized_image in colorized_images:
            merged += colorized_image

        # Clip the merged image to ensure values are within the valid range
        merged = np.clip(merged, 0, 1)

        # Display or save the composite image
        fig = plt.figure(figsize=(merged.shape[1]/100, merged.shape[0]/100))
        plt.axis(False)
        plt.imshow(merged)

        # Save the composite image with the specified output path
        composite_output_path = os.path.join(output_path, f'{base_name}_composite.png')
        if save_results:
            plt.savefig(composite_output_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            print(f"Saved composite image: {composite_output_path}")
        if display_results:
            plt.show()
        plt.close()

def Apply_Scale_Bar(colorized_image_path, Objective, display=False, save=True):
    """
    Apply a scale bar to images in a folder.

    Parameters
    ----------
    colorized_image_path : str
        Path to the folder containing the images.
    Objective : int
        Objective magnification (10 or 20).
    display : bool, optional
        Whether to display the images with the scale bar. Default is False.
    save : bool, optional
        Whether to save the images with the scale bar. Default is True.

    Returns
    -------
    None
    """
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib_scalebar.scalebar import ScaleBar

    # Get a list of all TIFF and PNG files in the folder
    image_files = [file for file in os.listdir(colorized_image_path) if file.endswith('.tiff') or file.endswith('.png')]

    # Set scale bar parameters based on the Objective value
    if Objective == 20:
        scale = 0.335
        length_fraction = 0.22
    elif Objective == 10:
        scale = 0.641
        length_fraction = 0.12
    else:
        raise ValueError("Unsupported Objective value. Only 10 and 20 are supported.")

    for image_file in image_files:
        # Skip images that end with "_composite"
        if image_file.endswith("_composite.png") or image_file.endswith("_composite.tiff"):

            continue
        # Construct the full path to the image file
        image_path = os.path.join(colorized_image_path, image_file)

        # Open the image file
        image = Image.open(image_path)

        # Color_Shade = 'white' if image_file.endswith("composite.png") else 'black'
        Color_Shade = 'white'

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image, cmap='gray')

        # Add the scale bar
        scalebar = ScaleBar(scale, 'um', length_fraction=length_fraction, color=Color_Shade, box_color='None', location='lower right')

        # Add the scale bar to the axes
        ax.add_artist(scalebar)

        # Remove the axis labels and ticks
        ax.axis('off')

        # Save the figure with the scale bar if save is enabled
        if save:
            output_path = os.path.join(colorized_image_path, image_file)
            plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


        # Display the figure if display is enabled
        if display:
            plt.show()

        # Close the figure to free memory
        plt.close(fig)


#For 3 channel
def Stack_Images_Row(colorized_image_path):
    import os
    import matplotlib.pyplot as plt
    from itertools import groupby
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Create the "0_Stacked_Row" folder if it does not exist
    output_path = os.path.join(colorized_image_path, "0_Stacked_Row")
    os.makedirs(output_path, exist_ok=True)

    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(colorized_image_path) if file.endswith('.png')]

    # Sort the image files based on their names
    image_files.sort()

    # Group the image files by their base names (everything before the last "_C" or "_composite")
    def get_base_name(filename):
        if '_composite' in filename:
            return filename.split('_composite')[0]
        elif '_C_' in filename:
            return '_'.join(filename.split('_C_')[:-1])  # Everything before the last "_C_"
        else:
            return os.path.splitext(filename)[0]  # Fallback to the full name without extension

    image_files_grouped = [list(group) for key, group in groupby(image_files, key=get_base_name)]

    # Iterate over the groups of image files
    for image_files in image_files_grouped:
        # Sort the image files so that the composite image is first, followed by C_0, C_1, and C_2
        image_files.sort(key=lambda x: (not x.endswith('_composite.png'), '_C_0' not in x, '_C_1' not in x, '_C_2' not in x))

        # Calculate the number of rows and columns for the subplots
        nrows = 1
        ncols = len(image_files)  # One column for each image in the group

        # Create a new figure for each group of images
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 10))
        
        # Make sure that axes is always a list
        axes = [axes] if ncols == 1 else axes

        # Extract the base name for the title
        base_name = get_base_name(image_files[0])

        # Add the title above the figure
        fig.suptitle(base_name, fontsize=10, color='white', y=0.67)

        # Iterate over the image files in the group
        for i, image_file in enumerate(image_files):
            # Read the image
            image_path = os.path.join(colorized_image_path, image_file)
            image = plt.imread(image_path)

            # Plot the image in the corresponding subplot
            ax = axes[i % ncols]
            ax.imshow(image)
            ax.axis('off')  # Remove the axis

        # Save the figure using the base name of the first image in the group
        output_file = os.path.join(output_path, f"{base_name}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
        plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory

def Stack_Images_Column(colorized_image_path):
    import os
    import matplotlib.pyplot as plt
    from itertools import groupby
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Create the "Compiled" folder if it does not exist
    output_path = os.path.join(colorized_image_path, "0_Stacked_Column")
    os.makedirs(output_path, exist_ok=True)

    # Get a list of all image files in the folder
    image_files = [file for file in os.listdir(colorized_image_path) if file.endswith('.png')]

    # Sort the image files based on their names
    image_files.sort()

    # Group the image files by their base names (first 2 elements)
    image_files_grouped = [list(group) for key, group in groupby(image_files, lambda x: '_'.join(x.split('_')[:2]))]

    # Iterate over the groups of image files
    for image_files in image_files_grouped:
        # Sort the image files so that the composite image is first
        image_files.sort(key=lambda x: (not x.endswith('composite.png'), x))

        # Calculate the number of rows and columns for the subplots
        ncols = 1
        nrows = 4  # One row for each image in the group

        # Create a new figure for each group of images
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5*nrows))
        
        # Make sure that axes is always a list
        axes = [axes] if nrows == 1 else axes

        # Iterate over the image files in the group
        for i, image_file in enumerate(image_files):
            # Read the image
            image_path = os.path.join(colorized_image_path, image_file)
            image = plt.imread(image_path)

            # Plot the image in the corresponding subplot
            ax = axes[i % nrows]
            ax.imshow(image)
            ax.axis('off')  # remove the axis

            # Add title to the composite image
            if image_file.endswith("composite.png"):
                ax.set_title(image_file, color='white', fontsize=8)

        # Save the figure
        output_file = os.path.join(output_path, image_files[0].rsplit('_', 1)[0] + '.png')
        plt.tight_layout()
        plt.savefig(output_file, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free memory
