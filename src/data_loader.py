import vtk
import logging
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from torch.utils.data import Dataset

from utilities import spatial_resample_scan

# Set up logging
log = logging.getLogger(__name__)

class CoronaryArteryDataLoader(Dataset):
    def __init__(self, config):
        # Configuration settings
        self.base_dir = config.base_settings.base_dir
        self.data_original_dir = config.data_loader.data_original_dir
        self.data_processed_dir = config.data_loader.data_processed_dir
        self.data_segmentation_dir = config.data_loader.data_segmentation_dir

        self.file_list_dir = config.data_loader.file_list_dir
        self.file_list = self._load_file_list()

        self.voxel_spacing = config.data_loader.voxel_spacing
    
    def _load_file_list(self):
        """ Load file list from text file into a list """
        file_list = Path(f"{self.base_dir}/{self.file_list_dir}").read_text().splitlines()
        return file_list

    def __len__(self):
        """ Return the total number of samples in file list """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """ Load image, label, and centerline data for a single sample """
        # Retrieve image index of sample
        img_index = self.file_list[index]

        # Paths 
        img_path = f"{self.base_dir}/{self.data_original_dir}/all/{img_index}.img.nii.gz" 
        label_path = f"{self.base_dir}/{self.data_original_dir}/all/{img_index}.label.nii.gz"
        centerline_path = f"{self.base_dir}/{self.data_processed_dir}/{img_index}.img/{img_index}_lad_centerline_hu.vtk"

        # Load image data (CT scan)
        img_nii = sitk.ReadImage(img_path)
        img_nii = spatial_resample_scan(img_nii, self.voxel_spacing) # Resample image to isotropic pixel spacing

        # Get metadata of the nii file 
        origin = img_nii.GetOrigin()
        spacing = img_nii.GetSpacing()
        direction = img_nii.GetDirection()

        # Convert image to numpy array (phyiscal coordinates --> index coorinates)
        img = sitk.GetArrayFromImage(img_nii)

        # Convert from (z, y, x) to (x, y, z) https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
        img = img.transpose(2, 1, 0) 

        # Load label (segmentation of full coronary tree)
        label_nii = sitk.ReadImage(label_path)
        label_nii = spatial_resample_scan(label_nii, self.voxel_spacing)
        label = sitk.GetArrayFromImage(label_nii)
        label = label.transpose(2, 1, 0) # Convert from (z, y, x) to (x, y, z)

        # Load centerline (of left anterior descending artery)
        centerline_reader = vtk.vtkPolyDataReader()
        centerline_reader.SetFileName(centerline_path)
        centerline_reader.Update()
        centerline = centerline_reader.GetOutput()

        # Extract points (physical coordinates) from the centerline
        points = centerline.GetPoints()
        num_points = points.GetNumberOfPoints()
        centerline_points = np.array([points.GetPoint(i) for i in range(num_points)]) # [[x1, y1, z1], [x2, y2, z2], ...]

        # Transform centerline points from physical coordinates to index coordinates
        centerline_indices = np.array([label_nii.TransformPhysicalPointToIndex(point) for point in centerline_points])

        # Extract the HU values of the centerline points
        scalars = centerline.GetPointData().GetScalars()
        centerline_values = np.array([scalars.GetValue(i) for i in range(num_points)])

        # Info about the sample
        log.info("\n---------------------------------------- Info ------------------------------------------")
        log.info(f"Image index: {img_index}")
        log.info(f"Image shape: {img.shape}")
        log.info(f"Label shape: {label.shape}")
        log.info(f"Unique values in label: {np.unique(label)}")
        log.info(f"Number of unique values in label: {len(np.unique(label))}")
        log.info(f"Number of centerline points: {num_points}")
        log.info(f"Min HU value: {np.min(centerline_values):.2f}, Max HU value: {np.max(centerline_values):.2f}")
        log.info("-----------------------------------------------------------------------------------------\n")

        # Debug: Comparison of centerline HU values with image HU values
        log.info("\n--------------------------------- Debugging HU values ----------------------------------")
        number = 10
        start = centerline_indices[number]
        vtk_value = centerline_values[number]
        img_value = img[tuple(start)]
        log.info(f"Centerline point (physical coordinates): {centerline_points[number]}")
        log.info(f"Centerline point (index coordinates): {start}")
        log.info(f"HU value from centerline: {vtk_value:.2f}")
        log.info(f"HU value from image: {img_value}")
        log.info("----------------------------------------------------------------------------------------\n")

        # Dictionary of sample data
        sample = {
                  'image': img, 
                  'label': label, 
                  'centerline': centerline, 
                  'centerline_indices': centerline_indices,
                  'centerline_values': centerline_values,
                  'image_index': img_index, 
                  'spacing': spacing,
                  'origin': origin,
                  'direction': direction}

        return sample