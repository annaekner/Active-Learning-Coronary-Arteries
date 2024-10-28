import vtk
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from vtkmodules.util.numpy_support import numpy_to_vtk

from tools import spatial_resample_scan
from data_loader import CoronaryArteryDataLoader

def load_sample(index, config, subset = 'train', label_type = 'full_coronary_tree'):
    """ Load scan from the dataset into the sample dictionary """

    # Create an instance of the data loader
    data_loader = CoronaryArteryDataLoader(config, subset, label_type)
    
    # Load a single scan
    sample = data_loader.__getitem__(index) # or simply: data_loader[0]

    return sample

def load_prediction(img_index, config):
    """ Load prediction (LAD segmentation) as a numpy array """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    data_predicted_dir = config.data_predicted.dataset_dir
    date = config.data_predicted.date

    # Load the prediction
    prediction_path = f"{base_dir}/{data_predicted_dir}/{date}/img{img_index}.nii.gz"

    # Convert from .nii.gz file to numpy array
    prediction_nii = sitk.ReadImage(prediction_path)
    prediction = sitk.GetArrayFromImage(prediction_nii)
    prediction = prediction.astype(np.bool)

    return prediction, prediction_nii

def save_resampled_img(img_index, config, subset = 'train'):
    """ Save resampled image (isotropic voxel spacing) as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    data_original_dir = config.data_loader.data_original_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_images_dir = config.data_processed.train_images_dir
    test_images_dir = config.data_processed.test_images_dir
    voxel_spacing = config.data_loader.voxel_spacing
    
    # Path to the original image
    img_path = f"{base_dir}/{data_original_dir}/all/{img_index}.img.nii.gz"

    # Resampe the image to isotropic voxel spacing
    img_nii = sitk.ReadImage(img_path)
    img_nii = spatial_resample_scan(img_nii, voxel_spacing) # Resample image to isotropic pixel spacing

    # Path for saving resampled image
    file_name = f"img{img_index}_0000.nii.gz"

    if subset == 'train':
         output_path = f"{base_dir}/{dataset_processed_dir}/{train_images_dir}/{file_name}"
        
    elif subset == 'test':
        output_path = f"{base_dir}/{dataset_processed_dir}/{test_images_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(img_nii, output_path)

def save_segmentation(segmentation, sample, config, subset = 'train', output_path = None):
    """ Save segmentation as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_labels_dir = config.data_processed.train_labels_dir
    test_labels_dir = config.data_processed.test_labels_dir

    # Convert segmentation array to SimpleITK image
    segmentation_nii = sitk.GetImageFromArray(segmentation.astype(int))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    segmentation_nii.SetOrigin(origin)
    segmentation_nii.SetSpacing(spacing)
    segmentation_nii.SetDirection(direction)

    # Path for saving segmentation
    file_name = f"img{sample['image_index']}.nii.gz"

    if not output_path:
        if subset == 'train':
            output_path = f"{base_dir}/{dataset_processed_dir}/{train_labels_dir}/{file_name}"
        
        elif subset == 'test':
            output_path = f"{base_dir}/{dataset_processed_dir}/{test_labels_dir}/{file_name}"

    # Write the segmentation image to file
    sitk.WriteImage(segmentation_nii, output_path)

def save_distance_map(distance_map, sample, config, subset = 'train', output_path = None):
    """ Save distance map as .nii.gz file """

    # Configuration settings
    base_dir = config.base_settings.base_dir
    dataset_processed_dir = config.data_processed.dataset_dir
    train_images_dir = config.data_processed.train_images_dir
    test_images_dir = config.data_processed.test_images_dir

    # Convert distance map array to SimpleITK image
    distance_map_nii = sitk.GetImageFromArray(distance_map.astype(float))
    
    # Retrieve original image metadata
    origin, spacing, direction = sample['origin'], sample['spacing'], sample['direction']
    distance_map_nii.SetOrigin(origin)
    distance_map_nii.SetSpacing(spacing)
    distance_map_nii.SetDirection(direction)

    # Path for saving distance map
    file_name = f"img{sample['image_index']}_0001.nii.gz"

    if not output_path:
        if subset == 'train':
            output_path = f"{base_dir}/{dataset_processed_dir}/{train_images_dir}/{file_name}"
        
        elif subset == 'test':
            output_path = f"{base_dir}/{dataset_processed_dir}/{test_images_dir}/{file_name}"

    # Write the distance map image to file
    sitk.WriteImage(distance_map_nii, output_path)

def save_prediction_centerline_to_vtk(prediction_centerline_indices, reference_nii, img_index, config):
    """ 
    Save the centerline from the LAD segmentation prediction to a VTK file.

    Args 
        prediction_centerline_indices: Indices of the centerline in the predicted LAD segmentation (numpy array of [z, y, x] index coordinates)
        reference_nii: Predicted LAD segmentation (SimpleITK image, .nii.gz) as reference to convert the indices to physical points
        output_filename: Path to the output VTK file
    """
    # Get configuration settings
    base_dir = f'{config.base_settings.base_dir}'
    centerline_prediction_dir = f'{config.centerline_predictions.dir}'
    date = f'{config.centerline_predictions.date}'
    file_name = f'img{img_index}_lad_centerline_from_prediction.vtk'
    output_filename = f'{base_dir}/{centerline_prediction_dir}/{date}/{file_name}'

    # Convert (z, y, x) to (x, y, z) for VTK compatibility
    prediction_centerline_indices = prediction_centerline_indices[:, [2, 1, 0]] 

    prediction_centerline_physical_points = np.array([reference_nii.TransformContinuousIndexToPhysicalPoint(point.tolist()) for point in prediction_centerline_indices])
    
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(numpy_to_vtk(prediction_centerline_physical_points))

    # Create a vtkPolyLine to represent the centerline
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(prediction_centerline_indices))
    for i in range(len(prediction_centerline_indices)):
        polyline.GetPointIds().SetId(i, i)

    # Create a vtkCellArray to store the lines in
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    # Create a vtkPolyData to hold the geometry and topology
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points_vtk)
    polydata.SetLines(cells)

    # Write the vtkPolyData to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(polydata)
    writer.Write()