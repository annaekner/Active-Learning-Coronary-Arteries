import SimpleITK as sitk

def spatial_resample_scan(image, desired_spacing):
    """
    Resample a scan to iso-tropic pixel spacing
    :param image: Original image with potentially anisotropic spacing
    :param desired_spacing: desired voxel spacing
    :return: resampled image
    """
    current_n_vox = image.GetWidth()
    current_spacing = image.GetSpacing()
    new_n_vox_in_slice: int = int(current_n_vox * current_spacing[0] / desired_spacing)

    # voxel size in the direction of the patient
    depth_spacing = current_spacing[2]
    n_vox_depth = image.GetDepth()
    new_n_vox_depth = int(n_vox_depth * depth_spacing / desired_spacing)

    new_volume_size = [new_n_vox_in_slice, new_n_vox_in_slice, new_n_vox_depth]

    # Create new image with desired properties
    new_image = sitk.Image(new_volume_size, image.GetPixelIDValue())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetSpacing([desired_spacing, desired_spacing, desired_spacing])
    new_image.SetDirection(image.GetDirection())

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    interpolator = sitk.sitkLinear
    # Create final resampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator)
    return resampled_image