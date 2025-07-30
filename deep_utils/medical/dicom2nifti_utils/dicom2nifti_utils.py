import os
from deep_utils.utils.str_utils.str_utils import StringUtils


class Dicom2Nifti:

    @staticmethod
    def make_command(nifti_dir, dicom_dir):
        command = f'dcm2niix -d 0 -z y -b y -ba n -f %f_%t_%p_%s_%d_%i_%e_%q_%z_%m_%a_%g -o "{nifti_dir}" "{dicom_dir}"'
        return command

    @staticmethod
    def dicom2nifti(nifti_path, dicom_path, verbose: bool=True):
        command = Dicom2Nifti.make_command(nifti_path, dicom_path)
        os.makedirs(nifti_path, exist_ok=True)
        output = os.system(command)
        if output != 0:
            StringUtils.print(f"[ERROR] Something wrong in converting: command: {command}", color="red")
        else:
            if verbose:
                StringUtils.print(f"[INFO] Successfully generated nifti {nifti_path} from dicom {dicom_path}", color="green")

