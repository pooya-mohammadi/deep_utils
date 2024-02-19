import re
from os.path import split, join


class NNUnetUtils:

    @staticmethod
    def get_lbl_from_img_path(img_path: str):
        path, identifier = split(img_path)
        path, img_dir = split(path)

        lbl_dir = img_dir.replace("images", "labels")
        path = join(path, lbl_dir)
        base_pattern = r"([\w\-\%]+)_\d+(\.[\w\.]+)"
        base_match = re.match(base_pattern, identifier)
        lbl_identifier = f"{base_match.group(1)}{base_match.group(2)}"
        return join(path, lbl_identifier)


if __name__ == '__main__':
    print(NNUnetUtils.get_lbl_from_img_path('/media/aici/11111bdb-a0c7-4342-9791-36af7eb70fc01/NNUNET_OUTPUT/nnunet_raw/Dataset008_ct_bern/imagesTr/TAVI_00203_20200702172417_Fl_CaSc_Th_LF_1.0_I70f_3_55%_18_Fl_CaSc_Th_LF_1.0_I70f_3_55%_TAVI_00203_0_Si_TAVI_00203_0000.nii.gz'))
