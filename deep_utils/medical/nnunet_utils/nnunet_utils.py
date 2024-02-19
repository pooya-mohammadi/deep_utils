import re
from os.path import split, join


class NNUnetUtils:

    @staticmethod
    def get_lbl_from_img_path(img_path: str):
        path, identifier = split(img_path)
        path, img_dir = split(path)

        lbl_dir = img_dir.replace("images", "labels")
        path = join(path, lbl_dir)
        base_pattern = r"([\w\-\%\,]+)_\d+(\.[\w\.]+)"
        base_match = re.match(base_pattern, identifier)
        lbl_identifier = f"{base_match.group(1)}{base_match.group(2)}"
        return join(path, lbl_identifier)


if __name__ == '__main__':
    sample = "/media/aici/11111bdb-a0c7-4342-9791-36af7eb70fc01/NNUNET_OUTPUT/nnunet_raw/Dataset008_ct_bern/imagesTr/TAVI_00556_20230517163114_CaSc_FL_1,00_Br40_Q2_30%_Th_4_CaSc_FL_1,00_Br40_Q2_30%_Th_TAVI_00556_0_Si_TAVI_00556_0000.nii.gz"
    print(NNUnetUtils.get_lbl_from_img_path(sample))
