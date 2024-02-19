import re
from os.path import split, join


class NNUnetUtils:

    @staticmethod
    def get_lbl_from_img_path(img_path: str):
        path, identifier = split(img_path)
        path, img_dir = split(path)

        lbl_dir = img_dir.replace("images", "labels")
        path = join(path, lbl_dir)
        base_pattern = r"([\w\-]+)_\d+(\.[\w\.]+)"
        base_match = re.match(base_pattern, identifier)
        lbl_identifier = f"{base_match.group(1)}{base_match.group(2)}"
        return join(path, lbl_identifier)


if __name__ == '__main__':
    print(NNUnetUtils.get_lbl_from_img_path( "/home/ai/projects/mm_data/nnunet_raw/Dataset005_mm/imagesTr/001_SA-1_0000.nii.gz"))
