class FaNLPUtils:
    @staticmethod
    def clean_str(input_str: str) -> str:
        return input_str.replace("\u200c", " ").replace('\xa0', ' ').replace('\xad', '').strip()
