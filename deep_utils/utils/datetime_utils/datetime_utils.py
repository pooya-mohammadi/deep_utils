from typing import Literal
import datetime

class DateTimeUtils:

    @staticmethod
    def parse_time_str(time_string, output: Literal['seconds']= "seconds"):
        """
        :param time_string: "00:01:13.232"
        :param output:
        :return:
        >>> DateTimeUtils.parse_time_str("00:01:13.232", output="seconds")
        73.232
        >>> DateTimeUtils.parse_time_str("00:02:15", output="seconds")
        135.0
        """
        start = time_string + ".000" if "." not in time_string else time_string

        x = datetime.datetime.strptime(start, "%H:%M:%S.%f").time()
        if output == "seconds":
            x = datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds=x.microsecond).total_seconds()
        return x
