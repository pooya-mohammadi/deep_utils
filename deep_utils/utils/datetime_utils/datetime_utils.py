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

    @staticmethod
    def parse_str_time(time_input, input: Literal['seconds'] = "seconds"):
        """
        :param time_input: 10
        :param input:
        :return:
        >>> DateTimeUtils.parse_str_time(12.5, input="seconds")
        '00:00:12.500'
        >>> DateTimeUtils.parse_str_time(12., input="seconds")
        '00:00:12.000'
        """


        if input == "seconds":
            milliseconds = int((time_input * 1e+3) % 1e+3)
            x = str(datetime.timedelta(hours=0, minutes=0, seconds=int(time_input),
                                   milliseconds=milliseconds))
            if x.startswith("0:"):
                x = "00:" + x[2:]
            if x.endswith("000"):
                x = x[:-3]
            if "." not in x:
                x = x + ".000"

            return str(x)