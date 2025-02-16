from datetime import datetime, timezone, timedelta
import re

_datetime_regex = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"\.(?P<microsecond>\d+)(?P<tzinfo>[+-]\d{2}:\d{2})$"
)
_tz_regex = re.compile(r"^(?P<dir>\+|-)(?P<hour>\d{2}):(?P<minute>\d{2})")


def parse_tz(tz: str) -> timezone:
    data = _tz_regex.match(tz).groupdict()
    direction = 1 if data["dir"] == "+" else -1
    hour = direction * int(data["hour"])
    minute = direction * int(data["minute"])

    delta = timedelta(hours=hour, minutes=minute)
    return timezone(delta)


def parse_date(date: str) -> datetime:
    data = _datetime_regex.match(date).groupdict()
    data["tzinfo"] = parse_tz(data["tzinfo"])
    data["microsecond"] = data["microsecond"][:6]
    data.update(
        {
            k: int(data[k])
            for k in ["year", "month", "day", "hour", "minute", "second", "microsecond"]
        }
    )

    return datetime(**data)
