import datetime
from utils import to_utc_datetime


def test_to_utc_datetime_parses_iso_and_epoch():
    dt1 = to_utc_datetime("1970-01-01T00:00:10Z")
    dt2 = to_utc_datetime(10)
    assert dt1 == datetime.datetime(1970, 1, 1, 0, 0, 10, tzinfo=datetime.timezone.utc)
    assert dt2 == dt1


def test_to_utc_datetime_with_timezone():
    dt = to_utc_datetime("1970-01-01 01:00:00", tz="Europe/Berlin")
    assert dt == datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
