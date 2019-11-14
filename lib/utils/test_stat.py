import unittest

import sys
from contextlib import contextmanager
from io import StringIO

from lib.utils.stat import Stat


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestStat(unittest.TestCase):

    def test_print_stat(self):
        stat = Stat()
        stat.add("acc", 0.83)
        stat.add("acc", 0.73)
        stat.add("acc", 0.63)
        with captured_output() as (out, err):
            stat.print_stat()

        output = out.getvalue().strip()
        self.assertEqual("acc - count 3 avg 0.73 std 0.082", output)

    def test_multi_print_stat(self):
        stat = Stat()
        stat.add("acc", 0.83)
        stat.add("acc", 0.73)
        stat.add("acc", 0.63)
        stat.add("recall", 0.73)
        stat.add("recall", 0.63)
        with captured_output() as (out, err):
            stat.print_stat()

        output = out.getvalue().strip()
        lines = output.splitlines()
        self.assertEqual("acc - count 3 avg 0.73 std 0.082", lines[0])
        self.assertEqual("recall - count 2 avg 0.68 std 0.05", lines[1])

    def test_get_count(self):
        stat = Stat()
        stat.add("acc", 0.83)
        stat.add("acc", 0.73)
        stat.add("acc", 0.63)
        stat.add("recall", 0.73)
        stat.add("recall", 0.63)
        results = stat.get_count(["acc", "recall"])
        self.assertEqual(3, results["acc"])
        self.assertEqual(2, results["recall"])

    def test_get_avg(self):
        stat = Stat()
        stat.add("acc", 0.83)
        stat.add("acc", 0.73)
        stat.add("acc", 0.63)
        stat.add("recall", 0.73)
        stat.add("recall", 0.63)
        results = stat.get_avg(["acc", "recall"])
        self.assertEqual(0.73, results["acc"])
        self.assertEqual((0.63 + 0.73) / 2, results["recall"])

    def test_get_avg(self):
        stat = Stat()
        stat.add("acc", 0.83)
        stat.add("acc", 0.73)
        stat.add("acc", 0.63)
        stat.add("recall", 0.73)
        stat.add("recall", 0.63)
        results = stat.get_avg(["acc", "recall"])
        self.assertEqual(0.73, results["acc"])
        self.assertEqual((0.63 + 0.73) / 2, results["recall"])


if __name__ == "main":
    unittest.main()
