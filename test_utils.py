import unittest
import utils


class TestUtils(unittest.TestCase):

    def test_list_anding(self):
        bitmaps = [31,  # 1, 1, 1, 1, 1],
                   7,  # 0, 0, 1, 1, 1],
                   29]  # 1, 1, 1, 0, 1]]

        self.assertEqual(utils.list_anding(bitmaps), [3, 5])

        bitmaps = [31,  # 1, 1, 1, 1, 1],
                   31]  # 1, 1, 1, 1, 1]]

        self.assertEqual(utils.list_anding(bitmaps), [1, 2, 3, 4, 5])

        bitmaps = [31,  # 1, 1, 1, 1, 1],
                   7,  # 0, 0, 1, 1, 1],
                   13]  # 1, 1, 0, 1]]

        self.assertEqual(utils.list_anding(bitmaps), [3, 5])

        bitmaps = [31,  # 1, 1, 1, 1, 1],
                   7,  # 0, 0, 1, 1, 1],
                   28]  # 1, 1, 1, 0, 0]]

        self.assertEqual(utils.list_anding(bitmaps), [3])

    def test_list_oring(self):
        bitmaps = [23,  # 1, 0, 1, 1, 1],
                   7,  # 0, 0, 1, 1, 1],
                   21]  # 1, 0, 1, 0, 1]]

        self.assertEqual(utils.list_oring(bitmaps), [1, 3, 4, 5])

        bitmaps = [31,  # 1, 1, 1, 1, 1],
                   31]  # 1, 1, 1, 1, 1],

        self.assertEqual(utils.list_oring(bitmaps), [1, 2, 3, 4, 5])

        bitmaps = [0,  # 0, 0, 0],
                   0,  # 0, 0, 0],
                   ]

        self.assertEqual(utils.list_oring(bitmaps), [])

    def test_find_offset(self):
        number = 31
        offsets = list(utils.find_offsets(number))
        self.assertEqual(offsets, [0, 1, 2, 3, 4])

        number = 0
        offsets = list(utils.find_offsets(number))
        self.assertEqual(offsets, [])


if __name__ == "__main__":
    unittest.main()
