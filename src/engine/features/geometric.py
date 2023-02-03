from .feature_sequence import FeatureSequence

# import pgmagick as magick
# from pgmagick import Image
# WHITE = magick.Color.scaleDoubleToQuantum(1)
# THRESHOLD = magick.Color.scaleDoubleToQuantum(.78)
WHITE = None


class GeoFeatures:
    """A geometrical features extractor.

    Based on the features described in U.V. Marti and H. Bunke, “Using a statistical language model to improve the
    performance of an HMM-based cursive handwriting recognition system”, International Journal of Pattern Recognition
    and Artificial Intelligence, vol. 15, no. 01, pp. 65–90, 2001, https://doi.org/10.1142/S0218001401000848.
    See numbers in parentheses

    The implementation is very slow and relies on the pgmagick lib, should not be used"""

    def __init__(self, file):
        self.image = Image(file)
        self.image.colorSpace(magick.ColorspaceType.GRAYColorspace)
        self.height = self.image.rows()
        self.width = self.image.columns()
        self.feature_sequence = False

    # public

    def extract_features(self):
        self.feature_sequence = FeatureSequence([self.feature_vector(x) for x in range(self.width)])

    # private

    def _grayscale_value_at(self, x, y):
        """Get pixel value"""
        return self.image.pixelColor(x, y).redQuantum()

    def _grayscale_white_border(self, x, y):
        """Get pixel value, outside the image is white"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return WHITE
        return self._grayscale_value_at(x, y)

    def feature_vector(self, x):
        return [
            self.histogram(x),
            self.upper_bound(x),
            self.lower_bound(x),
            self.upper_deviation(x),
            self.lower_deviation(x),
            self.between(x),
            self.black_white_transitions(x),
            self.gravity(x)
        ]

    # feature 1 (1): Count of black pixels divided by the height
    def histogram(self, x):
        return self.pixels_at(x, 0, self.height) / self.height

    # feature 2 (4): Highest black pixel position divided by the height
    def upper_bound(self, x):
        return self.upper(x) / self.height

    # feature 3 (5): Lowest black pixel position divided by the height
    def lower_bound(self, x):
        return self.lower(x) / self.height

    # feature 4 (6): Difference between the position of the highest black pixel for the next column and this column
    def upper_deviation(self, x):
        if x < self.width:
            return self.upper(x) - self.upper(x + 1)
        else:
            return 0.0

    # feature 5 (7): Same for the lowest black pixel
    def lower_deviation(self, x):
        if x < self.width:
            return self.lower(x) - self.lower(x + 1)
        else:
            return 0.0

    # feature 6 (9): Number of black pixels divided by the difference between the lower and the upper bound
    def between(self, x):
        high = self.upper(x)
        low = self.lower(x)
        if high < low:
            return self.pixels_at(x, high, (low - 1)) / (low - high)
        else:
            return 0.0

    # feature 7 (8): Number of vertical black → white transitions
    def black_white_transitions(self, x):
        flag = False
        transitions = 0
        for y in range(self.height):
            if not flag and is_black(self._grayscale_value_at(x, y)):
                transitions += 1
                flag = True
            elif flag and is_white(self._grayscale_value_at(x, y)):
                flag = False
        return transitions

    # feature 8 (9): Center of gravity divided by the height
    def gravity(self, x):
        gravity_value = 0.0
        for y in range(self.height):
            d = self.height / 2.0 - (y - 1)
            gravity_value += d * (1.0 - self._grayscale_value_at(x, y) / WHITE)
        return gravity_value / self.height

    # feature 9 (3): Unused
    def moment2(self, x):
        mom2 = 0.0
        for y in range(self.height):
            mom2 += ((y - 1) * (y - 1)) * (1.0 - self._grayscale_value_at(x, y) / WHITE)
        return mom2 / (self.height * self.height)

    # count the black pixels
    def pixels_at(self, x, upper_bound, lower_bound):
        pixels = 0
        if upper_bound <= lower_bound:
            for y in range(upper_bound, lower_bound):
                if is_black(self._grayscale_value_at(x, y)):
                    pixels += 1
        return pixels

    def upper(self, x):  # 0..height-1
        y = 1
        while y < self.height - 1 and is_white(self._grayscale_value_at(x, y)):
            y += 1
        return y

    def lower(self, x):  # height-1..0
        y = self.height - 1
        while y > 0 and is_white(self._grayscale_value_at(x, y)):
            y -= 1
        return y


def is_black(value):
    return value < THRESHOLD


def is_white(value):
    return value >= THRESHOLD
