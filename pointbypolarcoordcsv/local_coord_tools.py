"""
/******************************************************************************************
circ_tools
Set of tools to create polygons that shapes are based on circle and its parts (circle sector,
circle segment etc.).

copyright  : (C) 2018 by Paweł Strzelewicz
email      : aviationgisapp@gmail.com

******************************************************************************************/
"""
import re
import math
import datetime


def get_tmp_name():
    """ Creates temporary name based on current time """
    c_time = datetime.datetime.now()  # Current time
    tmp_name = str(c_time).replace('-', '')  # Remove hyphens
    tmp_name = tmp_name.replace(':', '')  # Remove colons
    tmp_name = tmp_name.replace(' ', '_')
    tmp_name = tmp_name.replace('.', '_')
    tmp_name = tmp_name[2:] # Trim two first digits of year
    return tmp_name


# Special constants to use instead of False, to avoid issue where result of function might equal 0
# and result of function will be used in if statements etc.
VALID = 'VALID'
NOT_VALID = 'NOT_VALID'

""" Distance """
# Units of measure
UOM_M = 'M'  # meters
UOM_KM = 'KM'  # kilometers
UOM_NM = 'NM'  # nautical miles
UOM_FEET = 'FEET'  # feet
UOM_SM = 'SM'  # statue miles

# Pattern for distance regular expression
REGEX_DIST = re.compile(r'^\d+(\.\d+)?$')


def check_distance(d):
    """ Distance validation.
    :param d: string, distance to validate
    :return :bool, True if distance is valid,
             False if distance is not valid (e.g distance is less than 0)
    """
    if REGEX_DIST.match(d):
        return True
    else:
        return False


def check_distance2(d):
    """ Distance validation. Uses float() function to check if parameters is a number
    :param d: string, distance to validate
    :return is_valid: True if distance is valid,
                     constant NOT_VALID if distance is not valid (e.g distance is less than 0)
    """
    try:
        dist = float(d)
        if dist < 0:  # Check if it is less than 0
            dist = NOT_VALID
    except ValueError:
        dist = NOT_VALID
    return dist


# Conversion kilometers, nautical miles, feet, statue miles,  to meters


def km2m(km):
    """ Converts kilometers to meters
    :param km: float, value in kilometers
    :return: value in meters
    """
    return km * 1000


def nautical_mile2m(nm):
    """ Converts nautical miles to meters
    :param nm: float, value in nautical miles
    :return: value in meters
    """
    return nm * 1852


def feet2m(feet):
    """ Converts feet to meters
    :param feet: float, value in feet
    :return: value in meters
    """
    return feet * 0.3048


def statute_mile2m(sm):
    """ Converts statue miles to meters
    :param sm: float, value in statue miles
    :return: value in meters
    """
    return sm * 1609.344


# Conversion meters, nautical miles, feet, statue miles to km


def m2km(m):
    """ Converts meters to kilometers
    :param m: float, value in meters
    :return: value in kilometers
    """
    return m / 1000


def nautical_mile2km(nm):
    """ Converts nautical miles to kilometers
    :param nm: float, value in nautical miles
    :return: value in kilometers
    """
    return nm * 1.852


def feet2km(feet):
    """ Converts feet to kilometers
    :param feet: float, value in feet
    :return: value in kilometers
    """
    return (feet * 0.3048) / 1000


def statute_mile2km(sm):
    """ Converts statue miles to kilometers
    :param sm: float, value in statue miles
    :return: value in kilometers
    """
    return sm * 1.609344


# Conversion meters, kilometers, feet, statue miles to nautical miles


def m2nautical_mile(m):
    """ Converts meters to nautical miles
    :param m: float, value in meters
    :return: value in nautical miles
    """
    return m / 1852


def km2nautical_mile(km):
    """ Converts kilometers to nautical miles
    :param km: float, value in kilometers
    :return: value in nautical miles
    """
    return km / 1.852


def feet2nautical_mile(feet):
    """ Converts feet to nautical miles
    :param feet: float, value in feet
    :return: value in nautical miles
    """
    return (feet * 0.3048) / 1852


def statute_mile2nautical_mile(sm):
    """ Converts statue miles to nautical miles
    :param sm: float, value in statue miles
    :return: value in nautical miles
    """
    return (sm * 1.609344) / 1.852


# Conversion meters, kilometers, nautical miles, statue miles to feet


def m2feet(m):
    """ Converts meters to feet
    :param m: float, value in meters
    :return: value in feet
    """
    return m * (1 / 0.3048)


def km2feet(km):
    """ Converts kilometers to feet
    :param km: float, value in kilometers
    :return: value in feet
    """
    return km * (1000 / 0.3048)


def nautical_mile2feet(nm):
    """ Converts nautical miles to feet
    :param nm: float, value in nautical miles
    :return: value in feet
    """
    return nm * 1852 * (1 / 0.3048)


def statute_mile2feet(sm):
    """ Converts statue miles to feet
    :param sm: float, value in statue miles
    :return: value in feet
    """
    return sm * 5280


# Conversion meters, kilometers, nautical miles, feet to statue miles


def m2statute_mile(m):
    """ Converts meters to statue miles
    :param m: float, value in meters
    :return: value in statue miles
    """
    return m / 1609.344


def km2statute_mile(km):
    """ Converts kilometers to statue miles
    :param km: float, value in kilometers
    :return: value in statue miles
    """
    return km / 1.609344


def nautical_mile2statute_mile(nm):
    """ Converts nautical miles to statue miles
    :param nm: float, value in nautical miles
    :return: value in statue miles
    """
    return nm * ((1852 / 0.3048) / 5280)


def feet2statute_mile(feet):
    """ Converts feet to statue miles
    :param feet: float, value in feet
    :return: value in statue miles
    """
    return feet / 5280


def m2all(d):
    """ Converts m to km, NM, feet, SM
    :param d: float, distance in meters
    :return tuple: tuple of distance in km, feet, NM, SM
    """
    return m2km(d), m2nautical_mile(d), m2feet(d), m2statute_mile(d)


def km2all(d):
    """ Converts km to m, NM, feet, SM
    :param d: float, distance in kilometers
    :return tuple: tuple of distance in m, feet, NM, SM
    """
    return km2m(d), km2nautical_mile(d), km2feet(d), km2statute_mile(d)


def nautical_mile2all(d):
    """ ConvertsNM to m, km, feet, SM
    :param d: float, distance in nautical miles
    :return tuple: tuple of distance in m, km, feet,SM
    """
    return nautical_mile2m(d), nautical_mile2km(d), nautical_mile2feet(d), nautical_mile2statute_mile(d)


def feet2all(d):
    """ Converts feet to m, km, NM, SM
    :param d: float, distance in feet
    :return tuple: tuple of distance in m, km, NM, SM
    """
    return feet2m(d), feet2km(d), feet2nautical_mile(d), feet2statute_mile(d)


def statute_mile2all(d):
    """ Converts SM to m, km, NM
    :param d: float, distance in statue miles
    :return tuple: tuple of distance in m, km, NM, feet
    """
    return statute_mile2m(d), statute_mile2km(d), statute_mile2nautical_mile(d), statute_mile2feet(d)


# Conversion between two units

def to_meters(d, from_unit):
    """ Converts distance given in specified unit to distance in meters
    :param d: float, distance in unit specified by parameter from_unit
    :param from_unit: constant unit of measure, unit of measure parameter d_unit
    :return float, distance in unit specified by parameter to_unit
    """
    if from_unit == UOM_M:
        return d
    elif from_unit == UOM_KM:
        return d * 1000
    elif from_unit == UOM_NM:
        return d * 1852
    elif from_unit == UOM_FEET:
        return d * 0.3048
    elif from_unit == UOM_SM:
        return d * 1609.344
    else:
        return NOT_VALID


def from_meters(d, to_unit):
    """ Converts distance given in meters to distance in specified unit
    :param d: float, distance in meters
    :param to_unit: constant unit of measurement
    :return float, distance in unit specified by parameter to_unit
    """
    if to_unit == UOM_M:
        return d
    elif to_unit == UOM_KM:
        return d / 1000
    elif to_unit == UOM_NM:
        return d / 1852
    elif to_unit == UOM_FEET:
        return d / 0.3048
    elif to_unit == UOM_SM:
        return d / 1609.344
    else:
        return NOT_VALID


def convert_distance(d, from_unit, to_unit):
    """ Convert distance between various units
    :param d: float, distance in units specified by parameter from_unit
    :param from_unit: constant measure of units
    :param to_unit: constant measure of unit
    :return float, distance in units specified by parameter to_unit
    """
    if from_unit == to_unit:
        return d
    else:
        d_m = to_meters(d, from_unit)  # Convert to meters
        return from_meters(d_m, to_unit)  # Convert from meters


""" Calculations on ellipsoid """

# Parameters of WGS84 ellipsoid

WGS84_A = 6378137.0  # semi-major axis of the WGS84 ellipsoid in m
WGS84_B = 6356752.314245  # semi-minor axis of the WGS84 ellipsoid in m
WGS84_F = 1 / 298.257223563  # flattening of the WGS84 ellipsoid


def vincenty_direct_solution(begin_lat, begin_lon, begin_azimuth, distance, a, b, f):
    """ Computes the latitude and longitude of the second point based on latitude, longitude,
    of the first point and distance and azimuth from first point to second point.
    Uses the algorithm by Thaddeus Vincenty for direct geodetic problem.
    For more information refer to: http://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf
    :param begin_lon: float, longitude of the first point; decimal degrees
    :param begin_lat: float, latitude of the first point; decimal degrees
    :param begin_azimuth: float, azimuth from first point to second point; decimal degrees
    :param distance: float, distance from first point to second point; meters
    :param a: float, semi-major axis of ellipsoid; meters
    :param b: float, semi-minor axis of ellipsoid; meters
    :param f: float, flattening of ellipsoid
    :return lat2_dd, lon2_dd: float, float latitude and longitude of the second point, decimal degrees
    """
    # Convert latitude, longitude, azimuth of the initial point to radians
    lat1 = math.radians(begin_lat)
    lon1 = math.radians(begin_lon)
    alfa1 = math.radians(begin_azimuth)
    sin_alfa1 = math.sin(alfa1)
    cos_alfa1 = math.cos(alfa1)

    # U1 - reduced latitude
    tan_u1 = (1 - f) * math.tan(lat1)
    cos_u1 = 1 / math.sqrt(1 + tan_u1 * tan_u1)
    sin_u1 = tan_u1 * cos_u1

    # sigma1 - angular distance on the sphere from the equator to initial point
    sigma1 = math.atan2(tan_u1, math.cos(alfa1))

    # sin_alfa - azimuth of the geodesic at the equator
    sin_alfa = cos_u1 * sin_alfa1
    cos_sq_alfa = 1 - sin_alfa * sin_alfa
    u_sq = cos_sq_alfa * (a * a - b * b) / (b * b)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    sigma = distance / (b * A)
    sigmap = 1
    sin_sigma, cos_sigma, cos2sigma_m = None, None, None

    while math.fabs(sigma - sigmap) > 1e-12:
        cos2sigma_m = math.cos(2 * sigma1 + sigma)
        sin_sigma = math.sin(sigma)
        cos_sigma = math.cos(sigma)
        d_sigma = B * sin_sigma * (cos2sigma_m + B / 4 * (
                    cos_sigma * (-1 + 2 * cos2sigma_m * cos2sigma_m) - B / 6 * cos2sigma_m * (
                        -3 + 4 * sin_sigma * sin_sigma) * (-3 + 4 * cos2sigma_m * cos2sigma_m)))
        sigmap = sigma
        sigma = distance / (b * A) + d_sigma

    var_aux = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alfa1  # Auxiliary variable

    # Latitude of the end point in radians
    lat2 = math.atan2(sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alfa1,
                      (1 - f) * math.sqrt(sin_alfa * sin_alfa + var_aux * var_aux))

    lamb = math.atan2(sin_sigma * sin_alfa1, cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alfa1)
    C = f / 16 * cos_sq_alfa * (4 + f * (4 - 3 * cos_sq_alfa))
    L = lamb - (1 - C) * f * sin_alfa * (
                sigma + C * sin_sigma * (cos2sigma_m + C * cos_sigma * (-1 + 2 * cos2sigma_m * cos2sigma_m)))
    # Longitude of the second point in radians
    lon2 = (lon1 + L + 3 * math.pi) % (2 * math.pi) - math.pi

    # Convert to decimal degrees
    lat2_dd = math.degrees(lat2)
    lon2_dd = math.degrees(lon2)

    return lat2_dd, lon2_dd


def vincenty_reverse_solution(lat1_dd, lon1_dd, lat2_dd, lon2_dd, a, b, f):
    """ Computes distance and bearing between two points
    :param lat1_dd: float, latitude in decimal degrees
    :param lon1_dd: float, longitude in decimal degrees
    :param lat2_dd: float, latitude in decimal degrees
    :param lon2_dd: float, longitude in decimal degrees
    :param a: float, semi-major axis of ellipsoid; meters
    :param b: float, semi-minor axis of ellipsoid; meters
    :param f: float, flattening of ellipsoid
    :return: alfa1: float: bearing from point 1 to point 2
             distance: distance between point 1 and 2
    """
    lat1 = math.radians(lat1_dd)
    lon1 = math.radians(lon1_dd)
    lat2 = math.radians(lat2_dd)
    lon2 = math.radians(lon2_dd)

    L = lon2 - lon1

    tan_u1 = (1 - f) * math.tan(lat1)
    cos_u1 = 1 / math.sqrt((1 + tan_u1 * tan_u1))
    sin_u1 = tan_u1 * cos_u1
    tan_u2 = (1 - f) * math.tan(lat2)
    cos_u2 = 1 / math.sqrt((1 + tan_u2 * tan_u2))
    sin_u2 = tan_u2 * cos_u2

    lamb = L
    lamb_p = L
    iterations = 0

    cos_sq_alfa, sin_sigma, cos2sigma__m, sin_lamb, cos_lamb, cos_sigma = None, None, None, None, None, None
    sigma = None

    while math.fabs(lamb - lamb_p) > 1e-12 or iterations < 1000:
        sin_lamb = math.sin(lamb)
        cos_lamb = math.cos(lamb)
        sin_sq_sigma = (cos_u2 * sin_lamb) * (cos_u2 * sin_lamb) + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lamb) * (
                    cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lamb)
        sin_sigma = math.sqrt(sin_sq_sigma)
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lamb
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alfa = cos_u1 * cos_u2 * sin_lamb / sin_sigma
        cos_sq_alfa = 1 - sin_alfa * sin_alfa
        cos2sigma__m = cos_lamb - 2 * sin_u1 * sin_u2 / cos_sq_alfa
        C = f / 16 * cos_sq_alfa * (4 + f * (4 - 3 * cos_sq_alfa))
        lamb_p = lamb
        lamb = L + (1 - C) * f * sin_alfa * (
                    sigma + C * sin_sigma * (cos2sigma__m + C * cos_sigma * (-1 + 2 * cos2sigma__m * cos2sigma__m)))
        iterations += 1

    u_sq = cos_sq_alfa * (a * a - b * b) / (b * b)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = B * sin_sigma * (cos2sigma__m + B / 4 * (
                cos_sigma * (-1 + 2 * cos2sigma__m * cos2sigma__m) - B / 6 * cos2sigma__m * (
                    -3 + 4 * sin_sigma * sin_sigma) * (-3 + 4 * cos2sigma__m * cos2sigma__m)))

    distance = b * A * (sigma - delta_sigma)
    alfa1 = math.atan2(cos_u2 * sin_lamb, cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lamb)
    alfa1 = (alfa1 + 2 * math.pi) % (2 * math.pi)
    alfa1 = math.degrees(alfa1)

    return alfa1, distance


""" Magnetic variation, bearing """


class Angle:
    def __init__(self, src_value):
        self.src_value = src_value
        self._is_valid = None
        self._dd_value = None
        self._err_msg = ''

    @staticmethod
    def check_angle_range(angle, min_value, max_value):
        """ Checks if angle is with closed interval <min_value, max_value>
        :param angle: float, angle value to check
        :param min_value: float, minimum value
        :param max_value: float, maximum value
        :return: tuple (bool, float) if angle is within the range
                 tuple (bool, None) if angle is out of range
        """
        if min_value <= angle <= max_value:
            return True, angle
        else:
            return False, None

    @staticmethod
    def normalize_src_input(src_input):
        """ Normalizes source (input) angle for further processing
        :param src_input: str, input angle string to normalize
        :return: norm_angle: str, normalized angle string

        """
        norm_input = str(src_input)
        norm_input = norm_input.replace(',', '.')
        norm_input = norm_input.upper()
        return norm_input

    @staticmethod
    def check_angle_dd(angle):
        """ Checks if angle is in DD format.
        :param angle: float, str: angle to check
        :return: float, vale of angle if angle is integer of float, const NOT_VALID otherwise
        """
        try:
            a = float(angle)
            return a
        except ValueError:
            return None

    @property
    def is_valid(self):
        return self._is_valid

    @is_valid.setter
    def is_valid(self, value):
        self._is_valid = value

    @property
    def dd_value(self):
        return self._dd_value

    @dd_value.setter
    def dd_value(self, value):
        self._dd_value = value

    @property
    def err_msg(self):
        return self._err_msg

    @err_msg.setter
    def err_msg(self, value):
        self._err_msg = value


""" Magnetic variation """
# Regular expression for magnetic variation pattern
# Magnetic variation, variation letter prefix or suffix decimal degrees e. g. E3.55, 0.77W
REGEX_MAG_VAR_VLDD = re.compile(r'^[WE]\d+\.\d+$|^[WE]\d+$')
REGEX_MAG_VAR_DDVL = re.compile(r'^\d+\.\d+[WE]$|^\d+[WE]$')


class MagVar(Angle):
    def __init__(self, mag_var_src):
        Angle.__init__(self, mag_var_src)
        self.parse_mag_var()

    def check_magvar_vletter_dd(self, mag_var):
        """ Check if magnetic variation is in decimal degrees with variation letter suffix or prefix format.
        e. g.: E3.55, 0.77W
        :return: float - magnetic variation in decimal degrees, or bool - False if input outside the range
        """
        if REGEX_MAG_VAR_VLDD.match(mag_var):
            h = mag_var[0]
            mv = self.check_angle_dd(mag_var[1:])
            if mv != NOT_VALID:
                if h == 'W':
                    mv = -mv
                return mv
            else:
                return None
        elif REGEX_MAG_VAR_DDVL.match(mag_var):
            h = mag_var[-1]
            mv = self.check_angle_dd(mag_var[:-1])
            if mv != NOT_VALID:
                if h == 'W':
                    mv = -mv
                return mv
            else:
                return None
        else:
            return None

    def parse_mag_var(self):
        """ Parse source value to convert it into decimal degrees value"""
        if self.src_value == '':  # If no value given - by default magnetic variation is 0.0
            self.dd_value = 0.0
            self.is_valid = True
            return
        else:
            norm_src = self.normalize_src_input(self.src_value)
            mv_dd = self.check_angle_dd(norm_src)  # Check if magnetic variation is in DD format
            if mv_dd is None:
                mv_dd = self.check_magvar_vletter_dd(norm_src) # Check if it is in HDD or DDH format
                if mv_dd is None:
                    self.is_valid = False
                    self.err_msg = 'Magnetic variation error!\n'

            if mv_dd is not None:  # Managed to get DD format of magnetic variation - check if it is within range
                self.is_valid, self.dd_value = self.check_angle_range(mv_dd, -120, 120)

            if self.is_valid is False:
                self.err_msg = 'Magnetic variation error!\n'


class Bearing(Angle):
    def __init__(self, brng_src):
        Angle.__init__(self, brng_src)
        self.dd_tbrng = None
        self.parse_brng()

    def parse_brng(self):
        """ Parse source value to convert it into decimal degrees value"""
        if self.src_value == '':  # No value
            self.is_valid = False
            self.err_msg = 'Enter bearing!\n'
        else:
            norm_src = self.normalize_src_input(self.src_value)
            brng = self.check_angle_dd(norm_src)  # Check if bearing is given in decimal degrees format
            if brng is None:
                self.is_valid = False
                self.err_msg = 'Bearing error!\n'

            if brng is not None:  # Managed to get DD format of bearing - check if it is within range
                self.is_valid, self.dd_value = self.check_angle_range(brng, 0, 360)

            if self.is_valid is False:
                self.err_msg = 'Bearing error!\n'

    def calc_tbrng(self, dd_mag_var):
        """ Calculates true bearing.
        :param: dd_mag_var: float, magnetic variation value
        """
        if dd_mag_var == 0:
            self.dd_tbrng = self.dd_value
        else:
            self.dd_tbrng = self.dd_value + dd_mag_var
            if self.dd_tbrng > 360:
                self.dd_tbrng -= 360
            elif self.dd_tbrng < 360:
                self.dd_tbrng += 360


""" latitude, longitude """

C_LAT = 'C_LAT'
C_LON = 'C_LON'

H_LAT = ['N', 'S']
H_LON = ['E', 'W']
H_NEGATIVE = ['-', 'S', 'W']
H_ALL = ['-', '+', 'N', 'S', 'E', 'W']

S_SPACE = ' '
S_HYPHEN = '-'
S_DEG_WORD = 'DEG'
S_DEG_LETTER = 'D'
S_MIN_WORD = 'MIN'
S_MIN_LETTER = 'M'
S_SEC_WORD = 'SEC'
S_ALL = [S_SPACE, S_HYPHEN, S_DEG_WORD, S_DEG_LETTER, S_MIN_WORD, S_MIN_LETTER, S_SEC_WORD]


# Degrees, minutes, seconds compacted
F_HDMS_COMP = 'F_HDMS_COMP'  # Hemisphere prefix DMS compacted
F_DMSH_COMP = 'F_DMSH_COMP'  # Hemisphere suffix DMS compacted
F_HDM_COMP = 'F_HDM_COMP'  # Hemisphere prefix DMS compacted
F_DMH_COMP = 'F_DMH_COMP'  # Hemisphere suffix DMS compacted

# Regular expression patterns for latitude and longitude
coord_regex = {F_HDMS_COMP: re.compile(r'''(?P<hem>^[NSEW])
                                           (?P<deg>\d{2,3})  # Degrees
                                           (?P<min>\d{2})  # Minutes
                                           (?P<sec>\d{2}(\.\d+)?$)  # Seconds 
                                        ''', re.VERBOSE),
               F_DMSH_COMP: re.compile(r'''(?P<deg>^\d{2,3})  # Degrees
                                           (?P<min>\d{2})  # Minutes
                                           (?P<sec>\d{2}(\.\d+)?)  # Seconds
                                           (?P<hem>[NSEW]$)   
                                        ''', re.VERBOSE),
               F_HDM_COMP: re.compile(r'''(?P<hem>^[NSEW])
                                          (?P<deg>\d{2,3})  # Degrees
                                          (?P<min>\d{2}(\.\d+)?$)  # Minutes
                                        ''', re.VERBOSE),
               F_DMH_COMP: re.compile(r'''(?P<deg>^\d{2,3})  # Degrees
                                          (?P<min>\d{2}(\.\d+)?)  # Minutes
                                          (?P<hem>[NSEW]$)   
                                       ''', re.VERBOSE)}


class CoordinatesPair:
    def __init__(self, src_lat, src_lon):
        self.src_lat = src_lat
        self.src_lon = src_lon
        self._is_valid = None
        self._err_msg = ''
        self._dd_lat = None
        self._dd_lon = None
        self.parse_src_coordinates()

    @staticmethod
    def normalize_src_input(src_input):
        """ Normalizes source (input) angle for further processing
        :param src_input: str, input angle string to normalize
        :return: norm_angle: str, normalized angle string

        """
        norm_input = str(src_input)
        norm_input = norm_input.strip()  # Trim leading and trailing space
        norm_input = norm_input.upper()  # Make all letters capitals
        norm_input = norm_input.replace(',', '.')  # Make sure that decimal separator is dot not comma
        return norm_input

    @staticmethod
    def parse_regex(regex_patterns, dms, c_type):
        """ Converts latitude or longitude in DMSH format into DD format.
        :param regex_patterns: dictionary of regex object, patterns of DMS formats
        :param dms: str, input coordinate
        :param c_type: const, type of coordinate, C_LAT or C_LON
        :return: dd:, float if DMS is valid format, None otherwise
        """

        dd = None
        for pattern in regex_patterns:  # Check if input matches any pattern
            if regex_patterns.get(pattern).match(dms):
                if pattern in [F_DMSH_COMP, F_HDMS_COMP]:
                    # If input matches to pattern get hemisphere, degrees, minutes and seconds values
                    groups = regex_patterns.get(pattern).search(dms)
                    h = groups.group('hem')
                    d = float(groups.group('deg'))
                    m = float(groups.group('min'))
                    s = float(groups.group('sec'))

                    if (h in H_LAT and c_type == C_LAT) or (h in H_LON and c_type == C_LON):

                        if h in ['N', 'S']:
                            if d > 90:  # Latitude is in range <-90, 90>
                                dd = None
                            elif d == 90 and (m > 0 or s > 0):
                                dd = None
                            else:
                                if m >= 60 or s >= 60:
                                    dd = None
                                else:
                                    dd = d + m / 60 + s / 3600
                                    if h == 'S':
                                        dd = -dd

                        elif h in ['E', 'W']:
                            if d > 180:  # Longitude is in range <-180, 180>
                                dd = None
                            elif d == 180 and (m > 0 or s > 0):
                                dd = None
                            else:
                                if m >= 60 or s >= 60:
                                    dd = None
                                else:
                                    dd = d + m / 60 + s / 3600
                                    if h == 'W':
                                        dd = -dd

        return dd

    @staticmethod
    def parse_coordinate(coord_norm, c_type):

        dd = None

        # First, check if input is in DD format
        try:
            dd = float(coord_norm)
        except ValueError:
            # Assume that coordinate is in DMS, DMSH, DM, DMH, HDD, DDH
            # Check first and last character
            h = coord_norm[0]
            if h in H_ALL:  # DMS, DM signed or HDMS, HDM, HDD,
                coord_norm = coord_norm[1:]
            else:  # Check last character
                h = coord_norm[-1]
                if h in H_ALL:
                    coord_norm = coord_norm[:-1]
                else:
                    h = coord_norm[0]
                    if h in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        if c_type == C_LAT:
                            h = 'N'
                        elif c_type == C_LON:
                            h = 'E'

            # Check if hemisphere letter matches coordinate type (c_type)
            if (h in H_LAT and c_type == C_LAT) or (h in H_LON and c_type == C_LON):
                # Trim spaces again
                coord_norm = coord_norm.strip()
                # Replace separators (delimiters) with blank (space)
                for sep in S_ALL:
                    coord_norm = re.sub(sep, ' ', coord_norm)
                # Replace multiple spaces into single spaces
                coord_norm = re.sub('\s+', ' ', coord_norm)
                c_parts = coord_norm.split(' ')
                if len(c_parts) == 3:  # Assume format DMS separated

                    try:
                        d = int(c_parts[0])
                        if d < 0:
                            return None
                    except ValueError:
                        return None

                    try:
                        m = int(c_parts[1])
                        if m < 0 or m >= 60:
                            return None
                    except ValueError:
                        return None

                    try:
                        s = float(c_parts[2])
                        if s < 0 or s >= 60:
                            return None
                    except ValueError:
                        return None

                    try:
                        dd = float(d) + float(m) / 60 + s / 3600
                        if h in H_NEGATIVE:
                            dd = - dd
                    except ValueError:
                        return None

                elif len(c_parts) == 2:  # Assume format DM separated
                    try:
                        d = int(c_parts[0])
                        if d < 0:
                            return None
                    except ValueError:
                        return None

                    try:
                        m = float(c_parts[1])
                        if m < 0 or m >= 60:
                            return None
                    except ValueError:
                        return None

                    try:
                        dd = float(d) + m / 60
                        if h in H_NEGATIVE:
                            dd = - dd
                    except ValueError:
                        return None

                elif len(c_parts) == 1:  # Assume format DMS, DM compacted or DD
                    try:
                        dd = float(c_parts[0])
                        if h in H_NEGATIVE:
                            dd = -dd
                    except ValueError:
                        return None
            else:
                return None

        # If we get dd - check is is withing range
        if dd is not None:
            if c_type == C_LAT:
                if -90 <= dd <= dd:
                    return dd
                else:
                    return None
            elif c_type == C_LON:
                if -180 <= dd <= 180:
                    return dd
                else:
                    return None

    def parse_src_coordinates(self):
        """ Parse latitude and longitude source values """
        if self.src_lat == '':  # Blank input
            self._err_msg += 'Enter latitude value!\n'
        else:
            norm_lat = self.normalize_src_input(self.src_lat)
            self.dd_lat = self.parse_regex(coord_regex, norm_lat, C_LAT)
            if self.dd_lat is None:
                self.dd_lat = self.parse_coordinate(norm_lat, C_LAT)
                if self.dd_lat is None:
                    self.err_msg += 'Latitude value wrong value!\n'

        if self.src_lon == '':  # Blank input
            self.err_msg += 'Enter longitude value!\n'
        else:
            norm_lon = self.normalize_src_input(self.src_lon)
            self.dd_lon = self.parse_regex(coord_regex, norm_lon, C_LON)
            if self.dd_lon is None:
                self.dd_lon = self.parse_coordinate(norm_lon, C_LON)
                if self.dd_lon is None:
                    self.err_msg += 'Longitude value wrong value!\n'

        if self.dd_lat is not None and self.dd_lon is not None:
            self.is_valid = True
        else:
            self.is_valid = False

    @property
    def is_valid(self):
        return self._is_valid

    @is_valid.setter
    def is_valid(self, result):
        self._is_valid = result

    @property
    def err_msg(self):
        return self._err_msg

    @err_msg.setter
    def err_msg(self, msg):
        self._err_msg = msg

    @property
    def dd_lat(self):
        return self._dd_lat

    @dd_lat.setter
    def dd_lat(self, dd):
        self._dd_lat = dd

    @property
    def dd_lon(self):
        return self._dd_lon

    @dd_lon.setter
    def dd_lon(self, dd):
        self._dd_lon = dd


class BasePoint(CoordinatesPair):
    def __init__(self, src_lat, src_lon, src_mag_var, origin_id=''):
        CoordinatesPair.__init__(self, src_lat, src_lon)
        self.mag_var = MagVar(src_mag_var)
        self.origin_id = origin_id
        self.check_init_data()

    def check_init_data(self):
        if self.mag_var.is_valid is False:
            self.is_valid = False
            self.err_msg += self.mag_var.err_msg


class PolarCoordPoint:
    def __init__(self, pole: BasePoint, angular_coord: Bearing, radial_coord_m):
        self.pole = pole
        self.angular_coord = angular_coord
        self.radial_coord_m = radial_coord_m
        self._ep_lon_dd = None
        self._ep_lat_dd = None
        self.calc_ep()

    def calc_ep(self):
        self.angular_coord.calc_tbrng(self.pole.mag_var.dd_value)
        self.ep_lat_dd, self.ep_lon_dd = vincenty_direct_solution(self.pole.dd_lat,
                                                                  self.pole.dd_lon,
                                                                  self.angular_coord.dd_tbrng,
                                                                  self.radial_coord_m,
                                                                  WGS84_A, WGS84_B, WGS84_F)

    @property
    def ep_lat_dd(self):
        return self._ep_lat_dd

    @ep_lat_dd.setter
    def ep_lat_dd(self, value):
        self._ep_lat_dd = value

    @property
    def ep_lon_dd(self):
        return self._ep_lon_dd

    @ep_lon_dd.setter
    def ep_lon_dd(self, value):
        self._ep_lon_dd = value


def check_azm_dist(azm, dist):
    is_valid = True
    err_msg = ''
    a = Bearing(azm)

    if a.is_valid is False:
        is_valid = False
        err_msg += '*Azimuth value error*'
    if check_distance2(dist) == NOT_VALID:
        is_valid = False
        err_msg += '*Distance value error*'
    return is_valid, err_msg
