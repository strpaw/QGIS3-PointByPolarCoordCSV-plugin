# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PointByPolarCoordCSV
                                 A QGIS plugin
 Creates points defined by bearing and distance against point with known coordinates and magnetic variation
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2019-01-02
        copyright            : (C) 2019 by Paweł Strzelewicz
        email                : aviationgisapp@gmial.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load PointByPolarCoordCSV class from file PointByPolarCoordCSV.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .point_polar_coord_csv import PointByPolarCoordCSV
    return PointByPolarCoordCSV(iface)