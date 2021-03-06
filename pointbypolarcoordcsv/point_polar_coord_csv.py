# -*- coding: utf-8 -*-
"""
/***************************************************************************
 PointByPolarCoordCSV
                                 A QGIS plugin
 Creates points defined by bearing and distance against point with known coordinates and magnetic variation
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2019-01-02
        git sha              : $Format:%H$
        copyright            : (C) 2019 by Paweł Strzelewicz
        email                : aviationgisapp@gmial.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QVariant
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget,  QFileDialog
from qgis.core import *

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .point_polar_coord_csv_dialog import PointByPolarCoordCSVDialog
import os.path
import csv

from . import local_coord_tools as lct

w = QWidget()


class PointByPolarCoordCSV:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.ref_point = None
        self.mlyr_name = ''
        self.input_file = ''
        self.output_file = ''
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'PointByPolarCoordCSV_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = PointByPolarCoordCSVDialog()
        self.dlg.pushButtonCalc.clicked.connect(self.calc_csv_points)
        self.dlg.pushButtonSelectInputCSV.clicked.connect(self.select_input_file)
        self.dlg.pushButtonSelectOutputCSV.clicked.connect(self.select_output_file)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&PointByPolarCoordCSV')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'PointByPolarCoordCSV')
        self.toolbar.setObjectName(u'PointByPolarCoordCSV')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('PointByPolarCoordCSV', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/point_polar_coord_csv/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'PointByPolarCoordCSV'),
            callback=self.run,
            parent=self.iface.mainWindow())


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&PointByPolarCoordCSV'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    @staticmethod
    def create_mem_lyr(lyr_name):
        """ Create temporary 'memory' layer to store results.
        :param lyr_name: string, layer name
        """
        mlyr = QgsVectorLayer('Point?crs=epsg:4326', lyr_name, 'memory')
        mprov = mlyr.dataProvider()
        mlyr.startEditing()
        mprov.addAttributes([QgsField("R_ID", QVariant.String),  # Reference point ID
                             QgsField("R_LATSRC", QVariant.String),   # Reference point latitude (source)
                             QgsField("R_LONSRC", QVariant.String),  # Reference point longitude (source)
                             QgsField("CP_NAME", QVariant.String),  # Calculated point name
                             QgsField("CP_LATDD", QVariant.String),  # Calculated point latitude (decimal degrees)
                             QgsField("CP_LONDD", QVariant.String),  # Calculated point longitude (decimal degrees)
                             QgsField("CP_DEF", QVariant.String)])  # Calculated point in polar coordinates 'definition'
        mlyr.commitChanges()
        QgsProject.instance().addMapLayer(mlyr)

    def check_ref_point_input(self):
        """ Check if input data for reference point is correct
        :return check_result: bool, True if input data is valid, False otherwise
        :return err_msg: bool, True if input data is valid, False otherwise
        """
        check_result = True
        err_msg = ''
        # Get input data from Qt LineEdit
        ref_point_id = self.dlg.lineEditRefId.text()
        src_ref_lat = self.dlg.lineEditRefLat.text()
        src_ref_lon = self.dlg.lineEditRefLon.text()
        src_ref_mag_var = self.dlg.lineEditRefMagVar.text()

        self.ref_point = lct.BasePoint(src_ref_lat, src_ref_lon, src_ref_mag_var, ref_point_id)

        if self.ref_point.is_valid is False:
            check_result = False
            err_msg += self.ref_point.err_msg

        return check_result, err_msg

    def get_csv_points_uom(self):
        """ Returns radius unit of measure """
        if self.dlg.comboBoxCSVDistUOM.currentIndex() == 0:  # m
            return lct.UOM_M
        elif self.dlg.comboBoxCSVDistUOM.currentIndex() == 1:  # km
            return lct.UOM_KM
        elif self.dlg.comboBoxCSVDistUOM.currentIndex() == 2:  # NM
            return lct.UOM_NM
        elif self.dlg.comboBoxCSVDistUOM.currentIndex() == 3:  # feet
            return lct.UOM_FEET
        elif self.dlg.comboBoxCSVDistUOM.currentIndex() == 4:  # SM
            return lct.UOM_SM

    def select_input_file(self):
        """ Select input csv file with data:
        ID of the point, azimuth, distance from reference point to the current point """
        self.input_file = QFileDialog.getOpenFileName(self.dlg, "Select input file ", "", '*.csv')[0]
        self.dlg.lineEditInputCSV.setText(self.input_file)

    def select_output_file(self):
        """ Select output csv file """
        self.output_file = QFileDialog.getSaveFileName(self.dlg, "Select output file ", "", '*.csv')[0]
        self.dlg.lineEditOutputCSV.setText(self.output_file)

    def check_csv_points_input(self):
        # Check reference (origin) point input and assign results to variables
        check_result, err_msg = self.check_ref_point_input()
        self.input_file = self.dlg.lineEditInputCSV.text()
        self.output_file = self.dlg.lineEditOutputCSV.text()

        if self.input_file == '':
            err_msg += 'Choose input file!\n'
            check_result = False

        if self.output_file == '':
            err_msg += 'Choose output file!\n'
            check_result = False

        if not check_result:
            QMessageBox.critical(w, "Message", err_msg)
        return check_result

    def calc_csv_points(self):
        if self.check_csv_points_input():
            true_mag = 'MAG'
            if self.ref_point.mag_var.src_value == '':
                true_mag = 'TRUE'

            layers = QgsProject.instance().layerTreeRoot().children()
            layers_list = []  # List of layers in current (opened) QGIS project
            for layer in layers:
                layers_list.append(layer.name())

            if self.mlyr_name == '':
                self.mlyr_name = lct.get_tmp_name()

            if self.mlyr_name not in layers_list:
                self.create_mem_lyr(self.mlyr_name)

            out_lyr = self.iface.activeLayer()
            out_lyr.startEditing()
            out_prov = out_lyr.dataProvider()
            feat = QgsFeature()

            out_csv_fnames = ['P_NAME', 'BRNG', 'DIST', 'LAT_DMS', 'LON_DMS', 'POLAR_COOR', 'NOTES']
            with open(self.input_file, 'r') as inCSV:
                with open(self.output_file, 'w') as outCSV:
                    reader = csv.DictReader(inCSV, delimiter=';')
                    writer = csv.DictWriter(outCSV, fieldnames=out_csv_fnames, delimiter=';')
                    for row in reader:
                        valid_azm_dist, notes = lct.check_azm_dist(row['BRNG'], row['DIST'])
                        if valid_azm_dist:  # azimuth or brng and distance are valid
                            csv_points_uom = self.get_csv_points_uom()
                            dist_m = lct.to_meters(float(row['DIST']), csv_points_uom)
                            cp_brng = lct.Bearing(row['BRNG'])
                            calc_point = lct.PolarCoordPoint(self.ref_point, cp_brng, dist_m)
                            cp_name = row['P_NAME']
                            cp_def = 'Ref point: {} Bearing {} {} Distance: {} {}'.format(self.ref_point.origin_id,
                                                                                          cp_brng.src_value,
                                                                                          true_mag,
                                                                                          row['DIST'],
                                                                                          csv_points_uom)
                            # Write result to output file
                            writer.writerow({'P_NAME': row['P_NAME'],
                                             'BRNG': row['BRNG'],
                                             'DIST': row['DIST'],
                                             'LAT_DMS': str(calc_point.ep_lat_dd),
                                             'LON_DMS': str(calc_point.ep_lon_dd),
                                             'POLAR_COOR': cp_def,
                                             'NOTES': ''})
                            # Write result to temporary layer
                            cp_qgs_point = QgsPointXY(calc_point.ep_lon_dd, calc_point.ep_lat_dd)
                            cp_attributes = [self.ref_point.origin_id,
                                             self.ref_point.src_lat,
                                             self.ref_point.src_lon,
                                             cp_name,
                                             calc_point.ep_lat_dd,
                                             calc_point.ep_lon_dd,
                                             cp_def]
                            feat.setGeometry(QgsGeometry.fromPointXY(cp_qgs_point))
                            feat.setAttributes(cp_attributes)
                            out_prov.addFeatures([feat])
                        else:
                            pass

            out_lyr.commitChanges()
            out_lyr.updateExtents()
            self.iface.mapCanvas().setExtent(out_lyr.extent())
            self.iface.mapCanvas().refresh()

    def run(self):
        """Run method that performs all the real work"""
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
