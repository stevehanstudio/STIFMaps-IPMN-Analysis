setImageType('FLUORESCENCE');
mergeSelectedAnnotations()
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":true,"constrainToParent":true}')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":true,"constrainToParent":false}')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":false,"constrainToParent":true}')
selectAnnotations();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":false,"channel2":true,"channel3":false,"channel4":false,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":false,"haralickMin":NaN,"haralickMax":NaN,"haralickDistance":1,"haralickBins":32}')
selectAnnotations();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":true,"channel2":true,"channel3":true,"channel4":true,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":false,"haralickMin":NaN,"haralickMax":NaN,"haralickDistance":1,"haralickBins":32}')
saveAnnotationMeasurements('/C:/Users/tomma/OneDrive/1_SCIENCE/1_RESEARCH/2024/2024_IPMN_stifmap/analysis/')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":false,"constrainToParent":true}')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":false,"constrainToParent":false}')
clearSelectedObjects(true);
clearSelectedObjects();
mergeSelectedAnnotations()
mergeSelectedAnnotations()
mergeSelectedAnnotations()
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":false,"constrainToParent":false}')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '{"radiusMicrons":100.0,"lineCap":"ROUND","removeInterior":false,"constrainToParent":false}')
clearSelectedObjects(true);
clearSelectedObjects();
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 72.70867535498915,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 47.349631233223896,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 50.24541009758284,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 81.93152881539753,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 73.53319182519905,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 38.627779725280995,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 41.72969178630448,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 100.0,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 51.44449536387555,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
runPlugin('qupath.lib.plugins.objects.DilateAnnotationPlugin', '
    {
      "radiusMicrons": 86.81253318499189,
      "lineCap":"ROUND",
      "removeInterior":true,
      "constrainToParent":true
    }
    ')
selectAnnotations();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons":2.0,"region":"ROI","tileSizeMicrons":25.0,"channel1":false,"channel2":true,"channel3":false,"channel4":false,"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,"doHaralick":false,"haralickMin":NaN,"haralickMax":NaN,"haralickDistance":1,"haralickBins":32}')
clearSelectedObjects(true);
clearSelectedObjects();
exportAllObjectsToGeoJson("B:\\Projects\\WeaverLab\\analysis_panel_1\\6488_annotations.geojson", "FEATURE_COLLECTION")
