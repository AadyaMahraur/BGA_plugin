// add aoi into gee assets and import as roi
var start_date = "2000-01-01";
var end_date = '2000-12-28';

var start_date_VIIRS = '2000-01-01';
var end_date_VIIRS = '2000-12-01';

// Load built-up shp from Assets


var L5_dataset = ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
  .filterBounds(roi)
  .filterDate(start_date, end_date);

var L5_clip = L5_dataset.mosaic().clip(roi);
var L5_clip_RGB = L5_clip.select(['B3', 'B2', 'B1']);

var trueColor321Vis = {
  min: 0.0,
  max: 0.4,
  gamma: 1.2
};

Map.addLayer(L5_clip_RGB, trueColor321Vis, 'L5 -Satellite_Image (NCC)');
Map.centerObject(roi, 12);

// Nighttime data
var dataset_VIIRS_Night = ee.ImageCollection('BNU/FGS/CCNL/v1').filter(ee.Filter.date(start_date_VIIRS,end_date_VIIRS));
var nighttime = dataset_VIIRS_Night.select('b1').mosaic().clip(roi);
var nighttimeVis = {min: 0.0, max: 60.0};
Map.addLayer(nighttime, nighttimeVis, 'Nighttime_Data', false);

// Nightlight stats
var night_stats = nighttime.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: roi.geometry(),
  crs: 'EPSG:4326',
  scale: 450
});
print('Nightlight min/max:', night_stats);

// Built-up data
var worldpop = ee.Image('JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1').select('built').clip(roi);
var built_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
var remap_values = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
var built_binary = worldpop.remap(built_values, remap_values).rename('remapped');
Map.addLayer(built_binary, {}, 'Built-Up Remapped', false);

// NDVI mask
var ndvi = L5_clip.normalizedDifference(['B4', 'B3']).rename('NDVI');
var ndvi_mask = ndvi.lt(0.2).and(ndvi.gt(0));
var built_masked = built_binary.multiply(ndvi_mask);
Map.addLayer(built_masked, {}, 'Built-Up Masked NDVI', false);

// Combine datasets
var ls_append_night = L5_clip.addBands(nighttime);
var ls_append_built = L5_clip.addBands(built_masked);
var ls_append_night_built = ls_append_night.addBands(built_masked);

// Sample data for classification
var training_samples = function(image, num_points) {
  return image.stratifiedSample({
    numPoints: num_points,
    classBand: 'remapped',
    region: roi.geometry(),
    scale: 30,
    geometries: true
  });
};

var training_1 = training_samples(ls_append_night_built, 1000);
var training_2 = training_samples(ls_append_built, 1000);

// Train SVM classifiers
var bands_l5 = ['B1','B2','B3','B4','B5','B6','B7'];
var bands_night = bands_l5.concat(['b1']);

var svm_1 = ee.Classifier.libsvm({
  kernelType: 'RBF', 
  gamma: 1, 
  cost: 10
}).train({
  features: training_1, 
  classProperty: 'remapped', 
  inputProperties: bands_night
});

var svm_2 = ee.Classifier.libsvm({
  kernelType: 'RBF', 
  gamma: 1, 
  cost: 10
}).train({
  features: training_2, 
  classProperty: 'remapped', 
  inputProperties: bands_l5
});

// Classify
var classified_1 = ls_append_night.select(bands_night).classify(svm_1);
var classified_2 = L5_clip.select(bands_l5).classify(svm_2);

// Final output (AND operation)
var final_classified = classified_1.and(classified_2);

Map.addLayer(classified_2, {min: 0, max: 1, palette: ['black', 'yellow']}, 'SVM Built Only', false);
Map.addLayer(final_classified, {min: 0, max: 1, palette: ['black', 'red']}, 'Final Built-Up SVM');
