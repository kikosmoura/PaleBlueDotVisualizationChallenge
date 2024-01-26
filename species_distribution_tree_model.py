# Import the necessary libraries
import io
import ee
import folium
import geemap
import random
import time

# Autenticar no Earth Engine
ee.Authenticate()
ee.Initialize()

#################################################################################################################
# This code is using the Earth Engine Python API to load data from a table called "neotreep_v4"
# for the species Luehea divaricata and defining a spatial resolution of 1000 meters.
# Then the "RemoveDuplicates" function is being defined to remove duplicate records from the table.
# To do this, a random image is created with the projection EPSG: 4326 and zero spatial resolution.
# Then, a sampling of points with a scale of 10 meters is carried out for the data in the table and the
# "distinct" function is applied to random point values, returning a collection of data without duplicates.
# It is important to remember that the effectiveness of the "RemoveDuplicates" function depends on how well distributed
# spatially the data is. If there are clusters of points close to each other, the function may not remove
# all duplicate records.
#################################################################################################################

# Data loading from the neotreep table for the species Luehea divaricata
#https://code.earthengine.google.com/?asset=projects/ee-kikosmoura/assets/df_neotree_amazonia
Data = ee.FeatureCollection('projects/ee-kikosmoura/assets/df_arroz')
#Data = ee.FeatureCollection('users/kikosmoura_ml_01/neotreep_v4')

# Set spatial resolution in meters
GrainSize = 1000

def RemoveDuplicates(data):
    randomraster = ee.Image.random().reproject('EPSG:4326', None, GrainSize)
    randpointvals = randomraster.sampleRegions(collection=ee.FeatureCollection(data), scale=10, geometries=True)
    
    return randpointvals.distinct('random')

Data = RemoveDuplicates(Data)

###############################################################################################################
# This code is using the Earth Engine Python API to load a collection of boundary geometries
# global admins, calling 'USDOS/LSIB_SIMPLE/2017' and filtering only matching geometries
# to Brazil, based on the country's ISO code ('BR').
# Then the "geometry()" function is applied to the filtered collection to extract the country's geometry and
# store it in the variable "AOI" (area of interest). The resulting geometry is a polygonal representation of the
# land area of Brazil, which can be used to carry out image and data processing operations
# in a way that is spatially limited to the country.
###############################################################################################################

# Defininfo o Brasil como região de interesse
AOI = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'BR')).geometry()

type(AOI)

####################################################################################################################
# This code is using the Earth Engine Python API to create a dataset of predictor variables
# for spatial modeling and analysis.
# First, three images are loaded from Earth Engine:
# 'WORLDCLIM/V1/BIO': represents data from 19 bioclimatic variables, such as temperature, precipitation and humidity,
# at 1 km spatial resolution.
# 'USGS/SRTMGL1_003': represents elevation data at 30 meter spatial resolution.
# 'MODIS/006/MOD44B': represents global vegetation cover data, derived from MODIS sensor images,
# at 500 meter spatial resolution.
# Next, preprocessing operations are applied to the data to create a set of predictor variables.
# The first step is to calculate the median vegetation cover for the period 2003 to 2020, in order to obtain a
# estimate of average vegetation cover for the region of interest.
# Then the three images are combined using the "addBands" function to form a single image with the
# predictor variables. The resulting image is then cropped to the area of interest defined by the variable
# 'AOI' and a water mask is created using elevation (elevation values greater than 0 are considered land).
# Finally, the image of predictor variables is updated with the water mask and only the bands of interest
#are selected ("bio04", "bio05", "bio06", "bio12", "elevation" and "Percent_Tree_Cover").
# The end result is a dataset of predictor variables that can be used for modeling or
# spatial analysis.
####################################################################################################################

LCT = ee.ImageCollection('MODIS/061/MCD12Q1').select(['LC_Type1']).median()

SO2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2').filterDate('2019-06-01', '2019-06-11').select(['SO2_column_number_density']).median()
       
PH = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')

# Carregue uma imagem multibanda do catálogo de dados
BIO = ee.Image("WORLDCLIM/V1/BIO")

# Load elevation data from the data catalog and calculate the slope, aspect and a simple shadow of the elevation model
# digital elevation of the terrain.
Terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))

# Load the NDVI 250 m collection and estimate the average annual tree cover value per pixel
MODIS = ee.ImageCollection("MODIS/006/MOD44B")
MedianPTC = MODIS.filterDate('2003-01-01', '2020-12-31').select(['Percent_Tree_Cover']).median()

# Combine bands into a single multiband image
predictors = BIO.addBands(Terrain).addBands(MedianPTC).addBands(PH).addBands(SO2).addBands(LCT)

# Mask ocean pixels from the predictor variable image
watermask =  Terrain.select('elevation').gt(0) # Cria uma máscara de água
predictors = predictors.updateMask(watermask).clip(AOI)

# Select subset of bands to maintain habitat suitability modeling
bands = ['bio04','bio05','bio06','bio12','elevation','Percent_Tree_Cover', 'b100', 'LC_Type1','SO2_column_number_density']
#predictors = predictors.select(bands)

##################################################################################################################
# This code is performing two main operations using the predictor variables created earlier.
# The first operation is to randomly sample a number of pixels from the predictor image. This is done using
# the "sample" method, which receives as arguments the sampling scale (in meters), the number of pixels to be
# sampled and a Boolean value to indicate whether the geometries (in this case, the area of interest defined by the
# variable "DataCor") must be included in the sampling results.
# The second operation is to sample the values of the predictor variables at points of interest defined by the
# variable "DataColor". This is done using the "sampleRegions" method, which takes as arguments the collection of points
# of interest, the sampling scale (in meters) and the "tileScale" (which controls the size of the tiles used
# for parallel processing). The result of this operation is a data table containing the values of all
# the predictor bands for each point of interest in the study area.
##################################################################################################################

DataCor = predictors.sample(scale=GrainSize, numPixels=5000, geometries=True) # Gera 5000 pontos aletórios
PixelVals = predictors.sampleRegions(collection=DataCor, scale=GrainSize, tileScale=16) # Extrair valores de covariáveis

##############################################################################################################
# This code aims to create a mask for the area of interest (AOI), applying a segmentation
# based on clustering (K-means), using a random sample of pixels for grouping.
# Reduces the Data table to a binary image where each pixel is equal to 1 if there is at least one non-zero value in the
# column 'random' and 0 otherwise. The image is reprojected to the EPSG:4326 projection and with the resolution
# specified in GrainSize. The image is masked to exclude pixels that are outside the area of interest.
# Randomly samples 200 pixels from the predictors image with the sampleRegions function.
# Uses the K-means clustering algorithm from the Weka package to cluster these sampled pixels into 2 groups
# distinct.
# Creates an image in which each pixel belongs to one of two groups and is represented by a random color.
# This image is added to the map.
# Randomly selects another 200 pixels from the clustered image and assigns each one the cluster number
# corresponding. It then calculates the mode of the cluster in which these pixels fall.
# Use the binary image generated in step 1 to create a mask and apply a second mask based on the
# cluster mode calculated in step 5. Returns a cropped image for the area of interest it represents
# the area for which preservation actions can be recommended based on the generated clusters.
##############################################################################################################

mask = Data.reduceToImage(
    properties=['random'],
    reducer=ee.Reducer.first()
).reproject('EPSG:4326', None, ee.Number(GrainSize)).mask().neq(1).selfMask()

# Extract environmental values for a random subset of presence data
PixelVals = predictors.sampleRegions(
    collection=Data.randomColumn().sort('random').limit(200),
    properties=[],
    tileScale=16,
    scale=GrainSize
)
# Perform k-means clustering on the cluster and train it using based on Euclidean distance.
clusterer = ee.Clusterer.wekaKMeans(
    nClusters=2,
    distanceFunction="Euclidean"
).train(PixelVals)

# Assign pixels to clusters using the trained clusterer
Clresult = predictors.cluster(clusterer)

# Display cluster results and identify cluster IDs for similar pixels and
# different from presence data
right = ee.Image(0).addBands(Clresult.randomVisualizer())
#Map.addLayer(right, {}, 'Clusters', 0)

# Mask pixels that are different from presence data.
# Get the cluster ID similar to the presence data and use the opposite cluster to define the allowed area
# to create pseudo-absences
clustID = Clresult.sampleRegions(
    collection=Data.randomColumn().sort('random').limit(200),
    properties=[],
    tileScale=16,
    scale=GrainSize
)
clustID = ee.FeatureCollection(clustID).reduceColumns(
    reducer=ee.Reducer.mode(),
    selectors=['cluster']
)
clustID = ee.Number(clustID.get('mode')).subtract(1).abs()
mask2 = Clresult.select(['cluster']).eq(clustID)
AreaForPA = mask.updateMask(mask2).clip(AOI)


################################################################################################################
# This code defines the makeGrid function that creates a grid of polygonal cells of size defined by the
# scale parameter within a given geometry defined by geometry. To create the grid, the code uses
# the lonLat image from the Google Earth Engine (GEE) platform, which contains longitude and latitude information
# for each pixel in the image. This image is then used to create lonGrid and latGrid images that contain
# longitude and latitude grids respectively.
# The reduceToVectors function is then applied to the grid of polygonal cells to calculate the average value of
# a given image (watermask in this case) within each polygonal cell. The filter function is used to
# remove cells with no value (where the average is equal to None). The end result is a feature collection (Grid)
# containing the polygonal cells with the average value of the watermask image.
################################################################################################################

# Define a function to create a grid over AOI
def makeGrid(geometry, scale):
    # pixelLonLat returns an image with each pixel labeled with longitude and latitude values.
    lonLat = ee.Image.pixelLonLat()
    # Select the longitude and latitude bands, multiply by a large number and truncate them to
    # integers.
    lonGrid = lonLat.select('longitude') \
                   .multiply(100000) \
                   .toInt()
    latGrid = lonLat.select('latitude') \
                   .multiply(100000) \
                   .toInt()
    return lonGrid.multiply(latGrid) \
                .reduceToVectors(geometry = geometry.buffer(distance=20000, maxError=1000), # O buffer permite que você verifique se a grade inclui as bordas da AOI.
                                 scale = scale,
                                 geometryType = 'polygon')

# Create grid and remove cells outside the AOI
Scale = 200000  # Defina o intervalo em m para criar blocos espaciais
grid = makeGrid(AOI, Scale)
Grid = watermask.reduceRegions(collection=grid, reducer=ee.Reducer.mean()).filter(ee.Filter.neq('mean', None))



# Define function to generate a vector of random numbers between 1 and 1000
def runif(length):
    return [random.randint(1, 1000) for i in range(length)]

###################################################################################################################
# This code defines a function SDM(x) that performs a species distribution modeling analysis
# (Species Distribution Modeling - SDM) in Google Earth Engine. The purpose of the function is to train a model
# classification using the Random Forest Classifier to predict the presence or absence of a given species
# in a given location based on environmental variables, such as temperature, precipitation, elevation, among others.
# The function uses data on the presence and absence of the species (or simulated points of presence), as well as points
# training and testing, which are generated using a spatial grid created by makeGrid(). The function also uses a
# dataset of predictors (or environmental variables) and a Random Forest classification model to generate
# a species classification map (probabilities or binary values). The function returns a list containing the
# probability classified map, the binary classified map, the training set and the test set.
###################################################################################################################

def SDM(x):
    Seed = ee.Number(x)
    
    # Randomly split blocks for training and validation
    GRID = ee.FeatureCollection(Grid).randomColumn(seed=Seed).sort('random')
    TrainingGrid = GRID.filter(ee.Filter.lt('random', split))  # Filtre pontos com propriedade 'aleatória' < porcentagem dividida
    TestingGrid = GRID.filter(ee.Filter.gte('random', split))  # Filtre pontos com propriedade 'aleatória' >= porcentagem dividida

    # Presence
    PresencePoints = ee.FeatureCollection(Data)
    PresencePoints = PresencePoints.map(lambda feature: feature.set('PresAbs', 1))
    TrPresencePoints = PresencePoints.filter(ee.Filter.bounds(TrainingGrid))  # Filtrar pontos de presença para treinamento
    TePresencePoints = PresencePoints.filter(ee.Filter.bounds(TestingGrid))  # Filtrar pontos de presença para teste
    
    # Pseudo-absences
    TrPseudoAbsPoints = AreaForPA.sample(region=TrainingGrid, scale=GrainSize, numPixels=TrPresencePoints.size().add(300), seed=Seed, geometries=True) # Adicionamos pontos extras para contabilizar aqueles pontos que caem em áreas mascaradas do raster e são descartados. Isso garante um conjunto de dados de presença/pseudo-ausência equilibrado
    TrPseudoAbsPoints = TrPseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TrPresencePoints.size())) # Reter aleatoriamente o mesmo número de pseudo-ausências como dados de presença 
    TrPseudoAbsPoints = TrPseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))
    
    TePseudoAbsPoints = AreaForPA.sample(region=TestingGrid, scale=GrainSize, numPixels=TePresencePoints.size().add(100), seed=Seed, geometries=True) # Adicionamos pontos extras para contabilizar aqueles pontos que caem em áreas mascaradas do raster e são descartados. Isso garante um conjunto de dados de presença/pseudo-ausência equilibrado
    TePseudoAbsPoints = TePseudoAbsPoints.randomColumn().sort('random').limit(ee.Number(TePresencePoints.size())) # Reter aleatoriamente o mesmo número de pseudo-ausências como dados de presença
    TePseudoAbsPoints = TePseudoAbsPoints.map(lambda feature: feature.set('PresAbs', 0))

    # Merge presence and pseudo-absence points
    trainingPartition = TrPresencePoints.merge(TrPseudoAbsPoints)
    testingPartition = TePresencePoints.merge(TePseudoAbsPoints)

    # Extract local covariate values from the multiband predictor image at the training points
    trainPixelVals = predictors.sampleRegions(collection=trainingPartition, properties=['PresAbs'], scale=GrainSize, tileScale=16, geometries=True)

    # Sort using random forest
    Classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=500,
        variablesPerSplit=None, 
        minLeafPopulation=10, 
        bagFraction=0.5, 
        maxNodes=None,
        seed=Seed 
    )
    
    
    # probability of presence
    ClassifierPr = Classifier.setOutputMode('PROBABILITY').train(trainPixelVals, 'PresAbs', bands) 
    ClassifiedImgPr = predictors.select(bands).classify(ClassifierPr)

    # Binary map of absence and presence
    ClassifierBin = Classifier.setOutputMode('CLASSIFICATION').train(trainPixelVals, 'PresAbs', bands) 
    ClassifiedImgBin = predictors.select(bands).classify(ClassifierBin)

    return ee.List([ClassifiedImgPr, ClassifiedImgBin, trainingPartition, testingPartition])
   
 ###################################################################################################################
# This code defines a "split" variable to determine the proportion of blocks used to select data from
# training. It then sets the "numiter" variable to 10 and applies the "SDM" function to a list of numbers
# provided: [35, 68, 43, 54, 17, 46, 76, 88, 24, 12].
# The "SDM" function is called for each of these numbers in the list. For each number, the "SDM" function performs a
# species distribution modeling analysis. The output of the "SDM" function for each number is stored in a
# list "results". Then the "results" list is flattened into a single results list.
###################################################################################################################


# Set partition for training and testing data
split = 0.70  # The proportion of blocks used to select training data

# Sets the number of repetitions
numiter = 10

# Although the runif function can be used to generate random seeds, we map the SDM function onto created numbers
# randomly for reproducibility of results
results = ee.List([35, 68, 43, 54, 17, 46, 76, 88, 24, 12]).map(SDM)

# Extract results from list
results = results.flatten()

##################################################################################################################
# This code creates a classification model based on a Randon Forest algorithm on a collection of images
# from Google Earth Engine.
# First, the code uses the ee.List.sequence() function to create a list of integers, which is used as
# indexes to access the images generated by the classification algorithm. Then the average of these images is
# calculated using the ee.ImageCollection.fromImages().mean() function, resulting in an average model.
# Finally, a distribution map is calculated, which represents the class with the highest frequency in each pixel
# between the images in the list, using the ee.ImageCollection.fromImages().mode() function.
##################################################################################################################


# Extract all predictions from the model
images = ee.List.sequence(0, ee.Number(numiter).multiply(4).subtract(1), 4).map(lambda x: results.get(x))

# Calculate the average of all individual model runs
ModelAverage = ee.ImageCollection.fromImages(images).mean()


# Extract all predictions from the model
images2 = ee.List.sequence(1, ee.Number(numiter).multiply(4).subtract(1), 4).map(lambda x: results.get(x))

# Calculate the average of all individual model runs
DistributionMap = ee.ImageCollection.fromImages(images2).mode()


# Export the image to Google Drive
task = ee.batch.Export.image.toDrive(
  image=DistributionMap,  
  description='PotentialDistribution_Arroz', 
  scale=GrainSize,  
  maxPixels=1e10,
  region=AOI  
)

# Start of export
task.start()

# Export the image to Google Drive
task = ee.batch.Export.image.toDrive(
  image=ModelAverage,  
  description='HSI_Handroanthus_Arroz',  
  scale=GrainSize,  
  maxPixels=1e10,
  region=AOI 
)

# Start the export task
task.start()  

