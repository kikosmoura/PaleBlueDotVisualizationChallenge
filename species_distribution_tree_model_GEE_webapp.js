

//var palette = '000096,9beb4a,af0000'
 
var legendLabels = ui.Panel({
        widgets: [
          ui.Label('0%', {margin: '4px 8px'}),
          ui.Label(('50%'), {margin: '4px 8px', textAlign: 'center', stretch: 'horizontal'}),
          ui.Label('100%', {margin: '4px 8px'})
        ],
        layout: ui.Panel.Layout.flow('horizontal')
      });

var legendTitle2 = ui.Label({
        value: 'Click on a point on the map of Brazil',
        style: {fontWeight: 'bold'}
      });


var legendTitle = ui.Label({
        value: 'Probability of Occurrence of the Specie',
        style: {fontWeight: 'bold'}
      });


var vis = {min:0, max:100, palette: ['000096','0064ff','00b4ff','33db80','9beb4a','ffeb00','ffb300','ff6400','eb1e00','af0000']};
      function makeColorBarParams(palette) {
        return {
          bbox: [0, 0, 1, 0.1],
          dimensions: '100x10',
          format: 'png',
          min: 0,
          max: 1,
          palette: palette,
        };
      }

var colorBar = ui.Thumbnail({
        image: ee.Image.pixelLonLat().select(0),
        params: makeColorBarParams(vis.palette),
        style: {stretch: 'horizontal', margin: '0px 8px', maxHeight: '24px'},
      });

/////////////// Criar um painel para a legenda
      
var legend = ui.Panel({
  style: {
    position: 'top-left',
    padding: '8px 50px'
  }
  });
      

Map.add(legendTitle2)
Map.add(legend.add(legendTitle).add(colorBar).add(legendLabels))


///////////////////////////////////////////////////////
//Map.addLayer(modelo, {min: 0.182934, max: 0.975633});
//Map.centerObject(modelo, 6);

Map.setControlVisibility({
  drawingToolsControl:false
})

var imagem_selecionada;
var nome_especie;

var layers = {
  'Hymenaea courbaril': hymenae_courbaril,
  'Cedrela odorata': cedrela_odorata,
  'Couratari guianensis': couratari_guianensis,
  'Handroanthus albus': handroanthus_albus,
  'Handroanthus incanus':handroanthus_incanus,
  'Swietenia macrophylla':swietenia_macrophylla,
  'Araucaria angustifolia':araucaria_angustifolia,
  'Euterpe oleracea':euterpe_oleracea

}

//create a function
function changeLayers(layerKey) {
  Map.layers().reset()
  imagem_selecionada = layers[layerKey]
  nome_especie = layerKey
  
  var precColors = '000096,0064ff,00b4ff,33db80,9beb4a,ffeb00,ffb300,ff6400,eb1e00,af0000';
  Map.addLayer(imagem_selecionada, {min: 0.182934, max: 0.975633, palette: precColors});
  Map.setCenter(-51.98, -12.43, 5);
  s.style().set('backgroundColor', 'lightgray');
  var widgetStyle = s.style();
  widgetStyle.set({border: '2px solid darkgray'});
  
}

// Make a selection ui.element that will update the layer
var s = ui.Select({
  placeholder: 'select a tree species',
  items: [
    {value: 'Hymenaea courbaril', label: 'Hymenaea courbaril'}, 
    {value: 'Cedrela odorata', label: 'Cedrela odorata'},
    {value: 'Couratari guianensis', label: 'Couratari guianensis'},
    {value: 'Handroanthus albus', label: 'Handroanthus albus'},
    {value: 'Handroanthus incanus', label: 'Handroanthus incanus'},
    {value: 'Swietenia macrophylla', label: 'Swietenia macrophylla'},
    {value: 'Araucaria angustifolia', label: 'Araucaria angustifolia'},
    {value: 'Euterpe oleracea', label: 'Euterpe oleracea'}
  
  ],
  onChange: changeLayers
})
//Map.add(s)

//print(imagem_selecionada instanceof ee.Image) 
//print(s.items()) 
//var precColors = '000096,0064ff,00b4ff,33db80,9beb4a,ffeb00,ffb300,ff6400,eb1e00,af0000';
Map.setCenter(-51.98, -12.43, 5);
Map.add(s, {min: 0.182934, max: 0.975633})
//Map.setCenter(-51.98, -12.43, 4);
//s.style().set('backgroundColor', 'lightgray');
//var widgetStyle = s.style();
//widgetStyle.set({border: '2px solid darkgray'});




//////////////// Criar um INSPECTOR

var header = ui.Label('Point Information', {fontWeight: 'bold', fontSize: '15px'})
var ToolPanel = ui.Panel([header], 'flow', {width: '200px', position: 'bottom-right'})


Map.onClick(function(coords) {
      var location = 'lon: ' + coords.lon.toFixed(4) + ' ' +
                     'lat: ' + coords.lat.toFixed(4);
  var click_point = ee.Geometry.Point(coords.lon, coords.lat);
  var imagem = imagem_selecionada.reduceRegion(ee.Reducer.min(), click_point, 10).evaluate(function(val){
  var demText = 'Probability of Occurrence of the Specie ' + nome_especie + ': ' +(val.b1*100).toFixed(1) + "%";
  
     ToolPanel.widgets().set(1, ui.Label(demText));


       })
  ToolPanel.widgets().set(2, ui.Label(location)); 
  Map.remove(ToolPanel)
  Map.add(ToolPanel)
})

 

Map.style().set({cursor: 'crosshair'})










