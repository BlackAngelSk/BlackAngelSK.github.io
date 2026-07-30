import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TEX_URL = {
  day: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg',
  night: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_lights_2048.png',
  clouds: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_clouds_2048.png',
  bump: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_normal_2048.jpg',
  spec: 'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_specular_2048.jpg',
};
const SEG = 256;
const BDR_W = 4096;
const BDR_H = 2048;
const BDR_URL = 'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson';

const CITIES = [
  ['New York',40.71,-74.01,1.2,'#ffdd44'],['London',51.51,-0.13,1.2,'#ff6644'],
  ['Tokyo',35.68,139.65,1.2,'#ff44aa'],['Paris',48.86,2.35,1.1,'#44aaff'],
  ['Sydney',-33.87,151.21,1.1,'#44ffaa'],['Moscow',55.76,37.62,1.1,'#ff8844'],
  ['Beijing',39.90,116.41,1.2,'#ff6644'],['Mumbai',19.08,72.88,1.0,'#ffaa44'],
  ['Cairo',30.04,31.24,1.0,'#ffcc44'],['Rio',-22.91,-43.17,1.0,'#44ff66'],
  ['Berlin',52.52,13.41,0.9,'#aaff66'],['Seoul',37.57,126.98,1.0,'#ff88aa'],
  ['Bangkok',13.76,100.50,0.9,'#ffaa88'],['Lagos',6.52,3.38,0.9,'#ffbb44'],
  ['Istanbul',41.01,28.98,1.0,'#ffcc66'],['Toronto',43.65,-79.38,0.9,'#66aaff'],
  ['Dubai',25.20,55.27,0.9,'#ffdd88'],['Singapore',1.35,103.82,0.9,'#ff66cc'],
  ['Rome',41.90,12.50,0.9,'#ffaa66'],['Mexico City',19.43,-99.13,1.0,'#ff88aa'],
  ['Washington',38.91,-77.04,0.9,'#66ccff'],['Brasilia',-15.80,-47.89,0.9,'#44ddff'],
  ['Buenos Aires',-34.60,-58.38,0.9,'#88ccff'],['Lima',-12.05,-77.04,0.8,'#88ddaa'],
  ['Jakarta',-6.21,106.85,0.9,'#ff4488'],['Nairobi',-1.29,36.82,0.7,'#aaffaa'],
  ['Johannesburg',-26.20,28.05,0.8,'#ffbb44'],['Stockholm',59.33,18.07,0.7,'#66aaff'],
  ['Oslo',59.91,10.75,0.7,'#66aaff'],['Copenhagen',55.68,12.57,0.7,'#66aaff'],
  ['Warsaw',52.23,21.01,0.8,'#aaff88'],['Vienna',48.21,16.37,0.8,'#ffcc88'],
  ['Madrid',40.42,-3.70,0.9,'#ffcc66'],['Lisbon',38.72,-9.14,0.7,'#ffaa88'],
  ['Amsterdam',52.37,4.90,0.8,'#ffcc44'],['Brussels',50.85,4.35,0.7,'#ffcc44'],
  ['Zurich',47.38,8.54,0.7,'#aaffcc'],['Munich',48.14,11.58,0.8,'#aaffcc'],
  ['Athens',37.98,23.73,0.8,'#ffcc66'],['Helsinki',60.17,24.94,0.7,'#88aaff'],
  ['Dublin',53.35,-6.26,0.7,'#ffdd66'],['Barcelona',41.39,2.17,0.9,'#ffcc88'],
  ['Budapest',47.50,19.04,0.7,'#ffdd88'],['Prague',50.08,14.44,0.7,'#ffdd88'],
  ['Havana',23.11,-82.37,0.7,'#ff88aa'],['Ankara',39.93,32.86,0.8,'#ffcc88'],
  ['Riyadh',24.71,46.68,0.8,'#ffcc88'],['Tehran',35.69,51.39,0.9,'#ff99aa'],
  ['Karachi',24.86,67.00,0.9,'#ffaa44'],['Delhi',28.70,77.10,1.0,'#ffaa44'],
  ['Dhaka',23.81,90.41,0.9,'#ffaa44'],['Manila',14.60,120.98,0.8,'#ff88aa'],
  ['Hong Kong',22.32,114.17,0.9,'#ffcc44'],['Shanghai',31.23,121.47,1.2,'#ff6644'],
  ['Taipei',25.03,121.57,0.7,'#ff88aa'],['Bratislava',48.15,17.11,0.6,'#ffcc88'],
  ['Ljubljana',46.06,14.51,0.5,'#88aaff'],['Zagreb',45.82,15.98,0.6,'#ffcc88'],
  ['Beograd',44.79,20.45,0.6,'#ffcc88'],['Sofia',42.70,23.32,0.6,'#ffcc88'],
  ['Bucharest',44.43,26.10,0.7,'#ffcc88'],['Tallinn',59.44,24.75,0.5,'#88aaff'],
  ['Riga',56.95,24.11,0.5,'#88aaff'],['Vilnius',54.69,25.28,0.5,'#88aaff'],
  ['Minsk',53.90,27.56,0.7,'#aaff88'],['Kyiv',50.45,30.52,0.8,'#aaff88'],
  ['Tbilisi',41.72,44.83,0.5,'#ffcc88'],['Yerevan',40.18,44.50,0.5,'#ffcc88'],
  ['Baku',40.41,49.87,0.6,'#ffcc88'],['Tashkent',41.30,69.24,0.6,'#ffcc88'],
  ['Almaty',43.22,76.85,0.6,'#ffcc88'],['Doha',25.29,51.53,0.7,'#ffcc88'],
  ['Kuwait City',29.38,47.98,0.7,'#ffcc88'],['Colombo',6.93,79.86,0.6,'#ffcc88'],
  ['San Francisco',37.77,-122.42,0.9,'#66ccff'],['LA',34.05,-118.24,0.9,'#66ccff'],
  ['Chicago',41.88,-87.63,0.9,'#66ccff'],['Vancouver',49.28,-123.12,0.8,'#66aaff'],
  ['Melbourne',-37.81,144.96,0.8,'#44ffcc'],['Auckland',-36.85,174.76,0.7,'#44ffaa'],
  ['Osaka',34.69,135.50,0.8,'#ff44aa'],['Sao Paulo',-23.55,-46.63,1.0,'#44ff66'],
  ['Houston',29.76,-95.37,0.8,'#66ccff'],['Denver',39.74,-104.99,0.7,'#66ccff'],
  ['Boston',42.36,-71.06,0.8,'#66ccff'],['Miami',25.76,-80.19,0.7,'#66ccff'],
  ['Seattle',47.61,-122.33,0.7,'#66aaff'],['Santiago',-33.45,-70.67,0.8,'#88ccff'],
  ['Bogota',4.71,-74.07,0.8,'#ff88aa'],['Montevideo',-34.90,-56.17,0.6,'#88ccff'],
  ['Asuncion',-25.26,-57.58,0.6,'#88ccff'],['La Paz',-16.49,-68.12,0.5,'#88ccff'],
  ['Quito',-0.18,-78.47,0.7,'#ff88aa'],['Guayaquil',-2.17,-79.92,0.5,'#ff88aa'],
  ['Caracas',10.48,-66.91,0.7,'#ff88aa'],['Guadalajara',20.66,-103.35,0.8,'#ff88aa'],
  ['Monterrey',25.69,-100.32,0.7,'#ff88aa'],['Santo Domingo',18.49,-69.93,0.7,'#ff88aa'],
  ['Kingston',18.02,-76.81,0.5,'#ff88aa'],['Casablanca',33.57,-7.59,0.7,'#ffcc44'],
  ['Tunis',36.81,10.18,0.7,'#ffcc44'],['Algiers',36.75,3.06,0.7,'#ffcc44'],
  ['Rabat',34.02,-6.84,0.5,'#ffcc44'],['Accra',5.60,-0.19,0.6,'#ffbb44'],
  ['Dakar',14.72,-17.47,0.6,'#ffbb44'],['Abuja',9.06,7.49,0.7,'#ffbb44'],
  ['Kampala',0.35,32.58,0.5,'#aaffaa'],['Dar es Salaam',-6.79,39.21,0.6,'#aaffaa'],
  ['Addis Ababa',9.03,38.75,0.7,'#aaffaa'],['Maputo',-25.97,32.57,0.5,'#aaffaa'],
  ['Harare',-17.83,31.03,0.5,'#aaffaa'],['Khartoum',15.50,32.56,0.7,'#aaffaa'],
  ['Kinshasa',-4.44,15.27,0.8,'#aaffaa'],['Luanda',-8.84,13.29,0.7,'#aaffaa'],
  ['Guangzhou',23.13,113.26,0.9,'#ff6644'],['Shenzhen',22.54,114.06,0.9,'#ff6644'],
  ['Chengdu',30.57,104.07,0.8,'#ff6644'],['Wuhan',30.59,114.31,0.8,'#ff6644'],
  ['Hangzhou',30.27,120.15,0.8,'#ff6644'],['Nanjing',32.06,118.80,0.7,'#ff6644'],
  ['Tianjin',39.34,117.36,0.8,'#ff6644'],['Chongqing',29.56,106.55,0.8,'#ff6644'],
  ['Sapporo',43.06,141.35,0.6,'#ff44aa'],['Hiroshima',34.39,132.46,0.6,'#ff44aa'],
  ['Fukuoka',33.59,130.40,0.6,'#ff44aa'],['Kyoto',35.01,135.77,0.6,'#ff44aa'],
  ['Yokohama',35.44,139.64,0.6,'#ff44aa'],['Brisbane',-27.47,153.03,0.7,'#44ffaa'],
  ['Perth',-31.95,115.86,0.7,'#44ffaa'],['Cape Town',-33.92,18.42,0.7,'#44ffaa'],
  ['Damascus',33.51,36.28,0.7,'#ffcc88'],['Beirut',33.89,35.50,0.6,'#ffcc88'],
  ['Baghdad',33.31,44.36,0.8,'#ffcc88'],['Mecca',21.39,39.86,0.6,'#ffcc88'],
  ['Ahmedabad',23.02,72.57,0.7,'#ffaa44'],['Hyderabad',17.39,78.49,0.8,'#ffaa44'],
  ['Chennai',13.08,80.27,0.8,'#ffaa44'],['Kolkata',22.57,88.36,0.9,'#ffaa44'],
  ['Bangalore',12.97,77.59,0.9,'#ffaa44'],['Pune',18.52,73.86,0.7,'#ffaa44'],
  ['Jaipur',26.91,75.79,0.6,'#ffaa44'],['Tel Aviv',32.09,34.78,0.6,'#ffcc88'],
  ['Novosibirsk',55.01,82.94,0.6,'#ff8844'],['Yekaterinburg',56.84,60.61,0.6,'#ff8844'],
  ['Kazan',55.79,49.12,0.5,'#ff8844'],['St. Petersburg',59.93,30.34,1.0,'#ff8844'],
  ['Ibadan',7.38,3.95,0.6,'#ffbb44'],['Mogadishu',2.05,45.32,0.5,'#aaffaa'],
  ['Cape Town',-33.92,18.42,0.7,'#44ffaa'],['Jeddah',21.49,39.19,0.7,'#ffcc88'],
  ['Sendai',38.27,140.87,0.5,'#ff44aa'],['Adelaide',-34.93,138.60,0.6,'#44ffaa'],
  ['Canberra',-35.28,149.13,0.5,'#44ffaa'],['Wellington',-41.29,174.78,0.5,'#44ffaa'],
  ['Nice',43.71,7.26,0.5,'#ffcc66'],['Milan',45.46,9.19,0.8,'#ffccaa'],
  ['Naples',40.85,14.27,0.7,'#ffccaa'],['Venice',45.44,12.32,0.6,'#ffccaa'],
  ['Florence',43.77,11.26,0.5,'#ffccaa'],['Genoa',44.41,8.95,0.5,'#ffccaa'],
  ['Turin',45.07,7.69,0.6,'#ffccaa'],['Lyon',45.76,4.84,0.7,'#ffcc66'],
  ['Marseille',43.30,5.37,0.6,'#ffcc66'],['Toulouse',43.60,1.44,0.5,'#ffcc66'],
  ['Glasgow',55.86,-4.25,0.5,'#ffdd66'],['Edinburgh',55.95,-3.19,0.5,'#ffdd66'],
  ['Manchester',53.48,-2.24,0.6,'#ffdd66'],['Birmingham',52.49,-1.89,0.5,'#ffdd66'],
  ['Hamburg',53.55,9.99,0.6,'#aaffcc'],['Frankfurt',50.11,8.68,0.6,'#aaffcc'],
  ['Krakow',50.06,19.95,0.5,'#ffcc88'],['Cluj-Napoca',46.77,23.62,0.4,'#ffcc88'],
  ['Reykjavik',64.15,-21.94,0.5,'#88aaff'],['Nova Scotia',44.65,-63.57,0.3,'#66aaff'],
  ['Bogota',4.71,-74.07,0.8,'#ff88aa'],['Medellin',6.25,-75.57,0.7,'#ff88aa'],
  ['Cali',3.45,-76.53,0.5,'#ff88aa'],['Barranquilla',10.97,-74.78,0.5,'#ff88aa'],
  ['Port-au-Prince',18.59,-72.31,0.5,'#ff88aa'],['San Juan',18.47,-66.11,0.4,'#ff88aa'],
  ['Manaus',-3.12,-60.02,0.6,'#ff88aa'],['Belem',-1.46,-48.50,0.5,'#ff88aa'],
  ['Recife',-8.05,-34.88,0.5,'#ff88aa'],['Salvador',-12.97,-38.51,0.5,'#ff88aa'],
  ['Fortaleza',-3.72,-38.54,0.5,'#ff88aa'],['Curitiba',-25.43,-49.27,0.5,'#88ccff'],
  ['Porto Alegre',-30.03,-51.22,0.5,'#88ccff'],['Asequipa',-16.41,-71.54,0.5,'#88ddaa'],
  ['Trujillo',-8.12,-79.03,0.4,'#88ddaa'],['Cusco',-13.53,-71.97,0.5,'#88ddaa'],
  ['Cordoba',-31.42,-64.19,0.5,'#88ccff'],['Mendoza',-32.89,-68.83,0.4,'#88ccff'],
  ['Rosario',-32.95,-60.64,0.5,'#88ccff'],['La Plata',-34.92,-75.95,0.4,'#88ccff'],
  ['Valparaiso',-33.05,-71.61,0.5,'#88ccff'],
  // Animation placeholder at origin
  ['__origin__',0,1,0,0.1,'#ffffff'],
];

const COUNTRY_INFO = {
  'United States':{lat:39.83,lon:-98.58,info:'Population: ~331M | Capital: Washington D.C. | 50 states'},
  'Canada':{lat:56.13,lon:-106.35,info:'Population: ~38M | Capital: Ottawa'},
  'Mexico':{lat:19.43,lon:-99.13,info:'Population: ~130M | Capital: Mexico City'},
  'Brazil':{lat:-14.24,lon:-51.93,info:'Population: ~214M | Capital: Brasilia | Largest in South America'},
  'Argentina':{lat:-38.42,lon:-63.62,info:'Population: ~46M | Capital: Buenos Aires'},
  'United Kingdom':{lat:55.38,lon:-3.44,info:'Population: ~67M | Capital: London'},
  'France':{lat:46.23,lon:2.21,info:'Population: ~67M | Capital: Paris'},
  'Germany':{lat:51.17,lon:10.45,info:'Population: ~84M | Capital: Berlin'},
  'Italy':{lat:41.87,lon:12.57,info:'Population: ~60M | Capital: Rome'},
  'Spain':{lat:40.46,lon:-3.75,info:'Population: ~47M | Capital: Madrid'},
  'Japan':{lat:36.20,lon:138.25,info:'Population: ~125M | Capital: Tokyo'},
  'China':{lat:35.86,lon:104.20,info:'Population: ~1.4B | Capital: Beijing'},
  'India':{lat:20.59,lon:78.96,info:'Population: ~1.4B | Capital: New Delhi'},
  'Russia':{lat:61.52,lon:105.32,info:'Population: ~144M | Capital: Moscow'},
  'Australia':{lat:-25.27,lon:133.78,info:'Population: ~26M | Capital: Canberra'},
  'South Africa':{lat:30.56,lon:22.94,info:'Population: ~60M | Capital: Pretoria'},
  'Turkey':{lat:38.96,lon:35.24,info:'Population: ~85M | Capital: Ankara'},
  'Saudi Arabia':{lat:23.89,lon:45.08,info:'Population: ~35M | Capital: Riyadh'},
  'Egypt':{lat:26.82,lon:30.80,info:'Population: ~104M | Capital: Cairo'},
  'South Korea':{lat:35.91,lon:127.77,info:'Population: ~52M | Capital: Seoul'},
  'Indonesia':{lat:-0.79,lon:113.92,info:'Population: ~275M | Capital: Jakarta'},
  'Pakistan':{lat:30.38,lon:69.35,info:'Population: ~220M | Capital: Islamabad'},
  'Nigeria':{lat:9.08,lon:8.68,info:'Population: ~220M | Capital: Abuja'},
  'Ukraine':{lat:48.38,lon:31.17,info:'Population: ~44M | Capital: Kyiv'},
  'Poland':{lat:51.92,lon:19.15,info:'Population: ~38M | Capital: Warsaw'},
  'Sweden':{lat:60.13,lon:18.64,info:'Population: ~10M | Capital: Stockholm'},
  'Norway':{lat:60.47,lon:8.47,info:'Population: ~5M | Capital: Oslo'},
  'Finland':{lat:61.92,lon:25.75,info:'Population: ~6M | Capital: Helsinki'},
  'Denmark':{lat:56.26,lon:9.50,info:'Population: ~6M | Capital: Copenhagen'},
  'I Ireland':{lat:53.14,lon:-7.69,info:'Population: ~5M | Capital: Dublin'},
  'I Iceland':{lat:64.96,lon:-19.02,info:'Population: ~370K | Capital: Reykjavik'},
  'Belgium':{lat:50.50,lon:4.47,info:'Population: ~12M | Capital: Brussels'},
  'Netherlands':{lat:52.13,lon:5.29,info:'Population: ~17M | Capital: Amsterdam'},
  'Switzerland':{lat:46.82,lon:8.23,info:'Population: ~9M | Capital: Bern'},
  'Austria':{lat:47.52,lon:14.55,info:'Population: ~9M | Capital: Vienna'},
  'Greece':{lat:39.07,lon:21.82,info:'Population: ~11M | Capital: Athens'},
  'Portugal':{lat:39.40,lon:-8.22,info:'Population: ~10M | Capital: Lisbon'},
  'Romania':{lat:45.94,lon:24.97,info:'Population: ~19M | Capital: Bucharest'},
  'Croatia':{lat:45.10,lon:15.20,info:'Population: ~4M | Capital: Zagreb'},
  'Serbia':{lat:44.02,lon:21.01,info:'Population: ~7M | Capital: Belgrade'},
  'Bulgaria':{lat:42.73,lon:25.49,info:'Population: ~7M | Capital: Sofia'},
  'Czech Republic':{lat:49.82,lon:15.47,info:'Population: ~11M | Capital: Prague'},
  'Hungary':{lat:47.16,lon:19.50,info:'Population: ~10M | Capital: Budapest'},
  'Slovakia':{lat:48.67,lon:19.70,info:'Population: ~5M | Capital: Bratislava'},
  'Slovenia':{lat:46.15,lon:14.99,info:'Population: ~2M | Capital: Ljubljana'},
  'Estonia':{lat:58.60,lon:25.01,info:'Population: ~1.3M | Capital: Tallinn'},
  'Latvia':{lat:56.88,lon:24.60,info:'Population: ~1.8M | Capital: Riga'},
  'Lithuania':{lat:55.17,lon:23.88,info:'Population: ~2.8M | Capital: Vilnius'},
  'Belarus':{lat:53.71,lon:27.95,info:'Population: ~9M | Capital: Minsk'},
  'Israel':{lat:31.05,lon:34.85,info:'Population: ~9M | Capital: Jerusalem'},
  'UAE':{lat:23.42,lon:53.85,info:'Population: ~10M | Capital: Abu Dhabi'},
  'Qatar':{lat:25.35,lon:51.18,info:'Population: ~3M | Capital: Doha'},
  'Thailand':{lat:15.87,lon:100.99,info:'Population: ~72M | Capital: Bangkok'},
  'Vietnam':{lat:14.06,lon:108.28,info:'Population: ~98M | Capital: Hanoi'},
  'Singapore':{lat:1.35,lon:103.82,info:'Population: ~5.9M | City-state'},
  'Philippines':{lat:12.88,lon:121.77,info:'Population: ~111M | Capital: Manila'},
  'Malaysia':{lat:4.21,lon:101.98,info:'Population: ~33M | Capital: Kuala Lumpur'},
  'Morocco':{lat:31.79,lon:-7.09,info:'Population: ~37M | Capital: Rabat'},
  'Algeria':{lat:28.03,lon:1.66,info:'Population: ~44M | Capital: Algiers'},
  'Tunisia':{lat:33.89,lon:9.54,info:'Population: ~12M | Capital: Tunis'},
  'Ethiopia':{lat:9.15,lon:40.49,info:'Population: ~120M | Capital: Addis Ababa'},
  'Kenya':{lat:-0.02,lon:37.91,info:'Population: ~55M | Capital: Nairobi'},
  'Tanzania':{lat:-6.37,lon:34.89,info:'Population: ~63M | Capital: Dodoma'},
  'DR Congo':{lat:-4.04,lon:21.76,info:'Population: ~99M | Capital: Kinshasa'},
  'Sudan':{lat:12.86,lon:30.22,info:'Population: ~44M | Capital: Khartoum'},
  'Iraq':{lat:33.31,lon:43.68,info:'Population: ~41M | Capital: Bagddad'},
  'Iran':{lat:32.43,lon:53.69,info:'Population: ~85M | Capital: Tehran'},
  'Uzbekistan':{lat:41.38,lon:64.59,info:'Population: ~35M | Capital: Tashkent'},
  'Kazakhstan':{lat:48.02,lon:66.92,info:'Population: ~19M | Capital: Astana'},
  'Mongolia':{lat:46.86,lon:103.85,info:'Population: ~3M | Capital: Ulan Bator'},
  'Georgia':{lat:42.32,lon:43.36,info:'Population: ~4M | Capital: Tbilisi'},
  'Armenia':{lat:40.07,lon:45.04,info:'Population: ~3M | Capital: Yerevan'},
  'Azerbaijan':{lat:40.14,lon:47.58,info:'Population: ~10M | Capital: Baku'},
  'Syria':{lat:34.80,lon:39.00,info:'Population: ~18M | Capital: Damascus'},
  'Jordan':{lat:30.59,lon:36.24,info:'Population: ~11M | Capital: Amman'},
  'Lebanon':{lat:33.85,lon:35.86,info:'Population: ~5M | Capital: Beirut'},
  'Bangladesh':{lat:23.68,lon:90.36,info:'Population: ~170M | Capital: Dhaka'},
  'Myanmar':{lat:21.91,lon:95.96,info:'Population: ~55M | Capital: Nay Pyi Taw'},
  'Sri Lanka':{lat:7.87,lon:80.77,info:'Population: ~22M | Capital: Colombo'},
  'Peru':{lat:-9.19,lon:-75.02,info:'Population: ~33M | Capital: Lima'},
  'Chile':{lat:-35.68,lon:-71.54,info:'Population: ~19M | Capital: Santiago'},
  'Colombia':{lat:4.57,lon:-74.30,info:'Population: ~51M | Capital: Bogota'},
  'Venezuela':{lat:6.42,lon:-66.59,info:'Population: ~28M | Capital: Caracas'},
  'Ecuador':{lat:-1.83,lon:-78.18,info:'Population: ~18M | Capital: Quito'},
  'Bolivia':{lat:-16.29,lon:-63.59,info:'Population: ~12M | Capital: Sucre'},
  'Paraguay':{lat:-23.44,lon:-58.44,info:'Population: ~7M | Capital: Asuncion'},
  'Uruguay':{lat:-32.52,lon:-55.77,info:'Population: ~4M | Capital: Montevideo'},
  'New Zealand':{lat:-40.90,lon:174.89,info:'Population: ~5M | Capital: Wellington'},
  'Cuba':{lat:21.52,lon:-77.78,info:'Population: ~11M | Capital: Havana'},
  'Haiti':{lat:18.97,lon:-72.29,info:'Population: ~11M | Capital: Port-au-Prince'},
  'Dominican Republic':{lat:18.74,lon:-70.16,info:'Population: ~11M | Capital: Santo Domingo'},
  'Jamaica':{lat:18.11,lon:-77.30,info:'Population: ~3M | Capital: Kingston'},
  'Trinidad and Tobago':{lat:10.69,lon:-61.22,info:'Population: ~1.4M | Capital: Port of Spain'},
  'Tunisia':{lat:33.89,lon:9.54,info:'Population: ~12M | Capital: Tunis'},
  'Libya':{lat:26.34,lon:17.23,info:'Population: ~7M | Capital: Tripoli'},
  'Mozambique':{lat:-18.67,lon:35.53,info:'Population: ~32M | Capital: Maputo'},
  'Zimbabwe':{lat:-19.02,lon:29.15,info:'Population: ~15M | Capital: Harare'},
  'Ghana':{lat:7.95,lon:-1.02,info:'Population: ~32M | Capital: Accra'},
  'Senegal':{lat:14.50,lon:-14.45,info:'Population: ~17M | Capital: Dakar'},
  'Mali':{lat:17.57,lon:-4.00,info:'Population: ~21M | Capital: Bamako'},
  'Niger':{lat:17.61,lon:8.08,info:'Population: ~26M | Capital: Niamey'},
  'Burkina Faso':{lat:12.37,lon:-1.52,info:'Population: ~22M | Capital: Ouagadougou'},
  'Cameroon':{lat:7.37,lon:12.36,info:'Population: ~27M | Capital: Yaounde'},
  'Angola':{lat:-11.20,lon:17.87,info:'Population: ~34M | Capital: Luanda'},
  'Uganda':{lat:1.37,lon:32.29,info:'Population: ~47M | Capital: Kampala'},
  'Rwanda':{lat:-1.94,lon:29.87,info:'Population: ~13M | Capital: Kigali'},
  'Nepal':{lat:28.39,lon:84.12,info:'Population: ~30M | Capital: Kathmand'},
  'Liberia':{lat:6.43,lon:-9.43,info:'Population: ~5M | Capital: Monrovia'},
  'Sierra Leone':{lat:8.46,lon:-11.78,info:'Population: ~8M | Capital: Freetown'},
};

function latLonToV3(lat,lon,r){
  const p=(90-lat)*Math.PI/180,t=(lon+180)*Math.PI/180;
  return new THREE.Vector3(-r*Math.sin(p)*Math.cos(t),r*Math.cos(p),r*Math.sin(p)*Math.sin(t));
}

function ll2c(lon,lat,w,h){return [(lon+180)/360*w,(90-lat)/180*h];}
function dP(ctx,ring,w,h){
  ctx.beginPath();ring.forEach((c,i)=>{const[x,y]=ll2c(c[0],c[1],w,h);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.closePath();
}

async function mkBorderTex(w=BDR_W,h=BDR_H){
  const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.clearRect(0,0,w,h);
  try{
    updateProgress('Loading country borders...');
    const r=await fetch(BDR_URL);if(!r.ok)throw new Error('HTTP '+r.status);const d=await r.json();
    d.features.forEach(f=>{const g=f.geometry;if(!g)return;
      const dr=(rings)=>{rings.forEach((ring,i)=>{dP(x,ring,w,h);if(i===0){x.fillStyle='rgba(220,255,200,0.08)';x.fill();x.strokeStyle='rgba(255,255,190,0.9)';x.lineWidth=1.2;x.stroke();}else{x.strokeStyle='rgba(255,255,190,0.9)';x.lineWidth=1.0;x.stroke();}});};
      if(g.type==='Polygon')dr(g.coordinates);else if(g.type==='MultiPolygon')g.coordinates.forEach(p=>dr(p));
    });
    updateProgress('Loaded '+d.features.length+' countries.');
  }catch(e){console.error('Borders failed:',e);updateProgress('Borders failed.');}
  return c;
}

function mkStars(cnt=4000){
  const c=document.createElement('canvas');c.width=4096;c.height=2048;const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);
  for(let i=0;i<cnt;i++){const px=Math.random()*c.width,py=Math.random()*c.height,b=Math.random(),s=Math.random()*1.5+0.3;
  x.fillStyle=b>0.8?`rgba(180,200,255,${b})`:`rgba(255,255,255,${b*0.6})`;x.beginPath();x.arc(px,py,s,0,Math.PI*2);x.fill();
  if(b>0.9){x.fillStyle=`rgba(200,220,255,${b*0.1})`;x.beginPath();x.arc(px,py,s*4,0,Math.PI*2);x.fill();}}
  return c;
}

function mkMarker(color){
  const c=document.createElement('canvas');c.width=64;c.height=64;const x=c.getContext('2d');x.clearRect(0,0,64,64);
  const g=x.createRadialGradient(32,32,0,32,32,28);
  g.addColorStop(0,color);g.addColorStop(0.3,color+'99');g.addColorStop(1,color+'00');
  x.fillStyle=g;x.beginPath();x.arc(32,32,28,0,Math.PI*2);x.fill();
  x.fillStyle='#fff';x.beginPath();x.arc(32,32,5,0,Math.PI*2);x.fill();
  x.fillStyle=color;x.beginPath();x.arc(32,32,3,0,Math.PI*2);x.fill();
  return new THREE.CanvasTexture(c);
}

// Scene
const cont=document.getElementById('globe-container');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.1,2000);
camera.position.set(0,0,350);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setSize(window.innerWidth,window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio,3));
cont.appendChild(renderer.domElement);

const aL=new THREE.AmbientLight(0x404060,1.0);scene.add(aL);
const sL=new THREE.DirectionalLight(0xffffff,2.5);sL.position.set(200,100,200);scene.add(sL);
const fL=new THREE.DirectionalLight(0x304070,0.3);fL.position.set(-200,-50,-200);scene.add(fL);

const ER=120;
const gG=new THREE.Group();scene.add(gG);
const loader=new THREE.TextureLoader();
let ld=0;const tot=6;
function onL(n){ld++;updateProgress('Loading textures... '+ld+'/'+tot);if(ld>=tot)hidLoad();}
function onE(n,e){console.warn(n,e);ld++;if(ld>=tot)hidLoad();}
function mkT(u,ok){return loader.load(u,t=>{t.colorSpace=THREE.SRGBColorSpace;if(ok)ok(t);onL(u);},undefined,e=>onE(u,e));}

const dT=mkT(TEX_URL.day,t=>{t.anisotropy=renderer.capabilities.getMaxAnisotropy();});
const nT=mkT(TEX_URL.night);
const cT=mkT(TEX_URL.clouds);
const bT=mkT(TEX_URL.bump);
const sT=mkT(TEX_URL.spec);

const eM=new THREE.MeshPhongMaterial({map:dT,bumpMap:bT,bumpScale:0.08,specularMap:sT,specular:new THREE.Color(0x333333),shininess:25});
gG.add(new THREE.Mesh(new THREE.SphereGeometry(ER,SEG,SEG),eM));

const nlM=new THREE.MeshBasicMaterial({map:nT,transparent:true,opacity:0,blending:THREE.AdditiveBlending,depthWrite:false});
gG.add(new THREE.Mesh(new THREE.SphereGeometry(ER*1.002,SEG,SEG),nlM));

const brM=new THREE.MeshBasicMaterial({transparent:true,opacity:1,depthWrite:false,blending:THREE.NormalBlending});
gG.add(new THREE.Mesh(new THREE.SphereGeometry(ER*1.005,SEG,SEG),brM));
(async()=>{const cv=await mkBorderTex();if(cv){brM.map=new THREE.CanvasTexture(cv);brM.needsUpdate=true;}})();

const clM=new THREE.MeshPhongMaterial({map:cT,transparent:true,opacity:0.4,depthWrite:false});
const clMh=new THREE.Mesh(new THREE.SphereGeometry(ER*1.01,SEG,SEG),clM);
gG.add(clMh);

// Atmosphere
const av=`varying vec3 vN;void main(){vN=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`;
const af=`varying vec3 vN;uniform vec3 gC;void main(){float i=pow(0.65-dot(vN,vec3(0,0,1)),2.0);gl_FragColor=vec4(gC,1.0)*i*1.2;}`;
const atM=new THREE.Mesh(new THREE.SphereGeometry(ER*1.15,SEG,SEG),new THREE.ShaderMaterial({vertexShader:av,fragmentShader:af,uniforms:{gC:{value:new THREE.Color(0.2,0.5,1.0)}},side:THREE.BackSide,blending:THREE.AdditiveBlending,transparent:true}));
gG.add(atM);
gG.add(new THREE.Mesh(new THREE.SphereGeometry(ER*1.08,SEG,SEG),new THREE.MeshBasicMaterial({color:0x3388ff,transparent:true,opacity:0.05,side:THREE.FrontSide,blending:THREE.AdditiveBlending,depthWrite:false})));

// Cities
const ctG=new THREE.Group();gG.add(ctG);
const mScr={};
const ctLabels=[]; // store label sprites for dynamic zoom scaling
CITIES.forEach(c=>{
  if(c[0]==='__origin__')return;
  const[nm,lt,ln,sz,cl]=c;
  if(!mScr[cl])mScr[cl]=mkMarker(cl);
  const mat=new THREE.SpriteMaterial({map:mScr[cl],transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,depthTest:true});
  const sp=new THREE.Sprite(mat);const sc=sz*1.8;sp.scale.set(sc,sc,1);sp.position.copy(latLonToV3(lt,ln,ER*1.04));
  sp.userData={name:nm,lat:lt,lon:ln};ctG.add(sp);
  const lcv=document.createElement('canvas');lcv.width=256;lcv.height=64;const lx=lcv.getContext('2d');
  lx.clearRect(0,0,256,64);lx.font='bold 18px Arial,sans-serif';lx.fillStyle='rgba(255,255,255,0.9)';
  lx.textAlign='center';lx.textBaseline='middle';lx.fillText(nm,128,32);
  const lm=new THREE.SpriteMaterial({map:new THREE.CanvasTexture(lcv),transparent:true,depthTest:false,depthWrite:false,opacity:0.85});
  const lb=new THREE.Sprite(lm);lb.scale.set(5,1.25,1);lb.position.copy(latLonToV3(lt,ln,ER*1.04+3));ctG.add(lb);ctLabels.push(lb);
});

// Stars
const stC=mkStars(4000);const stT=new THREE.CanvasTexture(stC);stT.anisotropy=renderer.capabilities.getMaxAnisotropy();
scene.add(new THREE.Mesh(new THREE.SphereGeometry(800,64,64),new THREE.MeshBasicMaterial({map:stT,side:THREE.BackSide})));

// Controls
const ctrl=new OrbitControls(camera,renderer.domElement);
ctrl.enableDamping=true;ctrl.dampingFactor=0.08;ctrl.rotateSpeed=0.5;ctrl.zoomSpeed=0.8;
ctrl.minDistance=150;ctrl.maxDistance=500;ctrl.enablePan=false;

let autoRo=true,shCloud=true,shAtm=true,shBrD=true,shCt=true,nMde=false;

// UI
const btnRot=document.getElementById('btn-rotate');
const btnCl=document.getElementById('btn-clouds');
const btnAz=document.getElementById('btn-atmosphere');
const btnNt=document.getElementById('btn-night');
const btnCt=document.getElementById('btn-cities');

btnRot.addEventListener('click',()=>{autoRo=!autoRo;btnRot.classList.toggle('active',autoRo);});
btnCl.addEventListener('click',()=>{shCloud=!shCloud;btnCl.classList.toggle('active',shCloud);});
btnAz.addEventListener('click',()=>{shAtm=!shAtm;btnAz.classList.toggle('active',shAtm);});
btnNt.addEventListener('click',()=>{nMde=!nMde;btnNt.classList.toggle('active',nMde);});
if(btnCt)btnCt.addEventListener('click',()=>{shCt=!shCt;btnCt.classList.toggle('active',shCt);});

// Country click
const countryPanel=document.getElementById('country-panel');
const countryName=document.getElementById('country-name');
const countryInfo=document.getElementById('country-info');
const closeBtn=document.getElementById('close-country');
closeBtn.addEventListener('click',()=>{countryPanel.classList.add('hidden');});

const ray=new THREE.Raycaster();
const ms=new THREE.Vector2();
let cntryMode=false,ativCn=null,blendTo=0;

renderer.domElement.addEventListener('dblclick',(e)=>{
  ms.x=(e.clientX/window.innerWidth)*2-1;
  ms.y=-(e.clientY/window.innerHeight)*2+1;
  ray.setFromCamera(ms,camera);
  const hits=ray.intersectObjects(gG.children);
  if(hits.length>0){
    const h=hits[0];
    const invQ=new THREE.Quaternion();gG.getWorldQuaternion(invQ).invert();
    const v=new THREE.Vector3();v.copy(h.point);gG.worldToLocal(v);
    const R=v.length(),n=v.clone().normalize();
    const lt=90-Math.asin(n.y)*180/Math.PI,ln=Math.atan2(n.z,-n.x)*180/Math.PI-180;
    showCountryAt(lt,ln,Math.min(R*3,300));
    const d=findCountry(lt,ln);
    if(d){
      countryName.textContent=d.name;
      countryInfo.textContent=d.info;
      countryPanel.classList.remove('hidden');
    }
  }
});

function showCountryAt(lat,lon,dist){
  if(!cntryMode){cntryMode=true;blendTo=1;}
  const tgt=latLonToV3(lat,lon,dist);
  gsap(tgt,camera.position,tgt,new THREE.Vector3(0,0,0),80);
}

let gsapDest=null,gsapOrigin=null,gsapT=0;
function gsap(d,o,lookAt,lookOrigin,dur){
  gsapDest=d;gsapOrigin=o.clone();gsapT=0;gsapDur=dur;
}

function findCountry(lat,lon){
  let best=null,bestDist=Infinity;
  for(const[k,v]of Object.entries(COUNTRY_INFO)){
    const d=latLonDist(lat,lon,v.lat,v.lon);
    if(d<bestDist&&d<30){bestDist=d;best={name:k,info:v.info};}
  }
  return best;
}

function latLonDist(l1,ln1,l2,ln2){
  const R=6371;const dLat=(l2-l1)*Math.PI/180;const dLon=(ln2-ln1)*Math.PI/180;
  const a=Math.sin(dLat/2)**2+Math.cos(l1*Math.PI/180)*Math.cos(l2*Math.PI/180)*Math.sin(dLon/2)**2;
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}

let gsapDur=80;

// Search functionality
const searchInput=document.getElementById('search');
const searchResults=document.getElementById('search-results');
let allSearchable=[];

// Build searchable list: cities + countries
function buildSearchIndex(){
  CITIES.forEach(c=>{
    if(c[0]==='__origin__')return;
    allSearchable.push({name:c[0],lat:c[1],lon:c[2],type:'city'});
  });
  Object.keys(COUNTRY_INFO).forEach(k=>{
    if(k.startsWith('I '))return;
    allSearchable.push({name:k,lat:COUNTRY_INFO[k].lat,lon:COUNTRY_INFO[k].lon,type:'country'});
  });
  allSearchable.sort((a,b)=>a.name.localeCompare(b.name));
}
buildSearchIndex();

function searchGlobe(q){
  if(!q||q.length<2){searchResults.classList.add('hidden');return;}
  const lower=q.toLowerCase();
  const matches=allSearchable.filter(s=>s.name.toLowerCase().includes(lower)).slice(0,12);
  if(matches.length===0){searchResults.classList.add('hidden');return;}
  searchResults.innerHTML='';
  matches.forEach(m=>{
    const div=document.createElement('div');
    div.className='sr-item';
    div.innerHTML=m.name+'<span class="sr-type">'+m.type+'</span>';
    div.addEventListener('click',()=>{
      searchInput.value=m.name;
      searchResults.classList.add('hidden');
      zoomTo(m.lat,m.lon);
    });
    searchResults.appendChild(div);
  });
  searchResults.classList.remove('hidden');
}

function zoomTo(lat,lon){
  const dist=Math.max(camera.position.length(),200);
  const tgt=latLonToV3(lat,lon,dist);
  gsapDest=tgt;gsapOrigin=camera.position.clone();gsapT=0;gsapDur=80;cntryMode=true;
  // Show info panel if country
  const cf=findCountry(lat,lon);
  if(cf){countryName.textContent=cf.name;countryInfo.textContent=cf.info;countryPanel.classList.remove('hidden');}
}

searchInput.addEventListener('input',()=>searchGlobe(searchInput.value));
searchInput.addEventListener('focus',()=>searchGlobe(searchInput.value));
document.addEventListener('click',(e)=>{
  if(!e.target.closest('#search-wrap'))searchResults.classList.add('hidden');
});

// Tooltip
const tooltip=document.createElement('div');tooltip.id='tooltip';document.body.appendChild(tooltip);
let hoverObj=null;

renderer.domElement.addEventListener('mousemove',(e)=>{
  ms.x=(e.clientX/window.innerWidth)*2-1;
  ms.y=-(e.clientY/window.innerHeight)*2+1;
  ray.setFromCamera(ms,camera);
  const hits=ray.intersectObjects(ctG.children);
  if(hits.length>0&&hits[0].object.userData&&hits[0].object.userData.name){
    hoverObj=hits[0].object;tooltip.style.display='block';tooltip.style.left=(e.clientX+12)+'px';tooltip.style.top=(e.clientY-20)+'px';
    tooltip.textContent=hoverObj.userData.name;
  }else{tooltip.style.display='none';}
});

// Animation
let time=0;
function animate(){
  requestAnimationFrame(animate);time+=0.001;
  if(autoRo)gG.rotation.y+=0.002;
  clMh.rotation.y+=0.0008;
  if(nMde){nlM.opacity=Math.min(nlM.opacity+0.015,1.0);aL.intensity=Math.max(aL.intensity-0.02,0.2);}
  else{nlM.opacity=Math.max(nlM.opacity-0.015,0.0);aL.intensity=Math.min(aL.intensity+0.02,1.0);}
  atM.visible=shAtm;clMh.visible=shCloud;ctG.visible=shCt;
  
  if(cntryMode&&gsapDest){
    gsapT++;
    const t=gsapT/gsapDur;
    if(t>=1){
      camera.position.copy(gsapDest);cntryMode=false;gsapT=0;gsapDest=null;
      autoRo=false;btnRot.classList.remove('active');
    }else{
      const e=t*t*(3-2*t);
      camera.position.lerpVectors(gsapOrigin,gsapDest,e);
    }
  }

  // Dynamic label scaling: labels grow a bit more when zoomed in
  const camDist=camera.position.length();
  const zoomK=Math.min(Math.max(Math.pow(350/camDist,0.3),0.85),1.8);
  ctLabels.forEach(lb=>{lb.scale.set(5*zoomK,1.25*zoomK,1);});

  ctrl.update();renderer.render(scene,camera);
}

window.addEventListener('resize',()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});

function hidLoad(){const le=document.getElementById('loading');if(le){le.classList.add('hidden');setTimeout(()=>{le.style.display='none';},900);}}
function updateProgress(msg){const pe=document.getElementById('loading-progress');if(pe)pe.textContent=msg;}

setTimeout(()=>{if(ld<tot){updateProgress('Some textures failed.');setTimeout(hidLoad,2000);}},20000);

animate();
console.log('🌍 Earth Globe loaded successfully!');