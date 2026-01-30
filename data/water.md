# Global freshwater use over the long-run - Data package

This data package contains the data that powers the chart ["Global freshwater use over the long-run"](https://ourworldindata.org/grapher/global-freshwater-use-over-the-long-run?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on January 30, 2026.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Freshwater use
Unit: m³  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Global freshwater use since 1900 - IGB – processed by Our World in Data

#### Full citation
Global freshwater use since 1900 - IGB – processed by Our World in Data. “Freshwater use” [dataset]. Global freshwater use since 1900 - IGB [original data].
Source: Global freshwater use since 1900 - IGB – processed by Our World In Data

### Additional information about this data
Data measures global freshwater use which is the sum of water withdrawals for agriculture, industrial and domestic uses. Data from 1900-2010 is sourced from the IGB Programme (full reference below). Global data has been extended to 2014 by combining with 2014 'World' figures as reported in the World Bank - World Development Indicators, under the variable "Annual Freshwater Withdrawals, Total (billion cubic meters)". Available at: http://data.worldbank.org/data-catalog/world-development-indicators [accessed 2017-11-08].

Data from 1900-2010 is sourced from the IGB Database. IGB's data is estimated using the WaterGAP model from Flörke et al. 2013 (full reference below). Data is available at aggregates in OECD, BRICS and Rest of the World (ROW). OECD members are defined as countries who were members in 2010 and their membership was carried back in time. BRICS countries are Brazil, Russia, India, China and South Africa.

Full references:
Alcamo, J., Döll, P., Henrichs, T., Kaspar, F., Lehner, B., Rösch, T., Siebert, S., 2003. Development and testing of the WaterGAP 2 global model of water use and availability. Hydrological Sciences Journal 48:317–337.
aus der Beek, T., Flörke, M., Lapola, D. M., Schaldach, R., Voß, F., and Teichert, E. 2010. Modelling historical and current irrigation water demand on the continental scale: Europe. Advances in Geoscience 27:79-85  doi:10.5194/adgeo-27-79-2010
Flörke, M., Kynast, E., Bärlund, I., Eisner, S., Wimmer, F., Alcamo, J. 2013. Domestic and industrial water uses of the past 60 years as a mirror of socio-economic development: A global simulation study. Global Environmental Change 23: 144-156


    