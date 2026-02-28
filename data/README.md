# Data

## Data Availability

The monitoring data used in this study are not publicly redistributable due to data-sharing agreements with the Chinese Academy of Environmental Planning. Researchers who wish to reproduce the results may request access by contacting the corresponding author.

## Required Data Structure

To run the BC-GSA pipeline, the following input files are required under `input_data/`:

```
input_data/
├── input_water_quality_data.csv    # Daily water quality (7 automated stations)
├── input_runoff_data.csv           # Daily discharge (21 hydrological stations)
├── input_metro_data.csv            # Daily meteorological data (8 stations)
└── point_source/
    ├── input_source.csv                        # Daily PS effluent loads 
    └── station_distance_matrix_directed.csv    # Directed distance matrix (stations × PS)
```

### File Formats

**input_water_quality_data.csv**
| Column | Description | Unit |
|--------|-------------|------|
| Date | Sampling date (YYYY-MM-DD) | - |
| Station | Station identifier | - |
| AmmoniaNitrogen | NH₃-N concentration | mg/L |
| TotalPhosphorus | TP concentration | mg/L |
| TotalNitrogen | TN concentration | mg/L |

**input_runoff_data.csv**
| Column | Description | Unit |
|--------|-------------|------|
| Date | Date (YYYY-MM-DD) | - |
| {StationName} | Daily mean discharge | m³/s |

**input_metro_data.csv**
| Column | Description | Unit |
|--------|-------------|------|
| Date | Date (YYYY-MM-DD) | - |
| Station | Weather station identifier | - |
| Precipitation | Daily precipitation | mm |
| Temperature | Daily mean temperature | °C |

**input_source.csv**
| Column | Description | Unit |
|--------|-------------|------|
| Date | Date (YYYY-MM-DD) | - |
| SourceID | Point source identifier (YL10001–YL10020) | - |
| NH4N | NH₃-N effluent load | kg/d |
| TP | TP effluent load | kg/d |
| TN | TN effluent load | kg/d |

## Station Information

Detailed station metadata (coordinates, drainage areas, river assignments) are defined in `bcgsa/config.py`. The study area encompasses:

- **7 automated water quality stations** on the Yi River, Luo River, and Yiluo River
- **21 hydrological gauging stations** (including 6 diversion canals)
- **8 national meteorological stations**
- **major industrial/municipal point sources**

See Tables S1–S3 in the Supplementary Information for complete station details.
