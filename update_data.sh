#!/bin/bash
echo "Download begins..."
curl https://services.swpc.noaa.gov/json/planetary_k_index_1m.json -o data/planetary_k_index.json
echo "planetary_k_index.json done"
curl https://services.swpc.noaa.gov/json/boulder_k_index_1m.json -o data/boulder_k_index.json 
echo "boulder_k_index.json done" 
curl https://services.swpc.noaa.gov/json/45-day-forecast.json -o data/45-day-forecast.json
echo "All done"
curl -L "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json" -o data/solar_flares.json
echo "solar_flares.json done"
echo "All done!"
