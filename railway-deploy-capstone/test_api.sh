#!/bin/bash

# Test forecast endpoint
echo "Testing /forecast_prices/"
curl -X POST -H "Content-Type: application/json" -d '{"sku":"54321", "time_key":20240601}' http://127.0.0.1:5000/forecast_prices/
echo -e "\n"

# Test actual prices
echo "Testing /actual_prices/"
curl -X POST -H "Content-Type: application/json" -d '{"sku":"54321", "time_key":20240601, "pvp_is_competitorA_actual":85.5, "pvp_is_competitorB_actual":135.0}' http://127.0.0.1:5000/actual_prices/
echo -e "\n"

# View all data
echo "Database contents:"
curl http://127.0.0.1:5000/view_database/
echo -e "\n"