#!/bin/bash

cd scripts
python generate_dlr.py

cd ..
git add .
git commit -m "update readme"
git push origin

