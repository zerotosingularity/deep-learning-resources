#!/bin/bash

#cd scripts
#python generate_dlr.py

./generate.sh

git status
git add .
git commit -m "update readme"
git push origin
