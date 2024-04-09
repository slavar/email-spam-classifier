#!/bin/sh
export FLASK_APP=./app/index.py
pipenv run flask --debug run
