#!/bin/bash
apt-get update
apt-get install -y wget unzip xvfb libxi6 libgconf-2-4
apt-get install -y libnss3 libxss1 libappindicator1 libindicator7
apt-get install -y fonts-liberation libasound2 libatk-bridge2.0-0 libgtk-3-0
apt-get install -y chromium-browser
#!/usr/bin/env bash
set -e
python -m playwright install --with-deps chromium
