#!/usr/bin/env bash

FILE="ted_talks.tar.gz"
wget https://www.dropbox.com/s/ah6x2ni3ev0i2lk/${FILE}
tar -xvf ${FILE} -C data/ && rm ${FILE}