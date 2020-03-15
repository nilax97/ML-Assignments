#!/bin/sh
case $1 in
  "1")
    python3 svm.py $2 $3
    ;;
  "2")
    python3 cnn.py $2 $3
    ;;
esac
