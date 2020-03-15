#!/bin/sh
case $1 in
  "1")
    python3 1_part.py $2 $3 $4 $5
    ;;
  "2")
    python3 2_part.py $2 $3 $4
    ;;
  "3")
    python3 3_part.py $2 $3
    ;;
  "4")
    python3 4_part.py $2 $3 $4
    ;;
esac
