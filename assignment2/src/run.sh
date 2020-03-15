
cd data_extract/
export CLASSPATH=$HOME/Desktop/Demo/Packages/stanford-corenlp-3.9.2.jar
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
python make_datafiles.py demo_data/class/ demo_data/dm/
cd finished_files/
tar -xf val.tar
cd ../../fast_abs_rl-master/
export DATA=$HOME/Desktop/Demo/data_extract/finished_files/
python decode_full_model.py --path=output/ --model=pretrained/model/ --beam=5 --val
