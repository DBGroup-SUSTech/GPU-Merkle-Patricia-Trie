#!/bin/bash -e
# WIKI
PROJECT_DIR=`pwd`
mkdir -p dataset && cd dataset
mkdir -p wiki && cd wiki
mkdir -p index && mkdir -p value
if [ ! -f 'value/enwiki-20230101-pages-articles-multistream1.xml-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
  mv enwiki-20230101-pages-articles-multistream1.xml-p1p41242 value/
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
  mv enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242 index/
fi
cd $PROJECT_DIR

# YCSB
cd YCSB-C && make
cd $PROJECT_DIR
mkdir -p dataset && cd dataset
mkdir -p ycsb && cd ycsb
$PROJECT_DIR/YCSB-C/ycsbc -db basic -threads 4 -P $PROJECT_DIR/YCSB-C/workloads/workloada.spec > workloada.txt

