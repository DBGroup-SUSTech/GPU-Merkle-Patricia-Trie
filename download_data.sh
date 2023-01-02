#!/bin/bash -e
mkdir -p dataset && cd dataset
mkdir -p wiki && cd wiki
mkdir -p index && mkdir -p value
if [ ! -f 'value/enwiki-20230101-pages-articles-multistream1.xml-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream1.xml-p1p41242.bz2
fi
if [ ! -f 'index/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242' ];
then
  wget https://dumps.wikimedia.org/enwiki/20230101/enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
  bzip2 -d enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242.bz2
fi
mv enwiki-20230101-pages-articles-multistream-index1.txt-p1p41242 index/
mv enwiki-20230101-pages-articles-multistream1.xml-p1p41242 value/