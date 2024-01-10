#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

echo "cooccurrence file $1"
echo "cooccurrence shuf file $1.shuf"
echo "save file $2"
BASE=/home/ganleilei/data/wiki
OVERFLOWFILE=$BASE/temp/overflow.word300
SHUFFLETEMPFILE=$BASE/temp/temp_shuffle.word200
VOCAB_FILE=data/vocab/vocab.wiki.word.txt
COOCCURRENCE_FILE=$1
COOCCURRENCE_SHUF_FILE=$1.shuf
BUILDDIR=build
SAVE_FILE=$2
VERBOSE=2
MEMORY=4.0
VECTOR_SIZE=300
MAX_ITER=15
BINARY=0
NUM_THREADS=8
X_MAX=10

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $SHUFFLETEMPFILE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $SHUFFLETEMPFILE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
