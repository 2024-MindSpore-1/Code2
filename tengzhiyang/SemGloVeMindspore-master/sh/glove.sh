#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make
<<COMMENT
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi
COMMENT

BASE=/home/ganleilei/data/wiki
CORPUS=$BASE/data.txt_plain
# CORPUS=text8
OVERFLOWFILE=$BASE/temp/overflow.word300
SHUFFLETEMPFILE=$BASE/temp/temp_shuffle.word200
VOCAB_FILE=data/vocab/vocab.wiki.word.txt
COOCCURRENCE_FILE=$BASE/bin/cooccurrence.wiki.word.dim300.bin
COOCCURRENCE_SHUF_FILE=$BASE/bin/cooccurrence.wiki.word.dim300.shuf.bin
BUILDDIR=build
SAVE_FILE=data/vectors/vectors.wiki.bpe.ws5.iter15.dim300
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=5
BINARY=0
NUM_THREADS=8
X_MAX=10


echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOWFILE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOWFILE < $CORPUS > $COOCCURRENCE_FILE

<<COMMENT
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $SHUFFLETEMPFILE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $SHUFFLETEMPFILE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ python eval/python/evaluate.py"
       python eval/python/evaluate.py
   fi
fi
COMMENT

<< COMMENT
if [ "$CORPUS" =~ 'gigaword' ]; then
    echo "$ evaluate chinese word analogy"
    python eval/python/ch_evaluate.py
fi
COMMENT
