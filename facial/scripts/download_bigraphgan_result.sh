FILE=$1

echo "Note: available results are facial"
echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/results/BiGraphGAN/${FILE}_results.tar.gz
TAR_FILE=../results_by_author/${FILE}_results.tar.gz
TARGET_DIR=../results_by_author/${FILE}_results/

wget -N $URL -O $TAR_FILE

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ../results_by_author/
rm $TAR_FILE
