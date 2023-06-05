# Enable tracing
set -x

INPUT_FILE=$(basename -- "${1}")
FILENAME="${INPUT_FILE%.*}"
OUTPUT_DIR="./models/${FILENAME}"

# Color
RED='\033[0;31m'
NC='\033[0m' # No Color

# If output dir is not empty
if ! [ -z "$2" ]
then
    OUTPUT_DIR=$2
fi

if [[ $1 == *.onnx ]]
then
    mo --input_model "${1}" \
        --compress_to_fp16 \
        --output_dir "${OUTPUT_DIR}"
else
    printf "${RED}input must be onnx file${NC}\n"
fi

# Disable tracing
set +x