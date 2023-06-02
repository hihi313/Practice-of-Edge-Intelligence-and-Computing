# Enable tracing
set -x

INPUT_FILE=$(basename -- "${1}")
FILENAME="${INPUT_FILE%.*}"
OUTPUT_DIR="./models/${FILENAME}"

# If output dir is not empty
if ! [ -z "$2" ]
then
    OUTPUT_DIR=$2
fi

mo --input_model "${1}" \
    --compress_to_fp16 \
    --output_dir "${OUTPUT_DIR}"

# Disable tracing
set +x