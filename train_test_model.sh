#!/usr/bin/env bash
set -e

# Display help message
show_help() {
    echo "Usage: bash train_test_model.sh [OPTIONS]"
    echo "Script to train, test and evaluate the LSTM character language model"
    echo ""
    echo "Options:"
    echo "  --train           Run training (default)"
    echo "  --test            Run testing"
    echo "  --docker          Use Docker for training/testing"
    echo "  --download-data   Download example multilingual data"
    echo "  --evaluate        Evaluate predictions against answer key"
    echo "  --all             Run all steps (train, test, evaluate)"
    echo "  --help            Display this help message"
    echo "  --multi           Use multilingual test files (input_multi.txt, answer_multi.txt)"
}

# Default options
RUN_TRAIN=false
RUN_TEST=false
USE_DOCKER=false
DOWNLOAD_DATA=false
EVALUATE=false
USE_MULTI_TEST_FILES=false # Default to standard test files

# Parse command-line arguments
if [ $# -eq 0 ]; then
    RUN_TRAIN=true  # Default action is train
else
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --train)
                RUN_TRAIN=true
                shift # consume --train
                ;;
            --test)
                RUN_TEST=true
                shift # consume --test
                ;;
            --docker)
                USE_DOCKER=true
                shift # consume --docker
                ;;
            --download-data)
                DOWNLOAD_DATA=true
                shift # consume --download-data
                ;;
            --evaluate)
                EVALUATE=true
                shift # consume --evaluate
                ;;
            --all)
                RUN_TRAIN=true
                RUN_TEST=true
                EVALUATE=true
                shift # consume --all
                ;;
            --multi)
                USE_MULTI_TEST_FILES=true
                shift # consume --multi
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
fi

# Set up directories
echo "Setting up directories..."
mkdir -p data work output example # Ensure example directory exists

# Define dynamic paths based on USE_MULTI_TEST_FILES
if [ "$USE_MULTI_TEST_FILES" = true ]; then
    TEST_FILE_BASENAME="input_multi.txt"
else
    TEST_FILE_BASENAME="input.txt"
fi

TEST_DATA_PATH="example/$TEST_FILE_BASENAME"
PRED_OUTPUT_BASENAME=$(echo "$TEST_FILE_BASENAME" | sed 's/^input/pred/')
TEST_OUTPUT_PATH="output/$PRED_OUTPUT_BASENAME"
ANSWER_FILE_BASENAME=$(echo "$TEST_FILE_BASENAME" | sed 's/^input/answer/')
ANSWER_FILE_PATH="example/$ANSWER_FILE_BASENAME"

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
numpy>=1.20.0
torch>=2.0.0
tqdm>=4.64.0
EOF
fi

# Ensure dependencies are installed for both training and testing
check_dependencies() {
    echo "Checking for required dependencies..."
    
    if [ "$USE_DOCKER" = true ]; then
        echo "Using Docker - dependencies will be managed in the container"
    else
        # Check Python dependencies
        echo "Verifying Python packages..."
        python -c "
import sys
missing = []
try:
    import torch
except ImportError:
    missing.append('torch')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import tqdm
except ImportError:
    missing.append('tqdm')

if missing:
    print(f'ERROR: Missing packages: {\", \".join(missing)}')
    print('Please install required packages: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('All required packages are installed.')
    print(f'PyTorch version: {torch.__version__}')
"
        if [ $? -ne 0 ]; then
            echo "Dependency check failed. Please install required packages."
            echo "pip install -r requirements.txt"
            exit 1
        fi
    fi
}

# Download example multilingual data
if [ "$DOWNLOAD_DATA" = true ]; then
    echo "Downloading example multilingual data..."
    
    # Check if curl is installed
    if ! command -v curl &> /dev/null; then
        echo "curl is not installed. Please install it or download data manually."
        exit 1
    fi
    
    # Download English text
    echo "Downloading English sample..."
    curl -s https://www.gutenberg.org/files/1342/1342-0.txt > data/english_pride_prejudice.txt
    
    # Download Spanish text
    echo "Downloading Spanish sample..."
    curl -s https://www.gutenberg.org/files/2000/2000-0.txt > data/spanish_don_quijote.txt
    
    echo "Sample data downloaded to data/ directory"
fi

# Check dependencies before training or testing
if [ "$RUN_TRAIN" = true ] || [ "$RUN_TEST" = true ]; then
    check_dependencies
fi

# Train the model
if [ "$RUN_TRAIN" = true ]; then
    echo "Starting training..."
    if [ "$USE_DOCKER" = true ]; then
        echo "Building Docker image..."
        docker build -t cse517-proj/mylstm -f Dockerfile .
        
        echo "Training with Docker..."
        docker run --rm -v $PWD/src:/job/src -v $PWD/data:/job/data -v $PWD/work:/job/work cse517-proj/mylstm bash -c "cd /job && python src/myprogram.py train --work_dir work"
    else
        echo "Training without Docker..."
        python src/myprogram.py train --work_dir work
    fi
    echo "Training complete."
fi

# Test the model
if [ "$RUN_TEST" = true ]; then
    echo "Starting testing..."
    echo "Note: Testing requires the same dependencies as training"

    if [ ! -f "$TEST_DATA_PATH" ]; then
        echo "Error: Test data file $TEST_DATA_PATH not found."
        exit 1
    fi
    echo "Using test data: $TEST_DATA_PATH"
    echo "Predictions will be saved to: $TEST_OUTPUT_PATH"
    
    if [ "$USE_DOCKER" = true ]; then
        echo "Testing with Docker..."
        # Ensure Docker image exists
        if (! docker image inspect cse517-proj/mylstm >/dev/null 2>&1); then
            echo "Docker image not found. Building image first..."
            docker build -t cse517-proj/mylstm -f Dockerfile .
        fi
        
        docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse517-proj/mylstm bash /job/src/predict.sh /job/data/"$TEST_FILE_BASENAME" /job/output/"$PRED_OUTPUT_BASENAME"
    else
        echo "Testing without Docker..."
        # Check if model exists
        if [ ! -f work/model.pt ] && [ ! -f work/model.checkpoint ]; then
            echo "Warning: No trained model found. Did you run training first?"
            echo "The test will run but may use random predictions."
        fi
        
        python src/myprogram.py test --work_dir work --test_data "$TEST_DATA_PATH" --test_output "$TEST_OUTPUT_PATH"
    fi
    echo "Testing complete. Predictions saved to $TEST_OUTPUT_PATH"
fi

# Evaluate the predictions
if [ "$EVALUATE" = true ]; then
    echo "Evaluating predictions..."
    if [ ! -f "$TEST_OUTPUT_PATH" ]; then
        echo "Error: Prediction file $TEST_OUTPUT_PATH not found for evaluation. Run testing first."
        exit 1
    fi
    if [ ! -f "$ANSWER_FILE_PATH" ]; then
        echo "Error: Answer file $ANSWER_FILE_PATH not found for evaluation."
        exit 1
    fi
    
    echo "Evaluating $TEST_OUTPUT_PATH against $ANSWER_FILE_PATH"
    python grader/grade.py "$TEST_OUTPUT_PATH" "$ANSWER_FILE_PATH" --verbose
fi

echo "All operations completed successfully!"
