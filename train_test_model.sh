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
}

# Default options
RUN_TRAIN=false
RUN_TEST=false
USE_DOCKER=false
DOWNLOAD_DATA=false
EVALUATE=false

# Parse command-line arguments
if [ $# -eq 0 ]; then
    RUN_TRAIN=true  # Default action is train
else
    for arg in "$@"; do
        case $arg in
            --train)
                RUN_TRAIN=true
                shift
                ;;
            --test)
                RUN_TEST=true
                shift
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --download-data)
                DOWNLOAD_DATA=true
                shift
                ;;
            --evaluate)
                EVALUATE=true
                shift
                ;;
            --all)
                RUN_TRAIN=true
                RUN_TEST=true
                EVALUATE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $arg"
                show_help
                exit 1
                ;;
        esac
    done
fi

# Set up directories
echo "Setting up directories..."
mkdir -p data work output

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
    
    if [ "$USE_DOCKER" = true ]; then
        echo "Testing with Docker..."
        # Ensure Docker image exists
        if ! docker image inspect cse517-proj/mylstm >/dev/null 2>&1; then
            echo "Docker image not found. Building image first..."
            docker build -t cse517-proj/mylstm -f Dockerfile .
        fi
        
        docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse517-proj/mylstm bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
    else
        echo "Testing without Docker..."
        # Check if model exists
        if [ ! -f work/model.pt ] && [ ! -f work/model.checkpoint ]; then
            echo "Warning: No trained model found. Did you run training first?"
            echo "The test will run but may use random predictions."
        fi
        
        python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output output/pred.txt
    fi
    echo "Testing complete. Predictions saved to output/pred.txt"
fi

# Evaluate the predictions
if [ "$EVALUATE" = true ]; then
    echo "Evaluating predictions..."
    if [ -f output/pred.txt ] && [ -f example/answer.txt ]; then
        python grader/grade.py output/pred.txt example/answer.txt --verbose
    else
        echo "Error: Missing prediction or answer files for evaluation"
        exit 1
    fi
fi

echo "All operations completed successfully!"
