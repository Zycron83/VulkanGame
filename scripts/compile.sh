SHADER_FOLDER="./shaders"
MARKER_FILE="$SHADER_FOLDER/.marker"

set -e

if [[ -f $MARKER_FILE ]]; then
    for file in $(find $SHADER_FOLDER -name "*.slang"); do
        if [[ $file -nt $MARKER_FILE ]]; then
            echo "Compiling: " $file
            $VULKAN_SDK/bin/slangc $file -g -target spirv -profile spirv_1_4 -o $file.spv
        fi
    done
fi
touch $MARKER_FILE

if ! [[ -d build ]]; then
    mkdir build
    cmake -B build
fi
    
cmake --build build

export LSAN_OPTIONS=suppressions=lsan_suppressions.txt
export ASAN_OPTIONS=detect_leaks=1
if [[ $1 == "--time" ]]; then
    /usr/bin/time -v ./build/VulkanGame
elif [[ $1 == "--run" ]]; then
    ./build/VulkanGame
fi


echo "Done"