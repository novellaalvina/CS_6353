files="BatchNormalization.ipynb
FullyConnectedNets.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done

rm -f assignment3.zip
zip -r assignment3.zip . -x "*.git*" "*cs6353/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs6353/build/*"
