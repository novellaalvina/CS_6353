rm -f assignment2.zip 
zip -r assignment2.zip . -x "*.git*" "*cs6353/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*"
