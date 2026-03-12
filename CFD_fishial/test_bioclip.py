import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from classifier_tester import ClassifierTester

def main():
    print("Testing bioclip-2.5-vith14...")
    tester = ClassifierTester(model_choice="bioclip-2.5-vith14")
    
    print("Loaded successfully!")
    tester.cleanup()

if __name__ == "__main__":
    main()
