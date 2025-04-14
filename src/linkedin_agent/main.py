from dotenv import load_dotenv
load_dotenv()

from .api import create_post
import argparse

def main():
    # Get topic from user
    stream = False
    topic = input("Enter the news topic you want to post about: ")
    
    results = create_post(topic, stream=stream)
    if stream:
        for i, result in enumerate(results[1:], 1):
            print(f"Step {i}: {result}\n")
    else:
        print("Done!")

if __name__ == "__main__":
    main()