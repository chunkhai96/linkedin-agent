from dotenv import load_dotenv
load_dotenv()

from .api import create_post, research_post
import argparse

def main():
    stream = False

    # Get topic from user
    # topic = input("Enter the news topic you want to post about: ")
    # results = create_post(topic, stream=stream)

    results = research_post(stream=stream)

    if stream:
        for i, result in enumerate(results[1:], 1):
            print(f"Step {i}: {result}\n")
    else:
        print("Done!")

if __name__ == "__main__":
    main()